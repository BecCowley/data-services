#!/usr/bin/python3
import argparse
import os
import sys
import tempfile
import re
from datetime import datetime
from netCDF4 import Dataset, date2num
import numpy as np
import numpy.ma as ma
import pandas as pd
import difflib

from imos_logging import IMOSLogging
from ship_callsign import ship_callsign_list
from xbt_line_vocab import xbt_line_info
from generate_netcdf_att import generate_netcdf_att, get_imos_parameter_info
from configparser import ConfigParser

# from local directory
from xbt_utils import _error, invalid_to_ma_array, decode_bytearray, temp_prof_info


class XbtProfile(object):
    """ Main class to parse a Mquest format netcdf file

    Input:
        fid file object of an open Mquest netcdf file.
        May be a *.raw or a *.ed file

    Output:
        Each time this class is initialised it reads
        the profile from the input file. The data are
        parsed into a set of dictionaries and lists.
        Functions are defined to convert some of the
        commonly used information from these.

        Example:
            fid = open("CSIROXBT2019/88/89/34/15ed.nc")
            profile = xbt_profile(fid) # Reads the profile and metadata.
            profile.latitude()  # Return the latitude of the profile.
            profile.z()         # Return the depths of the observations.
            fid.close()
    """

    def __init__(self, file_path_name):
        """ Read XBT files written in an un-friendly NetCDF format
        global attributes, data and annex information are added to the object
        """
        # record the file name
        self.XBT_input_filename = file_path_name

        # now read the data and metadata from the file
        LOGGER.info('Parsing %s' % self.XBT_input_filename)
        self.netcdf_file_obj = Dataset(file_path_name, 'r', format='NETCDF4')


class XbtKeys(object):
    """Class to parse an Mquest format *keys.nc netcdf file in preparation for reading and
    converting individual profile files

     Input:
         filename of an Mquest database.
         May be just the database name (eg: CSIROXBT2019) or
         may include the *keys.nc extension (eg: CSIROXBT2019_keys.nc)

     Output:
         Each time this class is initialised it reads
         the data from the database keys file. The station numbers are
         parsed into a list.

         Example:
             filename = "CSIROXBT2019_keys.nc"
             keysfile = xbt_keys(filename) # Reads the keys file.
             keysfile.uniqueid()  # Return the unique ids of all profiles in the database.
             keysfile.latitude()  # Return the latitudes of the profiles.
     """

    def __init__(self, filename):
        # record the keys file name and database filename
        if filename.input_xbt_campaign_path.endswith('_keys.nc'):
            self.keys_file_path = filename.input_xbt_campaign_path
            self.dbase_name = self.keys_file_path.replace('_keys.nc', '')
        else:
            self.dbase_name = filename.input_xbt_campaign_path
            self.keys_file_path \
                = '{campaign_path}_keys.nc'.format(campaign_path=filename.input_xbt_campaign_path.rstrip(os.path.sep))

        if not os.path.exists(self.keys_file_path):
            msg = '{keys_file_path} does not exist%s\nProcess aborted'.format(keys_file_path=self.keys_file_path)
            print(msg, file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(self.dbase_name):
            msg = '{dbpath} does not exist%s\nProcess aborted'.format(dbpath=self.dbase_name)
            print(msg, file=sys.stderr)
            sys.exit(1)

        # now get the station numbers from the keys file
        with Dataset(self.keys_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
            station_number = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in
                              netcdf_file_obj['stn_num'][:].data if bytearray(xx).strip()]
            # change station number to a numpy array
            station_number = np.asarray(station_number, dtype=np.int32)
            # sort it and keep unique station numbers where sometimes the keys has multiple values
            station_number, istn = np.unique(station_number, return_index=True)

            # read in the position information
            latitude = np.round(netcdf_file_obj['obslat'][:].data,4)
            longitude = np.round(netcdf_file_obj['obslng'][:].data,4)
            # sort them as per the station number
            latitude = latitude[istn]
            longitude = longitude[istn]
            # decode date/time information

            # callsign
            calls = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['callsign'][:].data
                     if bytearray(xx).strip()]
            # sort the same as station number
            calls = [x for _, x in sorted(zip(istn, calls))]

            self.data = {}
            self.data = {'station_number': [int(x) for x in station_number], 'latitude': [x for x in latitude],
                         'longitude': [x for x in longitude], 'callsign': [x for x in calls]}


def coordinate_data(profile_qc, profile_noqc, profile_raw):
    # perform checks and adjustments and combine data in preparation for writing out
    profile_qc, profile_noqc = parse_data_nc(profile_qc, profile_noqc, profile_raw)

    # get global attributes
    profile_qc = parse_globalatts_nc(profile_qc)
    profile_noqc = parse_globalatts_nc(profile_noqc)

    # assign the geospatial_vertical* to the no_qc file for checking consistency. Doesn't get assigned in previous call
    # because the data doesn't get written to the noqc profile
    profile_noqc.global_atts['geospatial_vertical_max'] = max(profile_qc.data['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_vertical_min'] = min(profile_qc.data['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_lat_max'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lat_min'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_max'] = profile_qc.data['LONGITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_min'] = profile_qc.data['LONGITUDE_RAW']

    # let's check if there are histories to parse and then handle
    profile_qc = parse_histories_nc(profile_qc)
    if int(profile_noqc.netcdf_file_obj['Num_Hists'][0].data) == 0:
        profile_noqc.histories = []
    else:
        # we need to carry the depths information into the history parsing, so copy the data array into profile_noqc
        profile_noqc.data = dict()
        profile_noqc.data['DEPTH'] = profile_qc.data['DEPTH_RAW']
        profile_noqc.data['TEMP_quality_control'] = profile_qc.data['TEMP_quality_control']
        profile_noqc = parse_histories_nc(profile_noqc)
        # check for histories in the noqc file and reconcile:
        if len(profile_noqc.histories) > 0:
            # TODO: figure out a handling here if there are extra histories in the RAW file or ones that aren't in ED file
            # reconcile histories where they exist in the noqc profile
            profile_qc = combine_histories(profile_qc, profile_noqc)

    # next section, only if there are QC flags present
    if len(profile_qc.histories) > 0:
        # handle special case of premature launch where raw and edited files have different profile lengths:
        profile_qc = check_for_PL_flag(profile_qc)

        # replace missing temperature data with actual values and appropriate QC flags
        # applies to CS flag in particular
        profile_qc = restore_temp_val(profile_qc)

        # adjust lat lon qc flags if required
        profile_qc = adjust_position_qc_flags(profile_qc)
        # adjust date and time QC flags if required
        profile_qc = adjust_time_qc_flags(profile_qc)

        # perform a check of the qc vs noqc global attributes and histories. Do any of these need reconciling?
        if len(profile_qc.global_atts.keys() - profile_noqc.global_atts):
            LOGGER.error('%s GLOBAL attributes in RAW and ED files are not consistent'
                         % profile_qc.XBT_input_filename)
            exit(1)

    # now, lets re-map these data code (QC reasons) and create the flag_and_feature type variable:
    profile_qc = create_flag_feature(profile_qc)

    # Probe type goes into a variable with coefficients as attributes, and assign QC to probe types
    profile_qc = get_fallrate_eq_coef(profile_qc, profile_noqc)

    # add uncertainties:
    profile_qc = add_uncertainties(profile_qc)

    return profile_qc


def get_recorder_type(profile):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')
    syst_list = read_section_from_xbt_config('SYST')

    att_name = 'XBT_recorder_type'
    if att_name in list(profile.global_atts.keys()):
        item_val = str(int(profile.global_atts[att_name]))
#        if item_val in list(syst_list.keys()):
#            item_val = syst_list[item_val].split(',')[0]

        if item_val in list(rct_list.keys()):
            return item_val, rct_list[item_val].split(',')[0]
        else:
            _error('{item_val} missing from recorder type part in xbt_config file'.format(item_val=item_val))
    else:
        _error('XBT_recorder_type missing from {input_nc_path}'.format(input_nc_path=profile.XBT_input_filename))


def parse_globalatts_nc(profile):
    """
    retrieve global attributes from input NetCDF file object
    """
    profile.global_atts = dict()

    # voyage/cruise identifier
    profile.global_atts['XBT_cruise_ID'] = decode_bytearray(
        profile.netcdf_file_obj.variables['Cruise_ID'][:]).strip()
    # which node the data entered into the GTS
    profile.global_atts['gts_insertion_node'] = \
        decode_bytearray(profile.netcdf_file_obj['Source_ID'][:]).replace('\x00', '').strip()
    # source_id = 'AMMC' if source_id == '' else source_id
    # these two variable are dimensioned by nprof
    profile.global_atts['digitisation_method_code'] = np.empty(profile.nprof)
    profile.global_atts['gtspp_precision_code'] = np.empty(profile.nprof)
    for count in range(profile.nprof):
        try:
            profile.global_atts['digitisation_method_code'][count] = \
                decode_bytearray(profile.netcdf_file_obj['Digit_Code'][count]).replace('\x00', '').strip()
            profile.global_atts['gtspp_precision_code'][count] \
                = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['Standard'][count].data)).replace('\x00', '').strip()
        except:
            profile.global_atts['digitisation_method_code'][count] = np.nan
            profile.global_atts['gtspp_precision_code'][count] = np.nan
    try:
        profile.global_atts['predrop_comments'] \
            = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['PreDropComments'][:].data)).replace(
            '\x00', '').strip()
        profile.global_atts['postdrop_comments'] \
            = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['PostDropComments'][:].data)).replace(
            '\x00', '').strip()
    except:
        profile.global_atts['predrop_comments'] = ''
        profile.global_atts['postdrop_comments'] = ''

    profile.global_atts['geospatial_vertical_units'] = 'meters'
    profile.global_atts['geospatial_vertical_positive'] = 'down'

    try:
        profile.global_atts['geospatial_lat_max'] = profile.data['LATITUDE']
        profile.global_atts['geospatial_lat_min'] = profile.data['LATITUDE']
        profile.global_atts['geospatial_lon_max'] = profile.data['LONGITUDE']
        profile.global_atts['geospatial_lon_min'] = profile.data['LONGITUDE']
        profile.global_atts['geospatial_vertical_max'] = max(profile.data['DEPTH'])
        profile.global_atts['geospatial_vertical_min'] = min(profile.data['DEPTH'])
    except:
        profile.global_atts['geospatial_lat_max'] = []
        profile.global_atts['geospatial_lat_min'] = []
        profile.global_atts['geospatial_lon_max'] = []
        profile.global_atts['geospatial_lon_min'] = []
        profile.global_atts['geospatial_vertical_max'] = []
        profile.global_atts['geospatial_vertical_min'] = []


    profile.global_atts['date_created'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Parse the surface codes into the global attributes too
    srfc_code_nc = profile.netcdf_file_obj['SRFC_Code'][:]
    srfc_parm = profile.netcdf_file_obj['SRFC_Parm'][:]
    nsrf_codes = int(profile.netcdf_file_obj['Nsurfc'][:])

    srfc_code_list = read_section_from_xbt_config('SRFC_CODES')

    # read a list of srfc code defined in the srfc_code conf file. Create a
    # dictionary of matching values
    for i in range(nsrf_codes):
        srfc_code_iter = decode_bytearray(srfc_code_nc[i])
        if srfc_code_iter in list(srfc_code_list.keys()):
            att_name = srfc_code_list[srfc_code_iter].split(',')[0]
            att_type = srfc_code_list[srfc_code_iter].split(',')[1]
            att_val = decode_bytearray(srfc_parm[i])
            profile.global_atts[att_name] = att_val
            try:
                if att_type == 'float':
                    profile.global_atts[att_name] = float(profile.global_atts[att_name].replace(' ', ''))
                elif att_type == 'int':
                    profile.global_atts[att_name] = int(profile.global_atts[att_name].replace(' ', ''))
                else:
                    profile.global_atts[att_name] = profile.global_atts[att_name].replace(' ', '')
            except ValueError:
                LOGGER.warning(
                    '"%s = %s" could not be converted to %s()' % (att_name, profile.global_atts[att_name],
                                                                  att_type.upper()))
        else:
            if srfc_code_iter != '':
                LOGGER.warning('%s code is not defined in srfc_code in xbt_config file. Please edit xbt_config'
                               % srfc_code_iter)

    # if the platform code didn't come through, need to stop
    if 'Platform_code' not in profile.global_atts.keys():
        LOGGER.error('Platform_code is missing, GCLL has not been read or is missing')
        breakpoint()

    # get the ship details
    # note that the callsign and ship name are filled from the original file values, but will be replaced here if they exist in the AODN vocabulary
    # for these older historical files, the Callsign and Platform_code are the same. In newer files, the platform_code
    # will be the GTSID or SOTID.
    profile.global_atts['Callsign'] = profile.global_atts['Platform_code'] # set here as can't have duplicate assignments in the config file
    ships = SHIP_CALL_SIGN_LIST
    if profile.global_atts['Platform_code'] in ships:
        profile.global_atts['ship_name'] = ships[profile.global_atts['Platform_code']]
    elif difflib.get_close_matches(profile.global_atts['Platform_code'], ships, n=1, cutoff=0.8) != []:
        profile.global_atts['Callsign'] = \
            difflib.get_close_matches(profile.global_atts['Platform_code'], ships, n=1, cutoff=0.8)[0]
        profile.global_atts['ship_name'] = ships[profile.global_atts['Callsign']]
        LOGGER.warning('Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s' % (
            profile.global_atts['Platform_code'], profile.global_atts['Callsign']))
    else:
        LOGGER.warning('Vessel call sign %s is unknown in AODN vocabulary, Please contact info@aodn.org.au' %
                       profile.global_atts['Platform_code'])

    # extract the information and assign correctly
    att_name = 'XBT_recorder_type'
    if att_name in list(profile.global_atts):
        recorder_val, recorder_type = get_recorder_type(profile)
        profile.global_atts['XBT_recorder_type'] = recorder_val + ', ' + recorder_type

    att_name = 'XBT_height_launch_above_water_in_meters'
    if att_name in list(profile.global_atts.keys()):
        if profile.global_atts[att_name] > 50:
            LOGGER.warning('HTL$, xbt launch height attribute seems to be very high: %s meters' %
                           profile.global_atts[att_name])

    # get xbt line information from config file
    xbt_config = _call_parser('xbt_config')
    # some files don't have line information
    isline = profile.global_atts.get('XBT_line')
    if not isline:
        profile.global_atts['XBT_line'] = 'NOLINE'
        # TODO: need to allow the user to assign a line to this cruise ID. Need to retain this information and apply to
        # all the profiles with this cruise ID.

    xbt_line_conf_section = [s for s in xbt_config.sections() if profile.global_atts['XBT_line'] in s]
    xbt_alt_codes = [s for s in list(XBT_LINE_INFO.keys()) if
                     XBT_LINE_INFO[s] is not None]  # alternative IMOS codes taken from vocabulary
    if xbt_line_conf_section:
        xbt_line_att = dict(xbt_config.items(xbt_line_conf_section[0]))
        profile.global_atts.update(xbt_line_att)
    elif profile.global_atts['XBT_line'] in xbt_alt_codes:
        xbt_line_conf_section = [s for s in xbt_config.sections()
                                 if XBT_LINE_INFO[profile.global_atts['XBT_line']] == s]
        xbt_line_att = dict(xbt_config.items(xbt_line_conf_section[0]))
        profile.global_atts.update(xbt_line_att)
    else:
        LOGGER.error(
            'XBT line : "%s" is not defined in conf file(Please edit), or an alternative code has to be set up ' %
            'by AODN in vocabs.ands.org.au(contact AODN)' %
            profile.global_atts['XBT_line'])
        exit(1)

    return profile


def parse_data_nc(profile_qc, profile_noqc, profile_raw):
    """ Parse variable data from all sources into a dictionary attached to the profile_qc structure
    """
    profile_qc.data = dict()

    # Location information
    profile_qc.data['LATITUDE'] = np.round(profile_qc.netcdf_file_obj['latitude'][0].__float__(), 4)
    profile_qc.data['LATITUDE_RAW'] = np.round(profile_noqc.netcdf_file_obj['latitude'][0].__float__(), 4)

    # check if scale factor has been applied, shouldn't have a negative longitude:
    lon = profile_noqc.netcdf_file_obj['longitude'][0].__float__()
    if lon < 0:
        if profile_qc.netcdf_file_obj['longitude'].scale:
            LOGGER.info('Scale Factor in ed file longitude attributes, changing longitude value from  %s' % lon)
            lon = lon * -1
        else:
            LOGGER.error('Negative longitude value with no scale factor %s' % lon)
            exit(1)
    profile_qc.data['LONGITUDE'] = np.round(lon, 4)
    profile_qc.data['LONGITUDE_RAW'] = np.round(profile_noqc.netcdf_file_obj['longitude'][0].__float__(), 4)

    # position and time QC - check this is not empty. Assume 1 if it is
    q_pos = int(profile_qc.netcdf_file_obj['Q_Pos'][0].data)
    if not q_pos:
        LOGGER.info('Missing position QC, flagging position with flag 1 %s' % profile_qc.XBT_input_filename)
        q_pos = 1
    profile_qc.data['LATITUDE_quality_control'] = q_pos
    profile_qc.data['LONGITUDE_quality_control'] = q_pos

    # Date time information
    woce_date = profile_qc.netcdf_file_obj['woce_date'][0]
    woce_time = profile_qc.netcdf_file_obj['woce_time'][0]

    # AW Add Original date_time from the raw .nc - date-time could be changed thru QC
    woce_date_raw = profile_noqc.netcdf_file_obj['woce_date'][0]
    woce_time_raw = profile_noqc.netcdf_file_obj['woce_time'][0]

    q_date_time = int(profile_qc.netcdf_file_obj['Q_Date_Time'][0])
    if not q_date_time:
        LOGGER.info('Missing time QC, flagging time with flag 1 %s' % profile_qc.XBT_input_filename)
        q_date_time = 1

    # need to be a bit more specific as some times have missing padding at the end, some at the start.
    # could break if hour is 00 and there are no zeros!
    # Let's try padding left and right, then convert to time for both
    rpad = str(woce_time).ljust(6, '0')
    lpad = str(woce_time).zfill(6)

    try:
        # insert zeros into dates with spaces
        xbt_date = '%sT%s' % (woce_date, rpad)
        str1 = [x.replace(' ', '0') for x in xbt_date]
        xbt_date = ''.join(str1)
        xbt_date = datetime.strptime(xbt_date, '%Y%m%dT%H%M%S')
    except:
        xbt_date = '%sT%s' % (woce_date, lpad)
        str1 = [x.replace(' ', '0') for x in xbt_date]
        xbt_date = ''.join(str1)
        xbt_date = datetime.strptime(xbt_date, '%Y%m%dT%H%M%S')

    # Raw date
    rpad = str(woce_time_raw).ljust(6, '0')
    lpad = str(woce_time_raw).zfill(6)

    try:
        # insert zeros into dates with spaces
        xbt_date_raw = '%sT%s' % (woce_date_raw, rpad)
        str1 = [x.replace(' ', '0') for x in xbt_date_raw]
        xbt_date_raw = ''.join(str1)
        xbt_date_raw = datetime.strptime(xbt_date_raw, '%Y%m%dT%H%M%S')
    except:
        xbt_date_raw = '%sT%s' % (woce_date_raw, lpad)
        str1 = [x.replace(' ', '0') for x in xbt_date_raw]
        xbt_date_raw = ''.join(str1)
        xbt_date_raw = datetime.strptime(xbt_date_raw, '%Y%m%dT%H%M%S')


    # AW - TIME_RAW is original date-time - set it too
    profile_qc.data['TIME'] = xbt_date
    profile_qc.data['TIME_quality_control'] = q_date_time
    profile_qc.data['TIME_RAW'] = xbt_date_raw

    # Pressure/depth information from both noqc and qc files
    # read into a dataframe
    df = pd.DataFrame()
    for s in [profile_qc, profile_noqc]:
        # cycle through the variables identified in the file:
        data_vars = temp_prof_info(s.netcdf_file_obj)
        for ivar, var in data_vars.items():
            if s is profile_noqc:
                var = var + '_RAW'
            # we want the DEPTH to be a single dataset, but read all depths for each variable
            if 'D' in decode_bytearray(s.netcdf_file_obj.variables['D_P_Code'][ivar]):
                depcode = 'depth'
            else:
                depcode = 'press'
            df[var + depcode] = np.round(s.netcdf_file_obj.variables['Depthpress'][ivar, :], 2)
            depth_press_flag = s.netcdf_file_obj.variables['DepresQ'][ivar, :, 0].flatten()
            df[var + depcode + '_quality_control'] = np.ma.masked_array(invalid_to_ma_array(depth_press_flag, fillvalue=0))

            prof = np.ma.masked_values(
                np.round(s.netcdf_file_obj.variables['Profparm'][ivar, 0, :, 0, 0],2), 99.99) #mask the 99.99 from CSA flagging of TEMP
            prof = np.ma.masked_invalid(prof) # mask nan and inf values
            prof.set_fill_value(-99.99)

            prof_flag = s.netcdf_file_obj.variables['ProfQP'][ivar, 0, :, 0, 0].flatten()
            prof_flag = np.ma.masked_array(
                invalid_to_ma_array(prof_flag, fillvalue=99))  # replace masked values for IMOS IODE flags
            df[var] = prof
            df[var + '_quality_control'] = prof_flag

    # check the depth columns for consistency and remove redundant ones
    dep_cols = [col for col in df.columns if 'depth' in col
                and not ('TEMPdepth' in col or 'TEMP_RAWdepth' in col or '_quality_control' in col)]

    for dat in dep_cols:
        if df['TEMPdepth'].equals(df[dat]):
            # delete the column
            df = df.drop(dat, axis=1)
            continue
        else:
            LOGGER.error('%s does not match depths for TEMP depth' % dat)
            # TODO: handle these problems as they arise here
            break

    # check we only have two depth columns left
    dep_cols = [col for col in df.columns if 'depth' in col
                and not ('TEMPdepth' in col or 'TEMP_RAWdepth' in col or '_quality_control' in col)]
    if dep_cols:
        LOGGER.error('Still multiple depth variables that need resolving, debug!!')
        breakpoint()
        # TODO: handle these problems as they arise here

    # rename and tidy
    # TODO: Check the salinity and conductivity variables when we get a profile with them in
    dd = {"TEMPdepth": "DEPTH",
           "TEMP_RAWdepth": "DEPTH_RAW",
           "SVEL": "SSPD",
           "SALT": "PSAL"
           }
    for key, val in dd.items():
        df.columns = df.columns.str.replace(key, val)

    # if we have other variables, there will be *depth_quality_control data left, let's remove it
    irem = [col for col in df.columns if 'depth' in col]
    df = df.drop(irem, axis=1)

    # drop rows where all NaN values which does happen in these old files sometimes
    df = df.dropna(how='all')

    # how many parameters do we have, not including DEPTH?
    profile_qc.nprof = len([col for col in df.columns if ('_quality_control' not in col and 'RAW'
                                                          not in col and 'DEPTH' not in col)])
    profile_noqc.nprof = profile_qc.nprof

    # let's write these out to the profile_qc in the appropriate format to suit the rest of the code
    for var in df.columns:
        profile_qc.data[var] = df[var].to_numpy()

    return profile_qc, profile_noqc


def adjust_position_qc_flags(profile):
    """ When a 'PE' flag is present in the Act_Code, the latitude and longitude qc flags need to be adjusted if not
    already set (applies to data processed with older versions of MQUEST
    Also, if the temperature QC flags are not set correctly (3 for PER, 2 for PEA), these should be updated.
    """

    # exit this if we don't have a position code
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("LA|LO|PER")]) == 0:
        return profile

    # get the temperature QC codes
    tempqc = profile.data['TEMP_quality_control']
    if profile.histories['HISTORY_QC_CODE'].str.contains('LA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if np.round(float(profile.histories.loc[
                        profile.histories['HISTORY_QC_CODE'].str.contains('LA'), 'HISTORY_PREVIOUS_VALUE'].values),
                    4) != np.round(profile.data['LATITUDE_RAW'], 4):
            LOGGER.error('LATITUDE_RAW not the same as the PREVIOUS_value!')
            exit(1)
        if profile.data['LATITUDE_quality_control'] != 5:
            # PEA on latitude
            profile.data['LATITUDE_quality_control'] = 5
            LOGGER.info('LATITUDE correction (PEA) in original file, changing LATITUDE flag to level 5.')
            # change to flag 2 for temperature for all depths where qc is less than 2
            tempqc[tempqc < 2] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('LO').any():
        # check HISTORY_PREVIOUS_VALUE matches the LONGITUDE_RAW value
        if np.round(float(profile.histories.loc[
                        profile.histories['HISTORY_QC_CODE'].str.contains('LO'), 'HISTORY_PREVIOUS_VALUE'].values),
                    4) != np.round(profile.data['LONGITUDE_RAW'], 4):
            LOGGER.error('LONGITUDE_RAW not the same as the PREVIOUS_value!')
            exit(1)
        if profile.data['LONGITUDE_quality_control'] != 5:
            # PEA on longitude
            profile.data['LONGITUDE_quality_control'] = 5
            LOGGER.info('LONGITUDE correction (PEA) in original file, changing LONGITUDE flag to level 5.')
            # change to flag 2 for temperature for all depths where qc is less than 2
            tempqc[tempqc < 2] = 2


    if profile.histories['HISTORY_QC_CODE'].str.contains('PER').any():
        # PER on longitude and latitude
        profile.data['LONGITUDE_quality_control'] = 3
        profile.data['LATITUDE_quality_control'] = 3
        LOGGER.info('Position Reject (PER) in original file, changing LONGITUDE & LATITUDE flags to level 3.')
        # change to flag 3 for temperature for all depths where qc is less than 3
        tempqc[tempqc < 3] = 3

    # update the temperature QC flags
    profile.data['TEMP_quality_control'] = tempqc

    return profile


def adjust_time_qc_flags(profile):
    """ When a 'TE' flag is present in the Act_Code, the TIME_quality_control qc flag needs to be adjusted if not
    already set (applies to data processed with older versions of MQUEST"""

    # exit this if we don't have a TEA or TER code
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("TEA|TER")]) == 0:
        return profile

    # get the temperature QC codes
    tempqc = profile.data['TEMP_quality_control']
    if profile.histories['HISTORY_QC_CODE'].str.contains('TEA').any() & profile.data['TIME_quality_control'] != 5:
        # TEA
        profile.data['TIME_quality_control'] = 5
        LOGGER.info('TIME correction (TEA) in original file, changing TIME flag to level 5.')
        # change to flag 2 for temperature for all depths where qc is less than 2
        tempqc[tempqc < 2] = 2
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if profile.histories.loc[
            profile.histories['HISTORY_QC_CODE'].str.contains('TEA'), 'HISTORY_PREVIOUS_VALUE'].values != profile.data[
            'TIME_RAW']:
            LOGGER.error('TIME_RAW not the same as the PREVIOUS_value!')
            exit(1)

    return profile


def add_uncertainties(profile):
    """ return the profile with added uncertainties"""

    # use standard uncertainties assigned by IQuOD procedure:
    # XBT manufacturers other than Sippican and TSK and unknown manufacturer / type:  0.2;  <= 230m: 4.6m; > 230 m: 2%
    # XBT deployed from submarines or Tsurumi - Seiki Co(TSK) manufacturer 0.15;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT Sippican manufacturer 0.1;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT deployed from aircraft 0.056
    # XCTD(pre - 1998) 0.06; 4 %
    # XCTD(post - 1998) 0.02; 2 %

    pt = int(profile.data['PROBE_TYPE'])
    if 1 <= pt <= 71:
        # Sippican probe type
        tunc = 0.1
        dunc = [0.02, 4.6]
    elif 201 <= pt <= 252:
        # TSK probe type
        tunc = 0.15
        dunc = [0.02, 4.6]
    elif 401 <= pt <= 501:
        # Sparton probe type
        tunc = 0.2
        dunc = [0.02, 4.6]
    elif pt == 81 or pt == 281 or pt == 510:
        # AIRIAL XBT probe types
        tunc = 0.056
        dunc = 0  # no depth uncertainty determined
    elif 700 <= pt <= 751:
        # XCTDs
        if profile.data['TIME'] < datetime.strptime('1998-01-01', '%Y-%m-%d'):
            tunc = 0.02
            dunc = 0.04
        else:
            tunc = 0.02
            dunc = 0.02
    else:
        # probe type not defined above, not in the code table 1770
        tunc = 0
        dunc = 0
    # temp uncertainties
    profile.data['TEMP_uncertainty'] = ma.empty_like(profile.data['TEMP'])
    profile.data['TEMP_uncertainty'][:] = tunc
    # depth uncertainties:
    unc = np.ma.MaskedArray(profile.data['DEPTH'] * dunc[0], mask=False)
    if len(dunc) > 1:
        unc[profile.data['DEPTH'] <= 230] = dunc[1]
    profile.data['DEPTH_uncertainty'] = np.round(unc, 2)

    return profile


def get_fallrate_eq_coef(profile_qc, profile_noqc):
    """return probe type name, coef_a, coef_b as defined in WMO1770"""
    fre_list = read_section_from_xbt_config('FRE')
    peq_list = read_section_from_xbt_config('PEQ$')
    ptyp_list = read_section_from_xbt_config('PTYP')

    att_name = 'XBT_probetype_fallrate_equation'
    nms = [profile_qc, profile_noqc]
    vv = ['PROBE_TYPE', 'PROBE_TYPE_RAW']
    xx = ['fallrate_equation_coefficients', 'fallrate_equation_coefficients_raw']
    ind = 0
    profile_qc.ptyp = {}
    for s in nms:
        if att_name in list(profile_qc.global_atts.keys()):
            item_val = s.global_atts[att_name]
            item_val = ''.join(item_val.split())
            if item_val in list(ptyp_list.keys()):
                # old PTYP surface code, need to match up PEQ$code
                item_val = ptyp_list[item_val].split(',')[0]

            if item_val in list(fre_list.keys()):
                probetype = peq_list[item_val].split(',')[0]
                coef_a = fre_list[item_val].split(',')[0]
                coef_b = fre_list[item_val].split(',')[1]

                profile_qc.data[vv[ind]] = item_val
                profile_qc.ptyp[vv[ind] + '_name'] = probetype
                profile_qc.ptyp[xx[ind]] = 'a: ' + coef_a + ', b: ' + coef_b
            else:
                profile_qc.ptyp[xx[ind]] = []
                profile_qc.data[vv[ind]] = []
                profile_qc.ptyp[vv[ind] + '_name'] = []
                LOGGER.warning('{item_val} missing from FRE part in xbt_config file'.format(item_val=item_val))
        else:
            _error('XBT_probetype_fallrate_equation missing from {input_nc_path}'.format(
                input_nc_path=profile_qc.XBT_input_filename))
        ind = ind + 1

    # select a QC flag for the probe type
    if not profile_qc.data['PROBE_TYPE']:
        # no probe type assigned
        profile_qc.data['PROBE_TYPE_quality_control'] = 3
        # TODO: need to handle the qC flags for temp and depth here, they both need to be changed to 3 and
        #  histories updated
        LOGGER.error('Probe type is unknown. Review code handling!')
        exit(1)
    else:
        # TODO: if the probe types are different in raw and edited, need to handle this.
        #  Has it been changed? what does the data look like? Need to assign 5 to changed profile, include the PR flag
        #  and adjust the QC on the temperature and depth
        if profile_qc.data['PROBE_TYPE'] != profile_qc.data['PROBE_TYPE_RAW']:
            LOGGER.error('Probe types are different in ed and raw files. Review code handling!')
            exit(1)
        else:
            profile_qc.data['PROBE_TYPE_quality_control'] = 1

    return profile_qc


def parse_histories_nc(profile):
    """ Parse the history records in Mquest files
    """
    # let's use a pandas dataframe
    df = pd.DataFrame()
    nhist = int(profile.netcdf_file_obj['Num_Hists'][0].data)

    df['HISTORY_INSTITUTION'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                                 for xx in profile.netcdf_file_obj['Ident_Code'][0:nhist].data if bytearray(xx).strip()]

    df['HISTORY_QC_CODE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                             for xx in profile.netcdf_file_obj['Act_Code'][0:nhist].data if bytearray(xx).strip()]

    df['HISTORY_PARAMETER'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                               for xx in profile.netcdf_file_obj['Act_Parm'][0:nhist].data if bytearray(xx).strip()]

    df['HISTORY_SOFTWARE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                              for xx in profile.netcdf_file_obj['PRC_Code'][0:nhist].data if bytearray(xx).strip()]

    df['HISTORY_DATE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                          for xx in profile.netcdf_file_obj['PRC_Date'][0:nhist].data if bytearray(xx).strip()]
    df['HISTORY_START_DEPTH'] = profile.netcdf_file_obj['Aux_ID'][0:nhist].data
    df['HISTORY_TEMP_QC_CODE_VALUE'] = profile.netcdf_file_obj['Flag_severity'][0:nhist].data
    df['HISTORY_SOFTWARE_RELEASE'] = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in
                                      profile.netcdf_file_obj['Version'][0:nhist].data if bytearray(xx).strip()]

    dat = [float(x.replace(':', '')) for x in
                                    [''.join(chr(x) for x in bytearray(xx).strip()).rstrip('\x00')
                                     for xx in profile.netcdf_file_obj.variables['Previous_Val'][0:nhist]] if x]
    if dat:
        df['HISTORY_PREVIOUS_VALUE'] = dat
    else:
        df['HISTORY_PREVIOUS_VALUE'] = np.nan

    df = df.astype({'HISTORY_SOFTWARE_RELEASE': np.str_,'HISTORY_QC_CODE': np.str_})

    if nhist > 0:
        df['HISTORY_QC_CODE'] = df['HISTORY_QC_CODE'].str.replace('\x00', '')
        df['HISTORY_DATE'] = df['HISTORY_DATE'].str.replace('\x00', '')
        df['HISTORY_DATE'] = df['HISTORY_DATE'].str.replace(' ', '0')
        df['HISTORY_PARAMETER'] = df['HISTORY_PARAMETER'].str.replace('\x00', '')
        df['HISTORY_SOFTWARE'] = df['HISTORY_SOFTWARE'].str.replace('\x00', '')
        # allow for history dates to be YYYYMMDD or DDMMYYYY
        date1 = pd.to_datetime(df['HISTORY_DATE'], errors='coerce', format='%Y%m%d')
        date2 = pd.to_datetime(df['HISTORY_DATE'], errors='coerce', format='%d%m%Y')
        df['HISTORY_DATE'] = date1.fillna(date2)
        # depth value of modified act_parm var modified

    # TODO: check this correctly identifies duplicated CS flags, might be faulty profile information where the
    # previous values are not recorded at all correctly
    # tidy up the aux_id, previous_val, etc. Remove duplicated values of CS (where there are 99.99 in previous_val)
    df_dups = df[(df['HISTORY_PREVIOUS_VALUE'] == 99.99) &
                 ((df['HISTORY_QC_CODE'] == 'CS') | (df['HISTORY_QC_CODE'] == 'QC'))].index
    if len(df_dups) > 0:
        df = df.drop(df_dups)
        nhist = len(df['HISTORY_QC_CODE'])
        LOGGER.warning("Removed duplicate CS and QC codes. Please check!!")

    # Arrange histories to suit new format
    act_code_full_profile = read_section_from_xbt_config('ACT_CODES_FULL_PROFILE')
    act_code_single_point = read_section_from_xbt_config('ACT_CODES_SINGLE_POINT')
    act_code_next_flag = read_section_from_xbt_config('ACT_CODES_TO_NEXT_FLAG')
    act_code_changed = read_section_from_xbt_config('ACT_CODES_CHANGED')
    act_code_list = {**act_code_full_profile, **act_code_single_point, **act_code_next_flag}
    # grab software names
    # names = read_section_from_xbt_config('VARIOUS')

    # add the QC description information
    df["HISTORY_QC_CODE_DESCRIPTION"] = [''] * nhist
    df['HISTORY_QC_CODE_DESCRIPTION'] = df['HISTORY_QC_CODE'].map(act_code_list, na_action='ignore')
    if any(df['HISTORY_QC_CODE_DESCRIPTION'].eq('')):
        missing = df.loc[df['HISTORY_QC_CODE_DESCRIPTION'] == '', 'HISTORY_QC_CODE']
        for val in missing:
            LOGGER.error("QC_FLAG CODE \"%s\" is not defined. Please edit config file" % val)
            exit(1)

    # update variable names to match what is in the file
    names = {'DEPH': 'DEPTH', 'DATI': 'DATE, TIME', 'DATE': 'DATE', 'TIME': 'TIME', 'LATI': 'LATITUDE',
             'LONG': 'LONGITUDE', 'LALO': 'LATITUDE, LONGITUDE', 'TEMP': 'TEMP'}
    df['HISTORY_PARAMETER'] = df['HISTORY_PARAMETER'].map(names, na_action='ignore')
    if any(df['HISTORY_PARAMETER'].isna()):
        LOGGER.error("HISTORY_PARAMETER values - some are not defined. Please review output for this file")
        exit(1)

    # update institute names to be more descriptive
    names = {'CS': 'CSIRO', 'BO': 'Australian Bureau of Meteorology', 'AO': 'Australian Ocean Data Network',
             'AD': 'Defence'}
    df['HISTORY_INSTITUTION'] = df['HISTORY_INSTITUTION'].map(names, na_action='ignore')
    if any(df['HISTORY_INSTITUTION'].isna()):
        LOGGER.error("HISTORY_INSTITUTION values - some are not defined. Please review output for this file")
        exit(1)

    # set the software value to 2.1 for CS flag as we are keeping them in place and giving a flag of 3
    df.loc[df.HISTORY_QC_CODE == 'CS', ['HISTORY_SOFTWARE_RELEASE', 'HISTORY_SOFTWARE']] = '2.1', 'CSCBv2'

    # update software names to be more descriptive
    names = {'CSCB': 'CSIRO Quality control cookbook for XBT data v1.1',
             'CSCBv2': 'Australian XBT Quality Control Cookbook Version 2.1'}
    df['HISTORY_SOFTWARE'] = df['HISTORY_SOFTWARE'].map(names, na_action='ignore')

    # sort the flags by depth order to help with finding STOP_DEPTH
    # TODO: will keep the stop depth for now. Consider re-writing to loop over each of the lists of act_code types
    df = df.sort_values('HISTORY_START_DEPTH')
    vals = profile.data['DEPTH']
    tempqc = profile.data['TEMP_quality_control']
    for idx, row in df.iterrows():
        # Ensure start depth is the same as the value in the depth array
        # Find the closest value to the start depth in the histories
        ii = (np.abs(vals - row['HISTORY_START_DEPTH'])).argmin()
        df.at[idx, 'HISTORY_START_DEPTH'] = vals[ii]

        # QC,RE, TE, PE and EF flag applies to entire profile, stop_depth is deepest depth
        res = row['HISTORY_QC_CODE'] in act_code_full_profile
        if res:
            df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

        # if the flag is in act_code_single_point list, then stop depth is same as start
        res = row['HISTORY_QC_CODE'] in act_code_single_point
        if res:
            df.at[idx, "HISTORY_STOP_DEPTH"] = df.at[idx, 'HISTORY_START_DEPTH']

        # need to assign IPA/IPR, SPA/SPR, HFA/HFR, TEA/TER, PEA/PER categories based on flag severity
        if row['HISTORY_QC_CODE'] in act_code_changed:
            if row['HISTORY_TEMP_QC_CODE_VALUE'] in [0, 1, 2, 5]:
                df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'A'
                # change code 0 if needed
                if row['HISTORY_TEMP_QC_CODE_VALUE'] in [0] and not row['HISTORY_QC_CODE'] == 'PE':
                    LOGGER.warning('Changed HISTORY_TEMP_QC_CODE for %s to %s.' % (row['HISTORY_QC_CODE'], tempqc[ii]))
                    df.at[idx, 'HISTORY_TEMP_QC_CODE_VALUE'] = tempqc[ii]
                elif row['HISTORY_TEMP_QC_CODE_VALUE'] in [0] and row['HISTORY_QC_CODE'] == 'PE':
                    LOGGER.warning('Changed HISTORY_TEMP_QC_CODE for %s to 2.' % row['HISTORY_QC_CODE'])
                    df.at[idx, 'HISTORY_TEMP_QC_CODE_VALUE'] = 2
            else:
                df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'R'

        # TODO: surface flags in the act_code_next_flag category need to ignore the CS flags
        # if the flag is in act_code_next_flag, then stop depth is the next depth or bottom
        # find next deepest flag depth
        res = row['HISTORY_QC_CODE'] in act_code_next_flag
        stop_idx = df['HISTORY_START_DEPTH'] > row['HISTORY_START_DEPTH']
        stop_depth = df['HISTORY_START_DEPTH'][stop_idx]
        if any(stop_idx) & res:
            ii = (np.abs(vals - stop_depth.values[0])).argmin()
            df.at[idx, "HISTORY_STOP_DEPTH"] = vals[ii]
        elif res:  # if there isn't a deeper flag, use deepest depth
            df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

        # Error check for any QC flag value still zero
        if df.at[idx, 'HISTORY_TEMP_QC_CODE_VALUE'] == 0:
            if df.at[idx, 'HISTORY_QC_CODE'] == 'CS':
                df.at[idx, 'HISTORY_TEMP_QC_CODE_VALUE'] = 3
            elif df.at[idx, 'HISTORY_QC_CODE'] != 'RE':
                LOGGER.error('QC code of zero for a flag that is not RE, please check.')
                exit(1)

    if nhist > 0:
        # Change the PEA flag to LA or LO and ensure the TEMP_QC_CODE_VALUE is set to 2, not 5
        df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
                (df['HISTORY_PARAMETER'].str.contains('LATITUDE'))),
               ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'LA', 2
        df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
                (df['HISTORY_PARAMETER'].str.contains('LONGITUDE'))),
               ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'LO', 2

        # Combine duplicated TEA flags to a single TEA for TIME variable TEMP_QC_CODE_VALUE is set to 2, not 5
        # Also change just DATE TEA flags to TIME
        df_dups = df.loc[df['HISTORY_QC_CODE'].str.contains('TEA')]
        if len(df_dups) > 0:
            ti = df.loc[df['HISTORY_PARAMETER'].str.contains('TIME'), 'HISTORY_PREVIOUS_VALUE'].values
            if len(ti) == 0:
                # get the time value from the TIME variable as this hasn't been changed
                ti = str(profile.data['TIME'].hour).ljust(2, '0') + str(profile.data['TIME'].minute).ljust(2, '0')
            dat = df.loc[df['HISTORY_PARAMETER'].str.contains('DATE'), 'HISTORY_PREVIOUS_VALUE'].values
            if len(dat) == 0:
                # get the date value from the TIME variable as this hasn't been changed
                ti = str(profile.data['TIME'].year).ljust(4, '0') + str(profile.data['TIME'].month).ljust(2, '0') + \
                     str(profile.data['TIME'].day).ljust(2, '0')
            try:
                dt = datetime.strptime(str(int(dat)) + str(int(ti)), '%Y%m%d%H%M')
            except:
                dt = datetime.strptime(str(int(dat)) + str(int(ti)), '%d%m%Y%H%M')

            # change the 'DATE' label to TIME  and update the TEA PREVIOUS_VALUE to the new datetime value
            df.loc[((df['HISTORY_PARAMETER'].str.contains('DATE')) &
                    (df['HISTORY_QC_CODE'].str.contains('TEA'))), ['HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE']] = 'TIME', dt

            # remove any duplicated lines
            df = df[~(df.duplicated(['HISTORY_PARAMETER','HISTORY_QC_CODE']) & df.HISTORY_PARAMETER.eq('TIME'))]
    profile.histories = df

    return profile


def combine_histories(profile_qc, profile_noqc):
    # check for global attributes in the noqc file and update the global atts as required
    # handle the longitude change where data was imported from dataset with a negative longitude where it should
    # have been positive. The *raw.nc previous value and *ed.nc previous value should be the same, update the LONG_RAW.
    if len(profile_noqc.histories) > 0:
        # copy this information to the LONGITUDE_RAW value if it isn't the same
        if np.round(profile_noqc.histories.loc[profile_noqc.histories['HISTORY_QC_CODE'].str.contains('LO'),
                                               'HISTORY_PREVIOUS_VALUE'], 4).values != np.round(profile_qc.data['LONGITUDE_RAW'], 4):
                LOGGER.warning('Updating raw longitude to match the previous value in *raw.nc file')
                profile_qc.data['LONGITUDE_RAW'] = profile_noqc.histories.loc[
                        profile_noqc.histories['HISTORY_QC_CODE'].str.contains('LO'), 'HISTORY_PREVIOUS_VALUE'][0]
    # TODO: handle other extra histories in noqc file here:
    if len(profile_noqc.histories) > 1:
        breakpoint()

    return profile_qc

def check_for_PL_flag(profile):
    # Special case, where the PLA code has been used, the temperature values are shifted up and the edited file
    # therefore has a different number of records to the raw file. Need to pad the edited to the same size as raw
    # since we are using the same DEPTH dimension for both:
    if profile.histories['HISTORY_QC_CODE'].str.contains('PL').any():
        # double check the length of the records is different, log it
        if len(profile.data['TEMP']) < len(profile.data['TEMP_RAW']):
            LOGGER.warning('Raw and edited profiles are different length due to PLA flag. Amending.')
            # edited temp is shorter, add blanks at end
            for var in ['TEMP', 'DEPTH']:
                tr = profile.data[var + '_RAW']
                tt = profile.data[var]
                t2 = np.ma.empty_like(tr)
                t2[0:len(tt)] = tt
                t2[len(tt):] = ma.masked
                profile.data[var] = t2
                tr = profile.data[var + '_RAW_quality_control']
                tt = profile.data[var + '_quality_control']
                t2 = np.ma.empty_like(tr)
                t2[0:len(tt)] = tt
                t2[len(tt):] = ma.masked
                profile.data[var + '_quality_control'] = t2

    return(profile)


def restore_temp_val(profile):
    """
    Restore the temperature values that are associated with
    the 'CS' (surface spike removed) flag. That means identifying them, putting them back into the
    TEMP field, then putting a flag of 3 (probably bad) on them. The values can also stay
    in the HISTORY_PREVIOUS_VALUE field. This process would need to apply to both the TEMP
    and TEMP_RAW (from the *raw.nc file).
    """

    # index of CS flags in histories:
    idx = profile.histories['HISTORY_QC_CODE'] == 'CS'
    depths = profile.histories['HISTORY_START_DEPTH'][idx].values.astype('float')
    temps = profile.histories['HISTORY_PREVIOUS_VALUE'][idx].values.astype('float')

    # check if the temperature values are missing & replace with previous value if they are:
    # do for both TEMP and TEMP_RAW
    for vv in ['', '_RAW']:
        deps = profile.data['DEPTH' + vv]
        tempsp = profile.data['TEMP' + vv]
        ind = np.in1d(np.round(deps, 2), np.round(depths, 2)).nonzero()[0]
        if np.isnan(tempsp[ind]).any():
            # we need to replace these with their original temperatures
            tempsp[ind] = temps
            profile.data['TEMP' + vv] = tempsp
            # and update the flag to 3 from 5
            profile.data['TEMP' + vv + '_quality_control'][ind] = '3'
            LOGGER.info('Updated CS flags for TEMP %s' % vv)
    return profile


def create_flag_feature(profile):
    """ Take the existing QC code values and turn them into a integer representation. One bit for every code."""

    # set up a dataframe of the codes and their values
    # will need to do a mapping for some of the flags. Should retain the originals except the CS flag as per
    # the version 2.1 release.
    # could also consider using hex encoding to represent these numbers if that looks more user-friendly

    # 23 codes - New Cookbook
    df = pd.DataFrame(
        {'Code': ['QC', 'CS', 'WB', 'WS', 'HB', 'LE', 'EIA', 'EIR', 'HFA', 'HFR', 'NG', 'RE', 'IV', 'TO', 'EF',
                  'ST', 'DO', 'CT', 'TEA', 'TER', 'LA', 'LO', 'PER', 'DU', 'TP', 'PR'],
         'Meaning': ['scientific_quality_control_applied', 'surface_transient', 'wire_break',
                     'wire_stretch', 'hit_bottom',
                     'electrical_leakage', 'electrical_interference_interpolated',
                     'electrical_interference_failed', 'high_frequency_noise_filtered',
                     'high_frequency_noise_failed', 'no_good', 'repeat_profile', 'temperature_inversion',
                     'temperature_offset', 'temperature_eddy_or_front', 'temperature_steps_or_structure',
                     'depth_offset', 'constant_temperature', 'time_error_corrected', 'time_error_rejected',
                     'latitude_error_corrected', 'longitude_error_corrected', 'position_error_rejected',
                     'duplicate_profile', 'test_probe', 'probe_type_error']})

    # now do the same for the old flag values to add to a separate variable
    dfold = pd.DataFrame(
        {'Code': ['FS', 'IPA', 'IPR', 'NT', 'NU', 'PL', 'PI', 'PS', 'SA', 'SO', 'TA', 'TD', 'OP', 'BO', 'CU', 'DR',
                  'MOA', 'MOR', 'PF', 'SBA', 'SBR', 'ML', 'NA', 'UR', 'BB', 'CL', 'DC', 'DE', 'DP'],
         'Meaning': ['fine_structure', 'insulation_penetration_interpolated', 'insulation_penetration_failed',
                     'no_trace', 'nub_inversion', 'premature_launch', 'probable_inversion',
                     'probable_steplike_structure', 'surface_temperature_anomaly', 'surface_offset',
                     'temperature_anomaly', 'temperature_difference_at_depth', 'other_probe_type_error',
                     'bowing_bathy_systems_fault', 'cusping_bathy_systems_leakage',
                     'delay_driver_error_sippican_mk9_fault',
                     'modulo10_spikes_bathy_systems_interpolated', 'modulo10_spikes_bathy_systems_failed',
                     'leakage_protecno_systems_fault', 'sticking_bit_sippican_mk9_interpolated',
                     'sticking_bit_sippican_mk9_failed', 'mixed_layer', 'not_assessed', 'under_resolved_profile',
                     'bad_bottle', 'contact_lost_to_probe', 'depth_fallrate_eq_corrected', 'depth_multiplied_by_10m',
                     'depth_fallrate_eq_corrected']})
    dfmap = pd.concat([df, pd.DataFrame({'Code': ['ST', 'EIA', 'EIR', 'NG', 'IV', 'DO', 'IV', 'ST', 'TO',
                                                  'TO', 'TO', 'TO', 'NG', 'LE', 'LE', 'DO', 'EIA', 'EIR',
                                                  'LE', 'EIA', 'EIR', '', '', 'TO', 'TO', 'NG', 'DO', 'DO',
                                                  'DO']})])
    # append the new codes with the old ones and change a couple:
    dfoldc = pd.concat([df, dfold])
    dfoldc = dfoldc.replace(
        ['EIA', 'EIR', 'PR', 'electrical_interference_interpolated', 'electrical_interference_failed',
         'probe_type_error'],
        ['SPA', 'SPR', 'DT', 'spike_interpolated', 'spike_failed', 'data_type_corrected'])

    # add the mappings to the new code
    dfoldc['New Code'] = dfmap['Code'].values

    # create a list of integers to represent binary numbers:
    n = [1]
    for i in range(1, len(dfoldc['Code'])):
        n.append(n[i - 1] * 2)

    # print("masks",n)

    # add the byte values for each code:
    df['byte_value'] = np.array(n[0:len(df['Code'])])  # 52 elements

    dfoldc['byte_value'] = np.array(n)
    # print('len(dfoldc[byte_value]),dfoldc[byte_value]',len(dfoldc['byte_value']),dfoldc['byte_value'])

    # set the fields to zeros to start
    profile.data['XBT_fault_and_feature_type'] = profile.data['DEPTH'] * 0

    # Keep this here for now in case we change our minds and want to do a translation to new cookbook codes
    # profile.data['XBT_fault_and_feature_type_original'] = profile.data['DEPTH'] * 0
    # make sure that we record the fault masks, meanings and the valid max
    profile.fft = {}
    profile.fft['flag_masks'] = dfoldc['byte_value'].values
    profile.fft['flag_meanings'] = dfoldc['Meaning'].values
    profile.fft['flag_codes'] = dfoldc['Code'].values
    # profile.ffot = {}
    # profile.ffot['flag_masks'] = dfoldc['byte_value'].values
    # profile.ffot['flag_meanings'] = dfoldc['Meaning'].values
    # profile.ffot['flag_codes'] = dfoldc['Code'].values

    # perform the flag mapping on the original flags and create the two new variables
    codes = profile.histories
    # check for duplicated history codes at the same depth so we don't duplicate the QC code in the fft variable
    dup_df = codes[codes.duplicated(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)]
    if len(dup_df) > 0:
        codes = codes.drop_duplicates(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH', 'HISTORY_PREVIOUS_VALUE'],
                                      keep='first')
        LOGGER.warning('Duplicate QC code encountered, and removed. Please review')

    # print("dfoldc",dfoldc)
    # print("codes",codes)

    mapold = pd.merge(dfoldc, codes, how='right', left_on='Code', right_on='HISTORY_QC_CODE')
    if mapold.empty:
        # no flags,
        profile.global_atts['qc_completed'] = 'no'
        return profile
    else:
        # adjust global attribute to say we have done scientific QC
        profile.global_atts['qc_completed'] = 'yes'

    # create equivalent map df with the new codes:
    mapnew = mapold.drop(['Code', 'byte_value'], axis=1)
    mapnew = pd.merge(df, mapnew[['New Code', 'HISTORY_INSTITUTION',
                                  'HISTORY_PARAMETER', 'HISTORY_SOFTWARE', 'HISTORY_DATE',
                                  'HISTORY_START_DEPTH', 'HISTORY_SOFTWARE_RELEASE',
                                  'HISTORY_PREVIOUS_VALUE',
                                  'HISTORY_STOP_DEPTH']], how='right', left_on='Code', right_on='New Code')
    mapnew = mapnew.drop(['Code'], axis=1)

    # tidy up the duplicated code/new code and assign new values to the HISTORIES codes and descriptions
    mapnew = mapnew.rename({'New Code': 'HISTORY_QC_CODE', 'Meaning': 'HISTORY_QC_CODE_DESCRIPTION'}, axis=1)
    # reset the software information:
    mapnew['HISTORY_SOFTWARE_RELEASE'] = '2.1'
    mapnew['HISTORY_SOFTWARE'] = 'Australian XBT Quality Control Cookbook Version 2.1'
    mapold = mapold.drop(['Code', 'New Code', 'HISTORY_QC_CODE_DESCRIPTION'], axis=1)
    mapold = mapold.rename({'Meaning': 'HISTORY_QC_CODE_DESCRIPTION'}, axis=1)

    # now, we can use either the old history codes, new ones or combine if we decide that is the way to go.
    # For now, keep the existing history codes to represent in the histories section and in the feature flag variable
    profile.histories = mapold[profile.histories.columns]

    # any flags not included?
    cc = list(dfoldc.Code)
    missingf = list(set(codes['HISTORY_QC_CODE'].values.tolist()) - set(cc))
    if missingf:
        # we have an extra flag that we haven't coded
        LOGGER.error('New QC code encountered, please code in the new value')
        exit(1)

    # now need to assign the codes to the correct depths.
    # code only added in one location at the start depth, QC flags indicate the quality applied
    # for each code, need an array of values same size as DEPTH, then add them all together
    # also check the TEMP_QC_CODE_VALUE is the same as the actual flag in the flag array
    deps = profile.data['DEPTH']

    # Iterate over the history table.
    # Using 'old' QC code mappings as this code is for re-formatting of the old files with old codes
    for idx, row in mapold.iterrows():
        nullarray = deps * 0
        # Get depth index
        ii = (np.abs(deps - row['HISTORY_START_DEPTH'])).argmin()
        # set that depth to byte value for that QC code from hist table
        nullarray[ii] = row['byte_value']
        # adding them together - is there a more correct way to do this?
        # Add byte values (masks)
        profile.data['XBT_fault_and_feature_type'] = profile.data['XBT_fault_and_feature_type'] + nullarray

    # for idx, row in mapold.iterrows():
    #    nullarray = deps * 0
    #    ii = (deps >= row['HISTORY_START_DEPTH']) * (deps <= row['HISTORY_STOP_DEPTH'])
    #    nullarray[ii] = row['byte_value']
    # adding them together - is there a more correct way to do this?
    #    profile.data['XBT_fault_and_feature_type_original'] = profile.data[
    #                                                              'XBT_fault_and_feature_type_original'] + nullarray

    return profile


def check_nc_to_be_created(profile):
    """ different checks to make sure we want to create a netcdf for this profile
    """
    # sometimes we have non-XBT data in the files, skip this
    # will probably need to think about XCTD data!!

    data_type = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['Data_Type'][:].data)).strip()
    duplicate_flag = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['Dup_Flag'][0].data)).strip()
    nhist = int(profile.netcdf_file_obj['Num_Hists'][0].data)
    histcodes = [''.join(chr(x) for x in bytearray(xx)).strip()
                 for xx in profile.netcdf_file_obj['Act_Code'][0:nhist].data if bytearray(xx).strip()]
    depth = np.round(profile.netcdf_file_obj.variables['Depthpress'][:], 2)

    if len(depth) == 0:
        LOGGER.error('No data in the file')
        return False

    if data_type != 'XB':
        LOGGER.error('Profile not processed as it is not an XBT')
        return False

    if duplicate_flag == 'D':
        LOGGER.error('Profile not processed. Tagged as duplicate in original netcdf file')
        return False

    if 'TP' in histcodes or 'DU' in histcodes:
        LOGGER.error('Profile not processed. Tagged as test probe in original netcdf file')
        return False

    #    if annex['no_prof'] > 1:
    #        LOGGER.error('Profile not processed. No_Prof variable is greater than 0')
    #        return False

    data_vars = temp_prof_info(profile.netcdf_file_obj)
    if 'TEMP' not in data_vars.values():
        LOGGER.error('Profile not processed, no TEMP in file.')
        return False

    return True


def create_filename_output(profile):
    filename = 'XBT_T_%s_%s_FV01_ID-%s' % (
        profile.data['TIME'].strftime('%Y%m%dT%H%M%SZ'), profile.global_atts['XBT_line'],
        profile.global_atts['XBT_uniqueid'])

    # decide what prefix is required
    names = read_section_from_xbt_config('VARIOUS')
    str = names['FILENAME']
    if str == 'Cruise_ID':
        str = profile.global_atts['XBT_cruise_ID']
        filename = '{}-{}'.format(str, filename)
    else:
        if profile.data['TIME'] > datetime(2008, 0o1, 0o1):
            filename = 'IMOS_SOOP-{}'.format(filename)

    if '/' in filename:
        LOGGER.error('The sign \'/\' is contained inside the NetCDF filename "%s". Likely '
                     'due to a slash in the XTB_line attribute. Please ammend '
                     'the XBT_line attribute in the config file for the XBT line "%s"'
                     % (filename, profile.global_atts['XBT_line']))
        exit(1)

    return filename


def write_output_nc(output_folder, profile, profile_raw=None):
    """output the data to the IMOS format netcdf version"""

    # now begin write out to new format
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(profile))
    LOGGER.info('Creating output %s' % netcdf_filepath)

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(profile.data['DEPTH']))
        output_netcdf_obj.createDimension('N_HISTORY', len(profile.histories.index))

        # Create the variables, no dimensions:
        varslist = ["TIME", "LATITUDE", "LONGITUDE"]
        for vv in varslist:
            output_netcdf_obj.createVariable(vv, datatype=get_imos_parameter_info(vv, '__data_type'),
                                             fill_value=get_imos_parameter_info(vv, '_FillValue'))
            # and associated QC variables:
            output_netcdf_obj.createVariable(vv + "_quality_control", "b", fill_value=99)
            # and the *_RAW variables:
            output_netcdf_obj.createVariable(vv + "_RAW", datatype=get_imos_parameter_info(vv, '__data_type'),
                                             fill_value=get_imos_parameter_info(vv, '_FillValue'))

        # Create the dimensioned variables:
        varslist = [key for key in profile.data.keys() if ('_quality_control' not in key and 'RAW' not in key
                                                           and 'TUDE' not in key and 'XBT' not in key
                                                           and 'TIME' not in key and 'uncertainty' not in key
                                                           and 'PROBE' not in key)]
        for vv in varslist:
            output_netcdf_obj.createVariable(vv, datatype=get_imos_parameter_info(vv, '__data_type'),
                                             dimensions=('DEPTH',),
                                             fill_value=get_imos_parameter_info(vv, '_FillValue'))
            # and associated QC variables:
            output_netcdf_obj.createVariable(vv + "_quality_control", "b", dimensions=('DEPTH',), fill_value=99)
            # and uncertainty values for DEPTH and TEMP
            output_netcdf_obj.createVariable(vv + "_uncertainty", "f", dimensions=('DEPTH',), fill_value=999999.0)
            # and the *_RAW variables:
            output_netcdf_obj.createVariable(vv + "_RAW", datatype=get_imos_parameter_info(vv, '__data_type'),
                                             dimensions=('DEPTH',),
                                             fill_value=get_imos_parameter_info(vv, '_FillValue'))
            # create a QC variable for the _RAW data if there are flags included
            # (some files are converted from QC'd datasets and therefore have flags associated with the 'raw' data
            if profile.data[vv + '_RAW_quality_control'].any() > 0:
                LOGGER.warning("QC values have been written to file for \"%s\" variable. Review." % vv)
                output_netcdf_obj.createVariable(vv + "_RAW_quality_control", "b", dimensions=('DEPTH',), fill_value=99)

            if vv == 'TEMP' and profile_raw is not None:
                # add the recording system variable:
                output_netcdf_obj.createVariable(vv + "_RECORDING_SYSTEM", "f", dimensions=('DEPTH',),
                                                 fill_value=999999.0)
                # and associated QC variables:
                output_netcdf_obj.createVariable(vv + "_RECORDING_SYSTEM_quality_control", "b", dimensions=('DEPTH',),
                                                 fill_value=-51)

        # Create the last variables that are non-standard:
        output_netcdf_obj.createVariable("PROBE_TYPE", 'S3')
        output_netcdf_obj.createVariable("PROBE_TYPE_quality_control", "b", fill_value=99)
        output_netcdf_obj.createVariable("PROBE_TYPE_RAW", 'S3')

        fftype = output_netcdf_obj.createVariable("XBT_fault_and_feature_type", "u8", dimensions=('DEPTH',),
                                                  fill_value=0)
        # ffotype = output_netcdf_obj.createVariable("XBT_fault_and_feature_type_original", "u8", dimensions=('DEPTH',),
        #                                          fill_value=0)

        # If the turo profile is handed in:
        if profile_raw is not None:
            output_netcdf_obj.createVariable("RESISTANCE", "f", dimensions=('DEPTH',), fill_value=float("nan"))
            output_netcdf_obj.createVariable("SAMPLE_TIME", "f", dimensions=('DEPTH',), fill_value=float("nan"))

        # create HISTORY variable set associated
        output_netcdf_obj.createVariable("HISTORY_INSTITUTION", "str", 'N_HISTORY')
        # output_netcdf_obj.createVariable("HISTORY_STEP", "str", 'N_HISTORY') # removed for now, RC August 2023
        output_netcdf_obj.createVariable("HISTORY_SOFTWARE", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_SOFTWARE_RELEASE", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_DATE", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_PARAMETER", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_START_DEPTH", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_STOP_DEPTH", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_QC_CODE", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_QC_CODE_DESCRIPTION", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_TEMP_QC_CODE_VALUE", "f", 'N_HISTORY')

        # write attributes from the generate_nc_file_att file, now that we have added the variables:
        conf_file = os.path.join(os.path.dirname(__file__), 'generate_nc_file_att')
        generate_netcdf_att(output_netcdf_obj, conf_file, conf_file_point_of_truth=True)
        # add the flag and feature type attributes:
        setattr(fftype, 'valid_max', int(profile.fft['flag_masks'].sum()))
        # AW use type  - 64 bit unsigned int, basic int type only up to 32 bit, this is same type as variable XBT_fault_and_feature_type
        # setattr(fftype, 'flag_masks', profile.fft['flag_masks'].astype(int))
        setattr(fftype, 'flag_masks', profile.fft['flag_masks'].astype(np.uint64))
        setattr(fftype, 'flag_meanings', ' '.join(profile.fft['flag_meanings']))
        setattr(fftype, 'flag_codes', ' '.join(profile.fft['flag_codes']))
        # setattr(ffotype, 'valid_max', int(profile.ffot['flag_masks'].sum()))
        # setattr(ffotype, 'flag_masks', profile.ffot['flag_masks'].astype(int))
        # setattr(ffotype, 'flag_meanings', ' '.join(profile.ffot['flag_meanings']))
        # setattr(ffotype, 'flag_codes', ' '.join(profile.ffot['flag_codes']))

        # write coefficients out to the attributes. In the PROBE_TYPE, PROBE_TYPE_RAW, DEPTH, DEPTH_RAW
        varnames = ['PROBE_TYPE', 'DEPTH']
        for v in varnames:
            setattr(output_netcdf_obj.variables[v], 'fallrate_coefficients',
                    profile.ptyp['fallrate_equation_coefficients'])
            setattr(output_netcdf_obj.variables[v], 'probe_type_name', profile.ptyp['PROBE_TYPE_name'])

        varnames = ['PROBE_TYPE_RAW', 'DEPTH_RAW']
        for v in varnames:
            setattr(output_netcdf_obj.variables[v], 'fallrate_coefficients',
                    profile.ptyp['fallrate_equation_coefficients_raw'])
            setattr(output_netcdf_obj.variables[v], 'probe_type_name', profile.ptyp['PROBE_TYPE_RAW_name'])

        # append the data to the file
        # qc'd
        for v in list(output_netcdf_obj.variables):
            if v not in list(profile.data) and v not in list(profile.histories):
                LOGGER.warning(
                    "Variable not written: \"%s\". Please check!!" % v)
                continue
            if v == 'TIME' or v == 'TIME_RAW':
                # AW DEBUG
                '''
                for attr in output_netcdf_obj[v].ncattrs():
                    print("attr",attr)
                print("var name, var",v,profile.data[v])
                print("units",output_netcdf_obj[v].units)
                print("calendar",output_netcdf_obj[v].calendar)
                '''
                time_val_dateobj = date2num(profile.data[v], output_netcdf_obj[v].units,
                                            output_netcdf_obj[v].calendar)
                output_netcdf_obj[v][:] = time_val_dateobj
            elif v in list(profile.data):
                if isinstance(output_netcdf_obj[v][:], str):
                    output_netcdf_obj[v][len(profile.data[v])] = profile.data[v]
                else:
                    output_netcdf_obj[v][:] = profile.data[v]
            else:
                # histories
                if v == 'HISTORY_DATE':
                    # fix history date time field
                    count = 0
                    for ii in profile.histories[v]:
                        history_date_obj = date2num(datetime.strptime(str(ii), '%Y-%m-%d %H:%M:%S'),
                                                    output_netcdf_obj['HISTORY_DATE'].units,
                                                    output_netcdf_obj['HISTORY_DATE'].calendar)
                        output_netcdf_obj[v][count] = history_date_obj
                        count += 1
                else:
                    output_netcdf_obj[v][:] = profile.histories[v].values

        # write out the extra global attributes we have collected
        # default value for abstract
        # if not hasattr(output_netcdf_obj, 'abstract'):
        #    setattr(output_netcdf_obj, 'abstract', output_netcdf_obj.title)
        for key, item in profile.global_atts.items():
            setattr(output_netcdf_obj, key, item)


def _call_parser(conf_file):
    """ parse a config file """
    parser = ConfigParser()
    parser.optionxform = str  # to preserve case
    conf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), conf_file)
    parser.read(conf_file_path)
    return parser


def read_section_from_xbt_config(section_name):
    "return all the elements in the section called section_name from the xbt_config file"
    xbt_config = _call_parser('xbt_config')
    if section_name in xbt_config.sections():
        return dict(xbt_config.items(section_name))
    else:
        _error('xbt_config file not valid. missing section: {section}'.format(section=section_name))


def args():
    """ define input argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-xbt-campaign-path', type=str,
                        help="path to *_keys.nc or campaign folder below the keys.nc file")
    parser.add_argument('-o', '--output-folder', nargs='?', default=1,
                        help="output directory of generated files")
    parser.add_argument('-l', '--log-file', nargs='?', default=1,
                        help="log directory")
    vargs = parser.parse_args()

    if vargs.output_folder == 1:
        vargs.output_folder = tempfile.mkdtemp(prefix='xbt_dm_')
    elif not os.path.isabs(os.path.expanduser(vargs.output_folder)):
        vargs.output_folder = os.path.join(os.getcwd(), vargs.output_folder)

    if vargs.log_file == 1:
        vargs.log_file = os.path.join(vargs.output_folder, 'xbt.log')
    else:
        if not os.path.exists(os.path.dirname(vargs.log_file)):
            os.makedirs(os.path.dirname(vargs.log_file))

    if not os.path.exists(vargs.input_xbt_campaign_path):
        msg = '%s not a valid path' % vargs.input_xbt_campaign_path
        print(msg, file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(vargs.output_folder):
        os.makedirs(vargs.output_folder)

    return vargs


def global_vars(vargs):
    global LOGGER
    logging = IMOSLogging()
    LOGGER = logging.logging_start(vargs.log_file)

    global NETCDF_FILE_PATH  # defined as glob to be used in exception

    global SHIP_CALL_SIGN_LIST
    SHIP_CALL_SIGN_LIST = ship_callsign_list()  # AODN CALLSIGN vocabulary

    global XBT_LINE_INFO
    XBT_LINE_INFO = xbt_line_info()

    global INPUT_DIRNAME  # in the case we're processing a directory full of NetCDF's and not ONE NetCDF only
    INPUT_DIRNAME = None


if __name__ == '__main__':
    """
    Example:
    ./xbt_dm_imos_conversion.py -i XBT/GTSPPmer2017/GTSPPmer2017MQNC_keys.nc -o /tmp/xb
    ./xbt_dm_imos_conversion.py -i XBT/GTSPPmer2017/GTSPPmer2017MQNC -o /tmp/xb
    """
    os.umask(0o002)
    vargs = args()
    global_vars(vargs)

    # read the keys file into a keys object
    # print("vargs",vargs)
    # Vargs contains:
    '''
    input_xbt_campaign_path='filename_keys.nc', 
    output_folder='output_directory_pathname', 
    log_file='path_to_xbt.log'
    '''

    keys = XbtKeys(vargs)

    for f in keys.data['station_number']:
        fpath = '/'.join(re.findall('..?', str(f))) + 'ed.nc'
        fname = os.path.join(keys.dbase_name, fpath)

        # if the file exists, let's make a profile object with all the
        # data and metadata attached.

        if os.path.isfile(fname):
            # read the edited profile
            profile_ed = XbtProfile(fname)
            # read the raw profile
            profile_raw = XbtProfile(fname.replace('ed.nc', 'raw.nc'))
            # TODO: check the keys data (date/time/lat/long etc) against what is in the data file
            # TODO: find the matching TURO profile if it is available:
            # profile_turo = turoProfile(profile_ed)
            profile_turo = []

            # now write it out to the new netcdf format
            if check_nc_to_be_created(profile_ed):
                # for example where depths are different, metadata is different etc between the ed and raw files.
                profile_ed = coordinate_data(profile_ed, profile_raw, profile_turo)
                write_output_nc(vargs.output_folder, profile_ed)
        else:
            LOGGER.warning('file %s is in keys file, but does not exist' % f)
