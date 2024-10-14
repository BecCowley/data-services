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

    def __init__(self, file_path_name, input_filename):
        """ Read XBT files written in an un-friendly NetCDF format
        global attributes, data and annex information are added to the object
        """
        # record the file name
        self.XBT_filename = file_path_name
        self.XBT_input_filename = input_filename

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
            latitude = np.round(netcdf_file_obj['obslat'][:].data, 4)
            longitude = np.round(netcdf_file_obj['obslng'][:].data, 4)
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
    profile_noqc.global_atts['geospatial_vertical_max'] = max(profile_qc.data['data']['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_vertical_min'] = min(profile_qc.data['data']['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_lat_max'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lat_min'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_max'] = profile_qc.data['LONGITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_min'] = profile_qc.data['LONGITUDE_RAW']
    profile_noqc.global_atts['time_coverage_start'] = profile_qc.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")
    profile_noqc.global_atts['time_coverage_end'] = profile_qc.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")

    # let's check if there are histories to parse and then handle
    profile_qc = parse_histories_nc(profile_qc)
    if int(profile_noqc.netcdf_file_obj['Num_Hists'][0].data) == 0:
        profile_noqc.histories = []
    else:
        # we need to carry the depths information into the history parsing, so copy the data array into profile_noqc
        profile_noqc.data = dict()
        profile_noqc.data['data'] = pd.DataFrame()
        profile_noqc.data['data']['DEPTH'] = profile_qc.data['data']['DEPTH_RAW']
        profile_noqc.data['data']['TEMP_quality_control'] = profile_qc.data['data']['TEMP_quality_control']
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

        # adjust lat lon qc flags if required
        profile_qc = adjust_position_qc_flags(profile_qc, profile_noqc)
        # adjust date and time QC flags if required
        profile_qc = adjust_time_qc_flags(profile_qc)

        # perform a check of the qc vs noqc global attributes and histories. Do any of these need reconciling?
        if len(profile_qc.global_atts.keys() - profile_noqc.global_atts):
            # if the difference in the global attributes is just the qc_completed key, continue
            if len(profile_qc.global_atts.keys() - profile_noqc.global_atts) == 1:
                if 'qc_completed' in profile_qc.global_atts.keys() - profile_noqc.global_atts:
                    pass
                else:
                    LOGGER.error('%s GLOBAL attributes in RAW and ED files are not consistent'
                                 % profile_qc.XBT_input_filename)

    # Probe type goes into a variable with coefficients as attributes, and assign QC to probe types
    profile_qc = get_fallrate_eq_coef(profile_qc, profile_noqc)

    # check that the sums of TEMP and TEMP_RAW and DEPTH and DEPTH_RAW are the same within a tolerance
    check_sums_of_temp_depth(profile_qc)

    # add uncertainties:
    profile_qc = add_uncertainties(profile_qc)

    return profile_qc


def check_sums_of_temp_depth(profile_qc):
    # check that the sums of TEMP and TEMP_RAW and DEPTH and DEPTH_RAW are the same within a tolerance
    # check the sum of the TEMP and TEMP_RAW columns
    if not np.isclose(np.sum(profile_qc.data['data']['TEMP']), np.sum(profile_qc.data['data']['TEMP_RAW']), rtol=1e-3):
        LOGGER.error('The sum of TEMP and TEMP_RAW are not the same in %s' % profile_qc.XBT_input_filename)

    # check the sum of the DEPTH and DEPTH_RAW columns
    if not np.isclose(np.sum(profile_qc.data['data']['DEPTH']), np.sum(profile_qc.data['data']['DEPTH_RAW']),
                      rtol=1e-3):
        LOGGER.error('The sum of DEPTH and DEPTH_RAW are not the same in %s' % profile_qc.XBT_input_filename)


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
            LOGGER.warning(
                '{item_val} missing from recorder type part in xbt_config file, using unknown for recorder'.format(
                    item_val=item_val))
            item_val = '99'
            return item_val, rct_list[item_val].split(',')[0]
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
    profile.global_atts['XBT_gts_insertion_node'] = \
        decode_bytearray(profile.netcdf_file_obj['Source_ID'][:]).replace('\x00', '').strip()
    # source_id = 'AMMC' if source_id == '' else source_id

    # get the institution code from the first two characters of the Stream_Ident
    institute = decode_bytearray(profile.netcdf_file_obj['Stream_Ident'][:]).strip()[:2]
    # create a dictionary of the institution codes
    institute_list = read_section_from_xbt_config('INSTITUTE')
    if institute in list(institute_list.keys()):
        profile.global_atts['institution'] = institute_list[institute].split(',')[0]
        profile.global_atts['Agency_code'] = institute_list[institute].split(',')[1]
    else:
        LOGGER.warning('Institute code %s is not defined in xbt_config file. Please edit xbt_config' % institute)

    for count in range(profile.nprof):
        try:
            profile.global_atts['gtspp_digitisation_method_code_' + profile.prof_type[count:]] = \
                decode_bytearray(profile.netcdf_file_obj['Digit_Code'][count]).replace('\x00', '').strip()
            profile.global_atts['gtspp_precision_code_' + profile.prof_type[count:]] \
                = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['Standard'][count].data)).replace('\x00',
                                                                                                              '').strip()
        except:
            profile.global_atts['gtspp_digitisation_method_code_' + profile.prof_type[count:]] = np.nan
            profile.global_atts['gtspp_precision_code_' + profile.prof_type[count:]] = np.nan
    try:
        profile.global_atts['XBT_predrop_comments'] \
            = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['PreDropComments'][:].data)).replace(
            '\x00', '').strip()
        profile.global_atts['XBT_postdrop_comments'] \
            = ''.join(chr(x) for x in bytearray(profile.netcdf_file_obj['PostDropComments'][:].data)).replace(
            '\x00', '').strip()
    except:
        profile.global_atts['XBT_predrop_comments'] = ''
        profile.global_atts['XBT_postdrop_comments'] = ''

    profile.global_atts['geospatial_vertical_units'] = 'meters'
    profile.global_atts['geospatial_vertical_positive'] = 'down'

    # include the input filename
    profile.global_atts['XBT_input_file'] = profile.XBT_input_filename

    try:
        profile.global_atts['geospatial_lat_max'] = profile.data['LATITUDE']
        profile.global_atts['geospatial_lat_min'] = profile.data['LATITUDE']
        profile.global_atts['geospatial_lon_max'] = profile.data['LONGITUDE']
        profile.global_atts['geospatial_lon_min'] = profile.data['LONGITUDE']
        profile.global_atts['geospatial_vertical_max'] = max(profile.data['data']['DEPTH'])
        profile.global_atts['geospatial_vertical_min'] = min(profile.data['data']['DEPTH'])
        # include time_coverage_start and time_coverage_end in the global attributes
        profile.global_atts['time_coverage_start'] = profile.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")
        profile.global_atts['time_coverage_end'] = profile.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        profile.global_atts['geospatial_lat_max'] = []
        profile.global_atts['geospatial_lat_min'] = []
        profile.global_atts['geospatial_lon_max'] = []
        profile.global_atts['geospatial_lon_min'] = []
        profile.global_atts['geospatial_vertical_max'] = []
        profile.global_atts['geospatial_vertical_min'] = []
        profile.global_atts['time_coverage_start'] = []
        profile.global_atts['time_coverage_end'] = []

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

    # if the platform code didn't come through, assign unknown type
    if 'Platform_code' not in profile.global_atts.keys():
        LOGGER.error('Platform_code is missing, GCLL has not been read or is missing')
        # assign unknown to the platform code
        profile.global_atts['Platform_code'] = 'Unknown'

    # get the ship details
    # note that the callsign and ship name are filled from the original file values, but will be replaced here if they exist in the AODN vocabulary
    # for these older historical files, the Callsign and Platform_code are the same. In newer files, the platform_code
    # will be the GTSID or SOTID.
    profile.global_atts['Callsign'] = profile.global_atts[
        'Platform_code']  # set here as can't have duplicate assignments in the config file
    ships = SHIP_CALL_SIGN_LIST
    if profile.global_atts['Platform_code'] in ships:
        profile.global_atts['ship_name'] = ships[profile.global_atts['Platform_code']][0]
        profile.global_atts['ship_IMO'] = ships[profile.global_atts['Platform_code']][1]
    elif difflib.get_close_matches(profile.global_atts['Platform_code'], ships, n=1, cutoff=0.8) != []:
        profile.global_atts['Callsign'] = \
            difflib.get_close_matches(profile.global_atts['Platform_code'], ships, n=1, cutoff=0.8)[0]
        profile.global_atts['ship_name'] = ships[profile.global_atts['Callsign']][0]
        profile.global_atts['ship_IMO'] = ships[profile.global_atts['Callsign']][1]
        LOGGER.warning('Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s' % (
            profile.global_atts['Platform_code'], profile.global_atts['Callsign']))
    else:
        profile.global_atts['ship_name'] = 'Unknown'
        profile.global_atts['ship_IMO'] = 'Unknown'
        LOGGER.warning('Vessel call sign %s is unknown in AODN vocabulary, Please contact info@aodn.org.au' %
                       profile.global_atts['Platform_code'])

    # extract the information and assign correctly
    att_name = 'XBT_recorder_type'
    if att_name in list(profile.global_atts):
        recorder_val, recorder_type = get_recorder_type(profile)
        profile.global_atts['XBT_recorder_type'] = recorder_val + ', ' + recorder_type
    else:
        profile.global_atts['XBT_recorder_type'] = '99, Unknown'

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

    return profile


def parse_data_nc(profile_qc, profile_noqc, profile_raw):
    """ Parse variable data from all sources into a dictionary attached to the profile_qc structure
    """
    profile_qc.data = dict()

    # Location information
    profile_qc.data['LATITUDE'] = np.round(profile_qc.netcdf_file_obj['latitude'][0].__float__(), 4)
    profile_qc.data['LATITUDE_RAW'] = np.round(profile_noqc.netcdf_file_obj['latitude'][0].__float__(), 4)

    # check if scale factor has been applied, shouldn't have a negative longitude:
    lon = profile_qc.netcdf_file_obj['longitude'][0].__float__()
    if lon < 0:
        if profile_qc.netcdf_file_obj['longitude'].scale:
            LOGGER.info('Scale Factor in ed file longitude attributes, changing longitude value from  %s' % lon)
            lon = lon * -1
        else:
            LOGGER.error('Negative longitude value with no scale factor %s' % lon)

    # Change the 360 degree longitude to degrees_east (0-180, -180 to 0)
    if lon > 180:
        lon = lon - 360
    profile_qc.data['LONGITUDE'] = np.round(lon, 4)

    lon_raw = np.round(profile_noqc.netcdf_file_obj['longitude'][0].__float__(), 4)
    # Change the 360 degree longitude to degrees_east (0-180, -180 to 0)
    if lon_raw > 180:
        lon_raw = lon_raw - 360
    profile_qc.data['LONGITUDE_RAW'] = np.round(lon_raw, 4)

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
    # get the number of depths
    ndeps = profile_qc.netcdf_file_obj.variables['No_Depths'][:]
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
            dep = np.round(s.netcdf_file_obj.variables['Depthpress'][ivar, :], 2)
            depth_press_flag = s.netcdf_file_obj.variables['DepresQ'][ivar, :, 0].flatten()
            qc = np.ma.masked_array(
                invalid_to_ma_array(depth_press_flag, fillvalue=0))

            prof = np.ma.masked_values(
                np.round(s.netcdf_file_obj.variables['Profparm'][ivar, 0, :, 0, 0], 2),
                99.99)  # mask the 99.99 from CSA flagging of TEMP
            prof = np.ma.masked_invalid(prof)  # mask nan and inf values
            prof.set_fill_value(999999)

            prof_flag = s.netcdf_file_obj.variables['ProfQP'][ivar, 0, :, 0, 0].flatten()
            prof_flag = np.ma.masked_array(
                invalid_to_ma_array(prof_flag, fillvalue=99))  # replace masked values for IMOS IODE flags
            # if the size of the array isn't equal to the number of depths, adjust here
            if len(prof) != ndeps[ivar]:
                LOGGER.warning('Resizing arrays to the number of depths recorded in original MQNC file')
                prof = np.ma.resize(prof, ndeps[ivar])
                prof_flag = np.ma.resize(prof_flag, ndeps[ivar])
                dep = np.ma.resize(dep, ndeps[ivar])
                qc = np.ma.resize(qc, ndeps[ivar])
            df[var + depcode] = dep
            df[var + depcode + '_quality_control'] = qc
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
    df = df.dropna(subset=['TEMP', 'DEPTH', 'TEMP_RAW', 'DEPTH_RAW'], how='all')

    # check for duplicated depths and log if found
    if df['DEPTH'].duplicated().any():
        LOGGER.warning('Duplicated depths found in %s' % profile_qc.XBT_input_filename)

    # check for mismatch in DEPTH and DEPTH_RAW
    if not np.allclose(df['DEPTH'], df['DEPTH_RAW'], rtol=1e-3):
        LOGGER.warning('DEPTH and DEPTH_RAW are not the same in %s' % profile_qc.XBT_input_filename)

    # how many parameters do we have, not including DEPTH?
    profile_qc.nprof = len([col for col in df.columns if ('_quality_control' not in col and 'RAW'
                                                          not in col and 'DEPTH' not in col)])
    profile_noqc.nprof = profile_qc.nprof

    profile_qc.prof_type = decode_bytearray(profile_qc.netcdf_file_obj.variables['Prof_Type'][:]).strip()
    profile_noqc.prof_type = profile_qc.prof_type

    # save the dataframe of DEPTH dimensioned data to the profile object
    profile_qc.data['data'] = df

    return profile_qc, profile_noqc


def adjust_position_qc_flags(profile, profile_noqc):
    """ When a 'PE' flag is present in the Act_Code, the latitude and longitude qc flags need to be adjusted if not
    already set (applies to data processed with older versions of MQUEST)
    Also, if the temperature QC flags are not set correctly (3 for PER, 2 for PEA), these should be updated.
    """

    # exit this if we don't have a position code
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("LA|LO|PE")]) == 0:
        return profile

    # get the temperature QC codes
    df = profile.data['data']
    if profile.histories['HISTORY_QC_CODE'].str.contains('LAA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if np.round(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LAA'), 'HISTORY_PREVIOUS_VALUE'].values),
                    4) != np.round(profile.data['LATITUDE_RAW'], 4):
            LOGGER.error('LATITUDE_RAW not the same as the PREVIOUS_value!')
        if profile.data['LATITUDE_quality_control'] != 5:
            # PEA on latitude
            profile.data['LATITUDE_quality_control'] = 5
            LOGGER.info('LATITUDE correction (PEA) in original file, changing LATITUDE flag to level 5.')
            # change to flag 2 for temperature for all depths where qc is less than 2
            mask = df['TEMP_quality_control'] < 2
            df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('LOA').any():

        # if there are duplicated LOA flags in the histories, keep the one where hISTORY_PREVIOUS_VALUE matches the LONGITUDE_RAW value
        if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains('LOA')]) > 1:
            # get the rows with LOA flags
            loa_rows = profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains('LOA')]
            # do any of these PREVIOUS_VALUE match the profile_noqc.histories PREVIOUS_VALUE?
            if np.any(np.round(loa_rows['HISTORY_PREVIOUS_VALUE'].values, 4) == np.round(profile_noqc.histories.loc[
                                                                                             profile_noqc.histories[
                                                                                                 'HISTORY_QC_CODE'].str.contains(
                                                                                                 'LOA'), 'HISTORY_PREVIOUS_VALUE'].values,
                                                                                         4)):
                # get the row where the PREVIOUS_VALUE matches the profile_noqc PREVIOUS_VALUE
                loa_row = loa_rows[
                    np.round(loa_rows['HISTORY_PREVIOUS_VALUE'].values, 4) == np.round(profile_noqc.histories.loc[
                                                                                           profile_noqc.histories[
                                                                                               'HISTORY_QC_CODE'].str.contains(
                                                                                               'LOA'), 'HISTORY_PREVIOUS_VALUE'].values,
                                                                                       4)]
                LOGGER.info(
                    'Duplicate LOA flags in original file, keeping the one where PREVIOUS_VALUE matches the RAW PREVIOUS_VALUE')
                # drop the other rows
                profile.histories = profile.histories.drop(loa_rows.index.difference(loa_row.index))
            else:
                LOGGER.error('Duplicate LOA flags in original file, none match the RAW PREVIOUS_VALUE!')
                exit(1)

        # check HISTORY_PREVIOUS_VALUE matches the LONGITUDE_RAW value
        if np.round(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LOA'), 'HISTORY_PREVIOUS_VALUE'].values),
                    4) != np.round(profile.data['LONGITUDE_RAW'], 4) or (
        np.allclose(profile.data['LONGITUDE_RAW'], profile.data['LONGITUDE'], atol=0.1)):

            # check if the profile_noqc history has a similar value within tolerance
            if not np.allclose(float(profile.histories.loc[
                                         profile.histories['HISTORY_QC_CODE'].str.contains(
                                             'LOA'), 'HISTORY_PREVIOUS_VALUE'].values),
                               float(profile_noqc.histories.loc[
                                         profile_noqc.histories['HISTORY_QC_CODE'].str.contains(
                                             'LOA'), 'HISTORY_PREVIOUS_VALUE'].values), atol=0.1):
                LOGGER.error('LONGITUDE_RAW not the same as the PREVIOUS_VALUE or the RAW PREVIOUS_VALUE!')
            else:
                # set the LONGITUDE_RAW to the value in the noqc file
                profile.data['LONGITUDE_RAW'] = np.round(float(profile_noqc.histories.loc[
                                                                   profile_noqc.histories[
                                                                       'HISTORY_QC_CODE'].str.contains(
                                                                       'LOA'), 'HISTORY_PREVIOUS_VALUE'].values), 4)
        if profile.data['LONGITUDE_quality_control'] != 5:
            # PEA on longitude
            profile.data['LONGITUDE_quality_control'] = 5
            LOGGER.info('LONGITUDE correction (PEA) in original file, changing LONGITUDE flag to level 5.')
            # change to flag 2 for temperature for all depths where qc is less than 2
            mask = df['TEMP_quality_control'] < 2
            df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('PER').any():
        # PER on longitude and latitude
        profile.data['LONGITUDE_quality_control'] = 3
        profile.data['LATITUDE_quality_control'] = 3
        LOGGER.info('Position Reject (PER) in original file, changing LONGITUDE & LATITUDE flags to level 3.')
        # change to flag 3 for temperature for all depths where qc is less than 3
        mask = df['TEMP_quality_control'] < 3
        df.loc[mask, 'TEMP_quality_control'] = 3

    # update the temperature QC flags
    profile.data['data'] = df

    return profile


def adjust_time_qc_flags(profile):
    """ When a 'TE' flag is present in the Act_Code, the TIME_quality_control qc flag needs to be adjusted if not
    already set (applies to data processed with older versions of MQUEST"""

    # exit this if we don't have a TEA or TER code
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("TEA|TER")]) == 0:
        return profile

    # change temperature QC codes
    if profile.histories['HISTORY_QC_CODE'].str.contains('TEA').any() & profile.data['TIME_quality_control'] != 5:
        # TEA
        profile.data['TIME_quality_control'] = 5
        LOGGER.info('TIME correction (TEA) in original file, changing TIME flag to level 5.')
        # change to flag 2 for temperature for all depths where qc is less than 2

        profile.data['data'].loc[profile.data['data']['TEMP_quality_control'] < 2, 'TEMP_quality_control'] = 2
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if pd.to_datetime(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'TEA'), 'HISTORY_PREVIOUS_VALUE'].values, format='%Y%m%d%H%M%S') != \
                profile.data['TIME_RAW']:
            LOGGER.error('TIME_RAW not the same as the PREVIOUS_value!')

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
    # test probe
    if pt == 104:
        tunc = [0]
        dunc = [0]
    elif 1 <= pt <= 71:
        # Sippican probe type
        tunc = [0.1]
        dunc = [0.02, 4.6]
    elif 201 <= pt <= 252:
        # TSK probe type
        tunc = [0.15]
        dunc = [0.02, 4.6]
    elif 401 <= pt <= 501:
        # Sparton probe type
        tunc = [0.2]
        dunc = [0.02, 4.6]
    elif pt == 81 or pt == 281 or pt == 510:
        # AIRIAL XBT probe types
        tunc = [0.056]
        dunc = [0]  # no depth uncertainty determined
    elif 700 <= pt <= 751:
        # XCTDs
        year_value = profile.netcdf_file_obj.time.dt.year.astype(int).values[0]
        dt = datetime.datetime(year_value, 1, 1, 0, 0, 0)
        if dt < datetime.datetime.strptime('1998-01-01', '%Y-%m-%d'):
            tunc = [0.02]
            dunc = [0.04]
        else:
            tunc = [0.02]
            dunc = [0.02]
    else:
        # probe type not defined above, not in the code table 1770
        tunc = [0]
        dunc = [0]
    # temp uncertainties
    profile.data['data']['TEMP_uncertainty'] = ma.empty_like(profile.data['data']['TEMP'])
    profile.data['data']['TEMP_uncertainty'] = tunc[0]
    # depth uncertainties:
    unc = np.ma.MaskedArray(profile.data['data']['DEPTH'] * dunc[0], mask=False)
    if len(dunc) > 1:
        unc[profile.data['data']['DEPTH'] <= 230] = dunc[1]
    profile.data['data']['DEPTH_uncertainty'] = np.round(unc, 2)

    return profile


def get_fallrate_eq_coef(profile_qc, profile_noqc):
    """return probe type name, coef_a, coef_b as defined in WMO1770"""
    fre_list = read_section_from_xbt_config('FRE')
    peq_list = read_section_from_xbt_config('PEQ$')
    ptyp_list = read_section_from_xbt_config('PTYP')

    att_name = 'XBT_probetype_fallrate_equation'
    nms = [profile_qc, profile_noqc]
    vv = ['PROBE_TYPE', 'PROBE_TYPE_RAW']
    xx = ['XBT_fallrate_equation_coefficients', 'XBT_fallrate_equation_coefficients_RAW']
    ind = 0

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
                profile_qc.global_atts[vv[ind] + '_name'] = probetype
                profile_qc.global_atts[xx[ind]] = 'a: ' + coef_a + ', b: ' + coef_b
            else:
                profile_qc.global_atts[xx[ind]] = []
                profile_qc.data[vv[ind]] = []
                profile_qc.global_atts[vv[ind] + '_name'] = []
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

    df['HISTORY_QC_CODE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                             for xx in profile.netcdf_file_obj['Act_Code'][:].data if bytearray(xx).strip()]

    # nhist can sometimes be incorrect, so we need to check the length of the data
    if nhist != len(df['HISTORY_QC_CODE']):
        nhist = len(df['HISTORY_QC_CODE'])
        LOGGER.warning('Updating nhist to match length of history codes')

    df['HISTORY_INSTITUTION'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                                 for xx in profile.netcdf_file_obj['Ident_Code'][0:nhist].data]

    df['HISTORY_PARAMETER'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                               for xx in profile.netcdf_file_obj['Act_Parm'][0:nhist].data]
    df['HISTORY_SOFTWARE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                              for xx in profile.netcdf_file_obj['PRC_Code'][0:nhist].data]
    df['HISTORY_DATE'] = [''.join(chr(x) for x in bytearray(xx)).strip()
                          for xx in profile.netcdf_file_obj['PRC_Date'][0:nhist].data]
    df['HISTORY_START_DEPTH'] = profile.netcdf_file_obj['Aux_ID'][0:nhist].data
    df['HISTORY_TEMP_QC_CODE_VALUE'] = profile.netcdf_file_obj['Flag_severity'][0:nhist].data
    df['HISTORY_SOFTWARE_RELEASE'] = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in
                                      profile.netcdf_file_obj['Version'][0:nhist].data]

    dat = [float(x.replace(':', '')) for x in
           [''.join(chr(x) for x in bytearray(xx).strip()).rstrip('\x00')
            for xx in profile.netcdf_file_obj.variables['Previous_Val'][0:nhist]] if x]
    if dat:
        df['HISTORY_PREVIOUS_VALUE'] = dat
    else:
        df['HISTORY_PREVIOUS_VALUE'] = np.nan

    df = df.astype({'HISTORY_SOFTWARE_RELEASE': np.str_, 'HISTORY_QC_CODE': np.str_})

    # convert only the CSIRO codes, find any institution codes that are not 'CS'
    if not df['HISTORY_INSTITUTION'].str.contains('CS').all():
        LOGGER.warning('Institution code for some flags is not CSIRO, contains %s' %
                       df.loc[~df['HISTORY_INSTITUTION'].str.contains('CS'), 'HISTORY_INSTITUTION'].unique())
        # remove any codes that are not CSIRO
        df = df[df['HISTORY_INSTITUTION'].str.contains('CS')]
        nhist = len(df)

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

    # update variable names to match what is in the file
    names = {'DEPH': 'DEPTH', 'DATI': 'DATE, TIME', 'DATE': 'DATE', 'TIME': 'TIME', 'LATI': 'LATITUDE',
             'LONG': 'LONGITUDE', 'LALO': 'LATITUDE, LONGITUDE', 'TEMP': 'TEMP'}
    df['HISTORY_PARAMETER'] = df['HISTORY_PARAMETER'].map(names, na_action='ignore')
    if any(df['HISTORY_PARAMETER'].isna()):
        LOGGER.error("HISTORY_PARAMETER values - some are not defined. Please review output for this file")

    # update institute names to be more descriptive
    names = read_section_from_xbt_config('INSTITUTE')
    df['HISTORY_INSTITUTION'] = df['HISTORY_INSTITUTION'].map(lambda x: names[x].split(',')[0] if x in names else x)
    if any(df['HISTORY_INSTITUTION'].isna()):
        LOGGER.error("HISTORY_INSTITUTION values - some are not defined. Please review output for this file")

    # set the software value to 2.1 for CS and PE, RE flags
    df.loc[
        df.HISTORY_QC_CODE.isin(['CS', 'PE', 'RE']), ['HISTORY_SOFTWARE_RELEASE', 'HISTORY_SOFTWARE']] = '2.1', 'CSCBv2'

    # update software names to be more descriptive
    names = {'CSCB': 'CSIRO Quality control cookbook for XBT data v1.1',
             'CSCBv2': 'Australian XBT Quality Control Cookbook Version 2.1'}
    df['HISTORY_SOFTWARE'] = df['HISTORY_SOFTWARE'].map(names, na_action='ignore')

    if nhist > 0:

        # sort the flags by depth order to help with finding STOP_DEPTH
        # TODO: will keep the stop depth for now. Consider re-writing to loop over each of the lists of act_code types
        df = df.sort_values('HISTORY_START_DEPTH')
        dfdat = profile.data['data']
        for idx, row in df.iterrows():
            # Ensure start depth is the same as the value in the depth array
            # Find the closest value to the start depth in the histories
            ii = (dfdat['DEPTH'] - row['HISTORY_START_DEPTH']).abs().idxmin()
            df.at[idx, 'HISTORY_START_DEPTH'] = dfdat.at[ii, 'DEPTH']

            # QC,RE, TE, PE and EF flag applies to entire profile, stop_depth is deepest depth
            res = row['HISTORY_QC_CODE'] in act_code_full_profile
            if res:
                df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

            # if the flag is in act_code_single_point list, then stop depth is same as start
            res = row['HISTORY_QC_CODE'] in act_code_single_point
            if res:
                df.at[idx, "HISTORY_STOP_DEPTH"] = df.at[idx, 'HISTORY_START_DEPTH']

            # TODO: surface flags in the act_code_next_flag category need to ignore the CS flags
            # if the flag is in act_code_next_flag, then stop depth is the next depth or bottom
            # find next deepest flag depth
            res = row['HISTORY_QC_CODE'] in act_code_next_flag
            stop_idx = df['HISTORY_START_DEPTH'] > row['HISTORY_START_DEPTH']
            stop_depth = df['HISTORY_START_DEPTH'][stop_idx]
            if any(stop_idx) & res:
                ii = (np.abs(dfdat['DEPTH'] - stop_depth.values[0])).argmin()
                df.at[idx, "HISTORY_STOP_DEPTH"] = dfdat['DEPTH'][ii]
            elif res:  # if there isn't a deeper flag, use deepest depth
                df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

        # append the 'A' or 'R' to each code
        for idx, row in df.iterrows():
            if df.at[idx, 'HISTORY_TEMP_QC_CODE_VALUE'] in [0, 1, 2, 5]:
                df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'A'
            else:
                df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'R'

        # change CSA to CSR and the flag to 3 to match new format
        df.loc[(df['HISTORY_QC_CODE'].str.contains('CSA')),
        ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'CSR', 3

        # this group of changes is here because I have reviewed all our QC codes in the historic databases and I know
        # there are some that are not correct. This is a one off change to correct them. Could be done more elegantly probably.

        # change ERA to PLA with flag 3 to reduce duplication of flags
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('ERA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'PLA', 3

        # change any REA or RER flags to REA and flag 0 to match new format
        df.loc[(df['HISTORY_QC_CODE'].str.contains('RE')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'REA', 0

        # change any NGA flags to NGR and flag 4
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('NGA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'NGR', 4

        # change any NTA flags to NTR and flag 4
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('NTA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'NTR', 4

        # change any TPA flags to TPR and flag 4
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('TPA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'TPR', 4

        # change any WBA flags to WBR and flag 4
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('WBA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'WBR', 4

        # change URA for BDA and flag 2
        df.loc[
            (df['HISTORY_QC_CODE'].str.contains('URA')), ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'BDA', 2
        # all BDA flags should be set to 2, historically have been 1, but as low res, make them 2
        df.loc[(df['HISTORY_QC_CODE'].str.contains('BDA')), 'HISTORY_TEMP_QC_CODE_VALUE'] = 2

        # change any FSR flags to FSA and flag 2, but first confirm that the TEMP_QC_CODE_VALUE is 2 at the same depth as FS
        if df['HISTORY_QC_CODE'].str.contains('FSR').any():
            # checking the QC values below the deepest CS flag
            idepth = df.loc[df['HISTORY_QC_CODE'].str.contains('CS'), 'HISTORY_START_DEPTH'].values.max() + 1
            # check the TEMP_QC_CODE_VALUE is 2 at the same depth as FS
            if len(idepth) > 0:
                if profile.data['data'].loc[
                    profile.data['data']['DEPTH'] == idepth[0], 'TEMP_quality_control'].values != 2:
                    df.loc[(df['HISTORY_QC_CODE'].str.contains('FSR')), ['HISTORY_QC_CODE',
                                                                         'HISTORY_TEMP_QC_CODE_VALUE']] = 'FSA', 2
                else:
                    LOGGER.error('TEMP_QC_CODE_VALUE is not 2 at the same depth as FSR flag, not changing it to FSA')
                    print(profile.XBT_input_filename)

        # Change the PEA flag to LA or LO and ensure the TEMP_QC_CODE_VALUE is set to 2, not 5
        df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
                (df['HISTORY_PARAMETER'].str.contains('LATITUDE'))),
        ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'LAA', 2
        df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
                (df['HISTORY_PARAMETER'].str.contains('LONGITUDE'))),
        ['HISTORY_QC_CODE', 'HISTORY_TEMP_QC_CODE_VALUE']] = 'LOA', 2

        # Combine duplicated TEA flags to a single TEA for TIME variable TEMP_QC_CODE_VALUE is set to 2, not 5
        # Also change just DATE TEA flags to TIME
        dfTEA = df[df['HISTORY_QC_CODE'] == 'TEA'].copy()
        if len(dfTEA) > 0:
            # get the date value from the TIME variable
            dtt = profile.data['TIME'].strftime('%Y%m%d')
            # get the TIME value from the TIME variable
            ti = profile.data['TIME'].strftime('%H%M%S')

            # is there a 'TIME' parameter in the TEA flags?
            timerows = dfTEA[dfTEA['HISTORY_PARAMETER'] == 'TIME'].copy()
            # include the date information
            timerows.loc[:, 'HISTORY_PREVIOUS_VALUE'] = timerows['HISTORY_PREVIOUS_VALUE'].apply(
                lambda x: dtt + str(int(x)) + '00').astype(float)

            # now check for any 'DATE' parameter in the TEA flags
            daterows = dfTEA[dfTEA['HISTORY_PARAMETER'] == 'DATE'].copy()
            try:
                daterows.loc[:, 'HISTORY_PREVIOUS_VALUE'] = daterows['HISTORY_PREVIOUS_VALUE'].apply(
                    lambda x: datetime.strptime(str(int(x)), '%Y%m%d').strftime('%Y%m%d') + ti).astype(float)
            except:
                daterows.loc[:, 'HISTORY_PREVIOUS_VALUE'] = daterows['HISTORY_PREVIOUS_VALUE'].apply(
                    lambda x: datetime.strptime(str(int(x)), '%d%m%Y').strftime('%Y%m%d') + ti).astype(float)

            # update the df with the new values
            df.update(timerows)
            df.update(daterows)

            # change the 'DATE' label to TIME  and update the TEA PREVIOUS_VALUE to the new datetime value
            df.loc[((df['HISTORY_PARAMETER'].str.contains('DATE') | df['HISTORY_PARAMETER'].str.contains('TIME')) &
                    (df['HISTORY_QC_CODE'].str.contains('TEA'))), ['HISTORY_PARAMETER']] = 'TIME'

            # remove any duplicated lines
            df = df[~(df.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_PREVIOUS_VALUE'])
                      & df.HISTORY_PARAMETER.eq('TIME'))]

    # assign the dataframe back to profile at this stage
    profile.histories = df

    # only do this next step if not a noqc file, won't have TEMP data
    if 'TEMP' in profile.data['data'].columns:
        # make our accept and reject code variables
        profile = create_flag_feature(profile)

    return profile


def combine_histories(profile_qc, profile_noqc):
    # check for global attributes in the noqc file and update the global atts as required
    # handle the longitude change where data was imported from dataset with a negative longitude where it should
    # have been positive. The *raw.nc previous value and *ed.nc previous value should be the same, update the LONG_RAW.
    if len(profile_noqc.histories) > 0:
        # copy this information to the LONGITUDE_RAW value if it isn't the same
        if 'LO' in profile_noqc.histories['HISTORY_QC_CODE'].values:
            if np.round(profile_noqc.histories.loc[profile_noqc.histories['HISTORY_QC_CODE'].str.contains('LO'),
            'HISTORY_PREVIOUS_VALUE'], 4).values != np.round(
                profile_qc.data['LONGITUDE_RAW'], 4):
                LOGGER.warning('Updating raw longitude to match the previous value in *raw.nc file')
                profile_qc.data['LONGITUDE_RAW'] = profile_noqc.histories.loc[
                    profile_noqc.histories['HISTORY_QC_CODE'].str.contains('LO'), 'HISTORY_PREVIOUS_VALUE'][0]
    # TODO: handle other extra histories in noqc file here:
    if len(profile_noqc.histories) > 1:
        print('QC flags and codes in the raw file')

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

    return profile


def restore_temp_val(profile):
    """
    Restore the temperature values that are associated with
    the 'CS' (surface spike removed) flag. That means identifying them, putting them back into the
    TEMP field, then putting a flag of 3 (probably bad) on them. The values can also stay
    in the HISTORY_PREVIOUS_VALUE field. This process would need to apply to both the TEMP
    and TEMP_RAW (from the *raw.nc file).
    """

    # index of CS flags in histories:
    idx = profile.histories['HISTORY_QC_CODE'] == 'CSR'
    depths = profile.histories['HISTORY_START_DEPTH'][idx].values.astype('float')
    temps = profile.histories['HISTORY_PREVIOUS_VALUE'][idx].values.astype('float')

    # check if the temperature values are missing & replace with previous value if they are:
    # do for both TEMP and TEMP_RAW
    df = profile.data['data']
    # find the depths in the profile data
    ind = np.in1d(np.round(df['DEPTH'], 2), np.round(depths, 2)).nonzero()[0]
    # temps should be equal to df['TEMP_RAW'][ind], let's check they are equal
    if (temps != df['TEMP_RAW'][ind]).all():
        # check they are within 0.01 of each other
        if not np.allclose(temps, df['TEMP_RAW'][ind], atol=0.01):
            # check the median difference with a bigger tolerance:
            if np.median(np.abs(temps - df['TEMP_RAW'][ind])) > 0.01:
                LOGGER.error('TEMP_RAW values do not match the HISTORY_PREVIOUS_VALUE for CS flags')
                exit(1)
            else:
                # update the HISTORY_PREVIOUS_VALUE to the TEMP_RAW value
                profile.histories.loc[idx, 'HISTORY_PREVIOUS_VALUE'] = df['TEMP_RAW'][ind]
                LOGGER.info('Updated HISTORY_PREVIOUS_VALUE for CS flags')
        else:
            # update the HISTORY_PREVIOUS_VALUE to the TEMP_RAW value
            profile.histories.loc[idx, 'HISTORY_PREVIOUS_VALUE'] = df['TEMP_RAW'][ind]
            LOGGER.info('Updated HISTORY_PREVIOUS_VALUE for CS flags')
    # update the TEMP values
    df.loc[ind, 'TEMP'] = df.loc[ind, 'TEMP_RAW']
    # update profile data
    profile.data['data'] = df

    return profile


def create_flag_feature(profile):
    """ Take the existing QC code values and turn them into a integer representation. One bit for every code.
    And there are now two variables, one for accept codes, one for reject codes."""

    # set up a dataframe of the codes and their values
    # codes from the new cookbook, read from csv file
    # Specify the file path
    a_file_path = os.path.join(os.path.dirname(__file__), 'xbt_accept_code.csv')
    r_file_path = os.path.join(os.path.dirname(__file__), 'xbt_reject_code.csv')

    # Read the CSV file and convert it to a DataFrame
    dfa = pd.read_csv(a_file_path)
    dfr = pd.read_csv(r_file_path)
    # merge the two dataframes
    df = pd.concat([dfa, dfr])

    df_data = profile.data['data'].copy(deep=True)

    # set the fields to zeros to start
    df_data['XBT_accept_code'] = 0
    df_data['XBT_reject_code'] = 0
    df_data['tempqc'] = 0

    # perform the flag mapping on the original flags and create the two new variables
    codes = profile.histories
    # if the TEMP_quality_control values are 0 and the TEMP_RAW_quality_control values are not, update the TEMP_quality_control
    # values to be the same as the TEMP_RAW_quality_control values
    idx = (df_data['TEMP_quality_control'] == 0) & (df_data['TEMP_RAW_quality_control'] != 0)
    if idx.any():
        LOGGER.warning('TEMP_quality_control values are 0 and TEMP_RAW_quality_control values are not. Updating.')
        df_data.loc[idx, 'TEMP_quality_control'] = df_data.loc[idx, 'TEMP_RAW_quality_control']
        # add QCA to the history
        codes = codes._append({'HISTORY_INSTITUTION': profile.global_atts['institution'],
                               'HISTORY_QC_CODE': 'QCA',
                               'HISTORY_PARAMETER': 'TEMP',
                               'HISTORY_SOFTWARE': 'Unknown',
                               'HISTORY_DATE': profile.data['TIME'].strftime('%Y-%m-%d %H:%M:%S'),
                               'HISTORY_START_DEPTH': df_data['DEPTH'].values[0],
                               'HISTORY_STOP_DEPTH': df_data['DEPTH'].values[-1],
                               'HISTORY_QC_CODE_DESCRIPTION': 'scientific_qc_applied',
                               'HISTORY_TEMP_QC_CODE_VALUE': 1,
                               'HISTORY_SOFTWARE_RELEASE': '',
                               'HISTORY_PREVIOUS_VALUE': 0}, ignore_index=True)

    # merge the codes with the flag codes
    mapcodes = pd.merge(df, codes, how='right', left_on='code', right_on='HISTORY_QC_CODE')

    if mapcodes.empty:
        profile.global_atts['qc_completed'] = 'no'
        return profile
    else:
        # adjust global attribute to say we have done scientific QC
        profile.global_atts['qc_completed'] = 'yes'

    # update the HISTORY_QC_CODE_DESCRIPTION to the df label
    mapcodes['HISTORY_QC_CODE_DESCRIPTION'] = mapcodes['label']

    # any flags not included? check for nan in the label column
    nan_values = mapcodes['label'].isna()
    if nan_values.any():
        # we have an extra flag that we haven't coded
        # if any of the flags are in this list which I know about, remove them
        if mapcodes.loc[nan_values, 'HISTORY_QC_CODE'].str.contains("BB|DC|GS|MS"):
            mapcodes = mapcodes[~nan_values]
        else:
            LOGGER.error('New QC code encountered, please code in the new value')

    # check for duplicated history codes at the same depth so we don't duplicate the QC code in the fft variable
    # this will keep the first value. If the PREVIOUS_VALUE is 99.99 and it is in the first position, it will be kept
    # however, we just want to check that the previous_values are the same as the TEMP_RAW values and if not, do something
    # first sort by start_depth and then previous_value to try and eliminate the 99.99 values
    mapcodes = mapcodes.sort_values(['HISTORY_START_DEPTH', 'HISTORY_PREVIOUS_VALUE'])
    dup_df = mapcodes[mapcodes.duplicated(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)]
    if len(dup_df) > 0:
        mapcodes = mapcodes.drop_duplicates(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'],
                                      keep='first')
        LOGGER.warning('Duplicate QC code encountered, and removed for flag_feature_type array. Please review')

    # now need to assign the codes to the correct depths.
    # code only added in one location at the start depth, QC flags indicate the quality applied
    # for each code, need an array of values same size as DEPTH, then add them all together
    # also check the TEMP_QC_CODE_VALUE is the same as the actual flag in the flag array

    # create a df with the same number of columns as the number of rows in the mapcodes table and number of rows is number of depths
    tempdf = pd.DataFrame(np.zeros((len(df_data), len(mapcodes))) * np.zeros(len(mapcodes)), columns=mapcodes['code'])

    # iterate over the mapcodes table and fill a column in tempdf with QC values from the tempqc field
    for idx, row in mapcodes.iterrows():
        # get the index of the depth in the data
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        # if this is a CSR flag, just fill the depth with the tempqc value
        if row['HISTORY_QC_CODE'] == 'CSR':
            tempdf.loc[ii, row['code']] = row['tempqc']
        else:
            # fill the tempdf from the depth index to the maximum index
            tempdf.loc[ii:, row['code']] = row['tempqc']
        # for flags that have been interpolated or filtered, these are 5 and 2 deeper. Change the flag at these depths to 5
        if row['HISTORY_QC_CODE'] in ['LAA', 'LOA', 'SPA', 'HFA', 'TEA', 'IPA']:
            # 2 should have been assigned above, now just overwriting with 5
            tempdf.loc[ii, row['code']] = 5

    # index of the tempdf rows that have a value of 5
    idx = tempdf.eq(5).any(axis=1)
    # calculate the maximum tempqc value for each depth
    tempdf['tempqc'] = tempdf.max(axis=1)
    # overwrite the tempqc value with 5 where there is a 5 in the tempdf
    tempdf.loc[idx, 'tempqc'] = 5

    # update the TEMP_quality_control field with the tempdf values
    df_data['TEMP_quality_control'] = tempdf['tempqc']

    # Iterate over the history table.
    for idx, row in mapcodes.iterrows():
        # Get depth index
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        # if this is an accept code (QC_Flag = 1, 2, 3, 5) then add it to the accept code array
        if row['HISTORY_TEMP_QC_CODE_VALUE'] in [0, 1, 2, 5]:
            # adding them together - is there a more correct way to do this?
            # Add byte values (masks) for accept codes
            df_data.loc[ii, 'XBT_accept_code'] = df_data.loc[ii, 'XBT_accept_code'] + np.float64(row['byte_value'])
        else:
            # Add byte values (masks) for reject codes
            df_data.loc[ii, 'XBT_reject_code'] = df_data.loc[ii, 'XBT_reject_code'] + np.float64(row['byte_value'])

    # update the histories with the correct tempqc values from mapcodes
    mapcodes['HISTORY_TEMP_QC_CODE_VALUE'] = mapcodes['tempqc']
    # drop unwanted columns
    mapcodes = mapcodes.drop(columns=['tempqc', 'byte_value', 'label', 'code'])
    df_data = df_data.drop(columns=['tempqc'])

    # update the histories
    profile.histories = mapcodes
    # update the profile data
    profile.data['data'] = df_data

    # make sure the previous_values are the same as the data['TEMP_RAW'] values and replace missing TEMP values at CS
    profile = restore_temp_val(profile)

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
                 for xx in profile.netcdf_file_obj['Act_Code'][0:nhist].data]
    depth = np.round(profile.netcdf_file_obj.variables['Depthpress'][:], 2)

    if np.sum(~depth.mask) == 0:
        LOGGER.error('No data in the file')
        return False

    if data_type != 'XB':  # and data_type != 'BA' and data_type != 'TE':
        LOGGER.error('Profile not processed as it is type %s' % data_type)
        return False

    if duplicate_flag == 'D':
        LOGGER.error('Profile not processed. Tagged as duplicate in original netcdf file')
        return False

    if 'DU' in histcodes:
        LOGGER.error('Profile not processed. Tagged as test probe in original netcdf file')
        return False

    data_vars = temp_prof_info(profile.netcdf_file_obj)
    if 'TEMP' not in data_vars.values():
        LOGGER.error('Profile not processed, no TEMP in file.')
        return False

    return True


def make_dataframe(profile_ed, profile_raw, profile_turo):
    # convert the data in profile to a parquet file
    # create a dataframe from the profile data
    df = pd.DataFrame(profile_ed.data['data'])
    # add the other data to the dataframe
    for key, value in profile_ed.data.items():
        # skip the data dataframe, already included
        if key == 'data':
            continue
        df[key] = value
    # make a global attributes dataframe
    gdf = pd.DataFrame(profile_ed.global_atts, index=[0])

    # add the raw global attributes to the dataframe
    for key, value in profile_raw.global_atts.items():
        gdf[key + '_RAW'] = value

    return df, gdf


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
    elif [index for index, item in enumerate(xbt_config.sections()) if section_name in item]:
        index = [index for index, item in enumerate(xbt_config.sections()) if section_name in item][0]
        return dict(xbt_config.items(xbt_config.sections()[index]))
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

    # make an empty dataframe to collect all the data
    dfall = pd.DataFrame()
    # make a second dataframe to hold the histories
    dfhist = pd.DataFrame()
    # and another dataframe to hold the global attributes
    globsall = pd.DataFrame()

    for f in keys.data['station_number']:
        # if f != 88946079:
        #       continue
        fpath = '/'.join(re.findall('..?', str(f))) + 'ed.nc'
        fname = os.path.join(keys.dbase_name, fpath)
        # make input_filename here
        input_filename = os.path.join(os.path.basename(keys.dbase_name), fpath)

        # if the file exists, let's make a profile object with all the
        # data and metadata attached.

        if os.path.isfile(fname):
            # read the edited profile
            profile_ed = XbtProfile(fname, input_filename)
            # read the raw profile
            profile_raw = XbtProfile(fname.replace('ed.nc', 'raw.nc'), input_filename.replace('ed.nc', 'raw.nc'))
            # TODO: check the keys data (date/time/lat/long etc) against what is in the data file
            # TODO: find the matching TURO profile if it is available:
            # profile_turo = turoProfile(profile_ed)
            profile_turo = []

            # now write it out to the new netcdf format
            if check_nc_to_be_created(profile_ed):
                # for example where depths are different, metadata is different etc between the ed and raw files.
                profile_ed = coordinate_data(profile_ed, profile_raw, profile_turo)
                profile_df, globals_df = make_dataframe(profile_ed, profile_raw, profile_turo)
                # add the station number to the dataframe
                profile_df['station_number'] = f
                globals_df['station_number'] = f
                # add to the big dataframes
                dfall = pd.concat([dfall, profile_df], ignore_index=True)
                globsall = pd.concat([globsall, globals_df], ignore_index=True)
                # add station number to the histories
                profile_ed.histories['station_number'] = f
                # add the histories to the big dataframe
                dfhist = pd.concat([dfhist, profile_ed.histories], ignore_index=True)
        else:
            LOGGER.warning('file %s is in keys file, but does not exist' % f)
    # write the dataframe to a parquet file
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name), os.path.basename(keys.dbase_name) + '.parquet')
    dfall.to_parquet(pq_filename, index=False)
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name),
                               os.path.basename(keys.dbase_name) + '_histories.parquet')
    dfhist.to_parquet(pq_filename, index=False)
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name), os.path.basename(keys.dbase_name) + '_globals.parquet')
    globsall.to_parquet(pq_filename, index=False)

    print('All done')