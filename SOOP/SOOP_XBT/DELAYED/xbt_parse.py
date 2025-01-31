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
import json
import pyarrow as pa
import pyarrow.parquet as pq
# from local directory
from xbt_utils import *


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
        # LOGGER.info('Parsing %s' % self.XBT_input_filename)
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
            latitude = np.round(netcdf_file_obj['obslat'][:].data, 6)
            longitude = np.round(netcdf_file_obj['obslng'][:].data, 6)
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

    # assign the geospatial_vertical* to the no_qc file for checking consistency. Doesn't get assigned in previous call
    # because the data doesn't get written to the noqc profile
    profile_noqc.global_atts['geospatial_vertical_max'] = max(profile_qc.data['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_vertical_min'] = min(profile_qc.data['DEPTH_RAW'])
    profile_noqc.global_atts['geospatial_lat_max'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lat_min'] = profile_qc.data['LATITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_max'] = profile_qc.data['LONGITUDE_RAW']
    profile_noqc.global_atts['geospatial_lon_min'] = profile_qc.data['LONGITUDE_RAW']
    profile_noqc.global_atts['time_coverage_start'] = profile_qc.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")
    profile_noqc.global_atts['time_coverage_end'] = profile_qc.data['TIME'].strftime("%Y-%m-%dT%H:%M:%SZ")

    # let's check if there are histories to parse and then handle
    profile_qc = parse_histories_nc(profile_qc)
    if int(profile_noqc.netcdf_file_obj['Num_Hists'][0].data) == 0:
        # if there are no histories in the noqc file, apply an empty dataframe with the same columns as profile_qc.histories
        profile_noqc.histories = pd.DataFrame(columns=profile_qc.histories.columns)
    else:
        # we need to carry the depths information into the history parsing, so copy the data array into profile_noqc
        profile_noqc.data['DEPTH'] = profile_qc.data['DEPTH_RAW']
        profile_noqc.data['TEMP_quality_control'] = profile_qc.data['TEMP_RAW_quality_control']
        profile_noqc = parse_histories_nc(profile_noqc)
    # check for histories in the noqc file and reconcile:
    profile_qc = combine_histories(profile_qc, profile_noqc)

    # make our accept and reject code variables
    profile_qc = create_flag_feature(profile_qc)

    # next section, only if there are QC flags present
    if len(profile_qc.histories) > 0:

        # adjust lat lon qc flags if required
        profile_qc = adjust_position_qc_flags(profile_qc)
        # adjust date and time QC flags if required
        profile_qc = adjust_time_qc_flags(profile_qc)

    # perform a check of the qc vs noqc global attributes and histories. Do any of these need reconciling?
    if len(profile_qc.global_atts.keys() - profile_noqc.global_atts):
        # if the difference in the global attributes is just the qc_completed key, continue
        if len(profile_qc.global_atts.keys() - profile_noqc.global_atts) == 1:
            if 'qc_completed' in profile_qc.global_atts.keys() - profile_noqc.global_atts:
                pass
            else:
                LOGGER.error('%s GLOBAL attributes in RAW and ED files are not consistent. Please review.'
                             % profile_qc.XBT_input_filename)

    # Probe type goes into a variable with coefficients as attributes, and assign QC to probe types
    profile_qc = get_fallrate_eq_coef(profile_qc, profile_noqc)
    # if probetype is not XBT return empty profile_qc
    if profile_qc.data['PROBE_TYPE'] == '':
        return []

    # check that the sums of TEMP and TEMP_RAW and DEPTH and DEPTH_RAW are the same within a tolerance
    # check_sums_of_temp_depth(profile_qc)

    # add uncertainties:
    profile_qc = add_uncertainties(profile_qc)

    # remove columns that are all NaN
    profile_qc.data = profile_qc.data.dropna(axis=1, how='all')

    return profile_qc


def get_recorder_type(df):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')
    syst_list = read_section_from_xbt_config('SYST')

    item_val = str(df['XBT_recorder_type'].unique().item())
    #        if item_val in list(syst_list.keys()):
    #            item_val = syst_list[item_val].split(',')[0]

    if item_val in list(rct_list.keys()):
        return item_val, rct_list[item_val].split(',')[0]
    else:
        LOGGER.warning(
            '{item_val} missing from recorder type part in xbt_config file, using unknown for recorder. %s'.format(
                item_val=item_val) % df['XBT_input_filename'].unique().item())
        item_val = '99'
        return item_val, rct_list[item_val].split(',')[0]


def parse_extra_vars(profile_qc, profile_noqc):
    """
    retrieve global attributes from input NetCDF file object
    """
    dataf = profile_qc.data
    vars_list = read_section_from_xbt_config('VARIABLES')
    for profile in [profile_qc, profile_noqc]:
        for var in list(vars_list.keys()):
            if var in list(profile.netcdf_file_obj.variables.keys()):
                var_name = vars_list[var].split(',')[0]
                var_type = vars_list[var].split(',')[1]
                vv = decode_bytearray(profile.netcdf_file_obj[var][:])
                if not vv or len(vv) == 0:
                    dataf[var_name] = ''
                else:
                    data = remove_control_chars(vv).strip()
                    try:
                        if var_type == 'float':
                            data = float(data.replace(' ', ''))
                        elif var_type == 'int':
                            data = int(data.replace(' ', ''))
                        else:
                            data = data.replace(' ', '')
                    except ValueError:
                        LOGGER.warning(
                            '"%s = %s" could not be converted to %s(). Please review. %s' % (var_name, data, var_type.upper(),
                                                                                          profile.XBT_input_filename))
                    # if the variable is institution, create a dictionary of the institution codes
                    if var == 'institution':
                        institute_list = read_section_from_xbt_config('INSTITUTE')
                        if data in list(institute_list.keys()):
                            dataf[var_name] = institute_list[data].split(',')[0]
                            dataf['Agency_GTS_code'] = institute_list[data].split(',')[1]
                        else:
                            LOGGER.warning('Agency_GTS_code code %s is not defined in xbt_config file. Please edit xbt_config %s'
                                           % (data, profile.XBT_input_filename))
                    if var == 'Digit_Code' or var == 'Standard':
                        for count in range(profile.nprof):
                            vv = decode_bytearray(profile.netcdf_file_obj[var][count])
                            if not vv or len(vv) == 0:
                                dataf[var_name + '_' + profile.prof_type[count]] = ''
                            else:
                                dataf[var_name + '_' + profile.prof_type[count]] = remove_control_chars(vv).strip()
            else:
                dataf[var_name] = ''

        # include the input filename
        dataf['XBT_input_filename'] = profile.XBT_input_filename
        # and date created
        dataf['date_created'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # create global attributes
        profile.global_atts = {'geospatial_vertical_units': 'meters', 'geospatial_vertical_positive': 'down'}
        # add geospatial information to global attributes
        try:
            profile.global_atts['geospatial_lat_max'] = np.unique(dataf['LATITUDE']).item()
            profile.global_atts['geospatial_lat_min'] = np.unique(dataf['LATITUDE']).item()
            profile.global_atts['geospatial_lon_max'] = np.unique(dataf['LONGITUDE']).item()
            profile.global_atts['geospatial_lon_min'] = np.unique(dataf['LONGITUDE']).item()
            profile.global_atts['geospatial_vertical_max'] = max(dataf['DEPTH'])
            profile.global_atts['geospatial_vertical_min'] = min(dataf['DEPTH'])
            # include time_coverage_start and time_coverage_end in the global attributes
            profile.global_atts['time_coverage_start'] = dataf['TIME'].unique().strftime("%Y-%m-%dT%H:%M:%SZ").item()
            profile.global_atts['time_coverage_end'] = dataf['TIME'].unique().strftime("%Y-%m-%dT%H:%M:%SZ").item()
        except:
            profile.global_atts['geospatial_lat_max'] = []
            profile.global_atts['geospatial_lat_min'] = []
            profile.global_atts['geospatial_lon_max'] = []
            profile.global_atts['geospatial_lon_min'] = []
            profile.global_atts['geospatial_vertical_max'] = []
            profile.global_atts['geospatial_vertical_min'] = []
            profile.global_atts['time_coverage_start'] = []
            profile.global_atts['time_coverage_end'] = []

        # Parse the surface codes into the variables too
        srfc_code_nc = profile.netcdf_file_obj['SRFC_Code'][:]
        srfc_parm = profile.netcdf_file_obj['SRFC_Parm'][:]
        nsrf_codes = int(profile.netcdf_file_obj['Nsurfc'][:])

        srfc_code_list = read_section_from_xbt_config('SRFC_CODES')

        # read a list of srfc code defined in the srfc_code conf file. Create a
        # dictionary of matching values
        missing_codes = []
        for i in range(nsrf_codes):
            srfc_code_iter = decode_bytearray(srfc_code_nc[i])
            if srfc_code_iter in list(srfc_code_list.keys()):
                att_name = srfc_code_list[srfc_code_iter].split(',')[0]
                att_type = srfc_code_list[srfc_code_iter].split(',')[1]
                att_val = decode_bytearray(srfc_parm[i])
                try:
                    if att_type == 'float':
                        dataf[att_name] = float(att_val.replace(' ', ''))
                    elif att_type == 'int':
                        dataf[att_name] = int(att_val.replace(' ', ''))
                    elif att_type == 'date':
                        dataf[att_name] = datetime.strptime(att_val.replace(' ', ''), '%Y%m%d')
                    else:
                        dataf[att_name] = att_val.replace(' ', '')
                except ValueError:
                    LOGGER.warning(
                        '"%s = %s" could not be converted to %s(). Please review. %s' % (
                        att_name, att_val, att_type.upper()), profile.XBT_input_filename)
            else:
                if srfc_code_iter != '' and srfc_code_iter != 'IOTA':
                    # collect the code in a list for the user to review
                    missing_codes.append(srfc_code_iter)

        if missing_codes:
                LOGGER.warning('%s codes not defined in srfc_code in xbt_config file. Please edit xbt_config %s'
                               % (missing_codes, profile.XBT_input_filename))

        # if the platform code didn't come through, assign unknown type
        if 'Platform_code' not in dataf.columns:
            LOGGER.warning('PLATFORM_CODE is missing, GCLL has not been read or is missing. %s' % profile.XBT_input_filename)
            # assign unknown to the platform code
            dataf['Platform_code'] = 'Unknown'
            dataf['ship_name'] = 'Unknown'
            dataf['ship_IMO'] = 'Unknown'

        # get the ship details
        # note that the callsign and ship name are filled from the original file values, but will be replaced here if they exist in the AODN vocabulary
        # for these older historical files, the Callsign and Platform_code are the same. In newer files, the platform_code
        # will be the GTSID or SOTID.
        dataf['Callsign'] = dataf[
            'Platform_code']  # set here as can't have duplicate assignments in the config file
        ships = SHIP_CALL_SIGN_LIST
        calls = dataf['Platform_code'].unique().item()
        if calls in ships:
            dataf['Ship_name'] = ships[calls][0]
            dataf['Ship_IMO'] = ships[calls][1]
        elif difflib.get_close_matches(calls, ships, n=1, cutoff=0.8) != []:
            dataf['Callsign'] = \
                difflib.get_close_matches(calls, ships, n=1, cutoff=0.8)[0]
            dataf['Ship_name'] = ships[dataf['Callsign'].unique().item()][0]
            dataf['Ship_IMO'] = ships[dataf['Callsign'].unique().item()][1]
            LOGGER.warning(
                'PLATFORM_CODE: Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s %s' % (
                    dataf['Platform_code'].unique().item(), dataf['Callsign'].unique().item(), profile.XBT_input_filename))
        else:
            dataf['Platform_code'] = 'Unknown'
            dataf['Ship_name'] = 'Unknown'
            dataf['Ship_IMO'] = 'Unknown'

        # extract the information and assign correctly
        if 'XBT_recorder_type' in dataf.columns:
            recorder_val, recorder_type = get_recorder_type(dataf)
            dataf['XBT_recorder_type_name'] = recorder_type
        else:
            dataf['XBT_recorder_type_name'] = 'Unknown'
            dataf['XBT_recorder_type'] = '99'

        # check deployment height
        if 'XBT_height_launch_above_water' in dataf.columns:
            if dataf['XBT_height_launch_above_water'].unique().item() > 50:
                LOGGER.warning('HTL$, xbt launch height attribute seems to be very high. Please review: %s meters %s' %
                               (dataf['XBT_height_launch_above_water'].unique().item(), profile.XBT_input_filename))

        # some files don't have line information
        line = dataf['XBT_line'].unique().item()
        if not line:
            line = 'NOLINE'
            dataf['XBT_line'] = 'NOLINE'
            LOGGER.warning('XBT line is not recorded, assigning NOLINE %s' %
                           profile.XBT_input_filename)

        xbt_line_codes = [s for s in list(XBT_LINE_INFO.keys())]  # IMOS codes taken from vocabulary
        if line in xbt_line_codes:
            xbt_line_att = XBT_LINE_INFO[line]
            dataf['XBT_line_description'] = xbt_line_att[1]
        else:
            LOGGER.error(
                'XBT line : "%s" is not defined in AODN vocabs.ands.org.au(contact AODN) %s' %
                (line, profile.XBT_input_filename))

    return profile_qc, profile_noqc


def parse_data_nc(profile_qc, profile_noqc, profile_raw):
    """ Parse variable data from all sources into a dictionary attached to the profile_qc structure
    """
    # create column headers from the variable names in generate_nc_file_att file
    meta, variable_names = generate_table_att(os.path.join(os.path.dirname(__file__), 'generate_nc_file_att'))
    # remove variable_names that contain 'HISTORY' as these are not data columns
    variable_names = [x for x in variable_names if 'HISTORY' not in x]
    # create profile_qc.data
    profile_qc.data = pd.DataFrame(columns=variable_names)

    # Pressure/depth information from both noqc and qc files
    for s in [profile_qc, profile_noqc]:
        # assign '_RAW' if s is profile_noqc, otherwise assign ''
        raw = '_RAW' if s == profile_noqc else ''
        # get the number of depths
        ndeps = s.netcdf_file_obj.variables['No_Depths'][:][0]
        # cycle through the variables identified in the file, for XBT files, this should only be TEMP:
        data_vars = temp_prof_info(s.netcdf_file_obj)
        # assign the data_vars to the profile object
        s.prof_type = list(data_vars.values())

        if len(data_vars) > 1:
            LOGGER.error('Profile contains %s variables and is not an XBT %s' % (data_vars, s.XBT_input_filename))
            exit(1)
        # should only be one variable, TEMP, but leave as a loop for future proofing
        for ivar, var in data_vars.items():
            # we want the DEPTH to be a single dataset, but read all depths for each variable
            if 'P' in decode_bytearray(s.netcdf_file_obj.variables['D_P_Code'][ivar]):
                LOGGER.error('Pressure data found in %s. This is not a valid XBT file' % s.XBT_input_filename)
                exit(1)
            dep = np.round(s.netcdf_file_obj.variables['Depthpress'][ivar, :], 4)
            # eliminate nan depths if there are any
            dep = np.ma.masked_invalid(dep)
            # resize the arrays to eliminate empty values
            dep = np.ma.masked_array(dep.compressed())

            # if the size of the depth array is not the same as ndeps, change ndeps
            if len(dep) != ndeps:
                ndeps = len(dep)
            depth_press_flag = s.netcdf_file_obj.variables['DepresQ'][ivar, :, 0].flatten()
            # resize the arrays to eliminate empty values
            depth_press_flag = np.ma.masked_array(depth_press_flag.compressed())
            qc = np.ma.masked_array(
                invalid_to_ma_array(depth_press_flag, fillvalue=0))
            # if the size of the array isn't equal to the number of depths, adjust here
            if len(qc) != ndeps:
                if len(qc) < ndeps:
                    # Create a new array of the desired size filled with NaN
                    resized_qc = np.full(ndeps, np.nan)
                    resized_qc[:len(qc)] = qc
                    qc = resized_qc
                else:
                    # qc is bigger than the number of depths, so resize the qc
                    qc = qc[:ndeps]

            prof = np.round(s.netcdf_file_obj.variables['Profparm'][ivar, 0, :, 0, 0], 4)
            # resize the arrays to eliminate empty values
            prof = np.ma.masked_array(prof.compressed())
            # Is there a mismatch in DEPTH and TEMP lengths?
            if ndeps < len(prof):
                # check the extra length contains valid data
                prof_rem = prof[ndeps:]
                if np.isnan(prof_rem).all() or np.all(prof_rem > 99) or prof_rem.mask.all():
                    # keep the valid data
                    prof = prof[:ndeps]
                    print('Check this bit of code!! %s' % s.XBT_input_filename)
                    exit(1)
                else:
                    LOGGER.error('Profile %s has %s depths but %s values for %s' % (s.XBT_input_filename, ndeps, len(prof), var))
                    exit(1)
            # if the size of the variable isn't equal to the number of depths, exit
            if (len(prof) != ndeps):
                LOGGER.error('Profile %s has %s depths but %s values for %s' % (s.XBT_input_filename, ndeps, len(prof), var))
                exit(1)

            # make any values >99 equal to 99.99. Some profiles have different values for invalid data
            if 'TEMP' in var:
                prof[prof > 99] = 99.99
            # TODO: consider other variables that might have different invalid values
            prof = np.ma.masked_invalid(prof)  # mask nan and inf values
            prof.set_fill_value(999999)

            prof_flag = s.netcdf_file_obj.variables['ProfQP'][ivar, 0, :, 0, 0].flatten()
            # resize the arrays to eliminate empty values
            prof_flag = np.ma.masked_array(prof_flag.compressed())
            prof_flag = np.ma.masked_array(
                invalid_to_ma_array(prof_flag, fillvalue=99))  # replace masked values for IMOS IODE flags

            if len(prof_flag) != ndeps:
                if len(prof_flag) < ndeps:
                    LOGGER.warning(
                        'Resizing %s and %s arrays to the number of depths recorded in MQNC file. %s' % (var, var, s.XBT_input_filename))
                    # Create a new array of the desired size filled with NaN
                    resized_prof = np.full(ndeps, np.nan)
                    resized_prof[:len(prof)] = prof
                    prof = resized_prof
                    resized_prof_flag = np.full(ndeps, np.nan)
                    resized_prof_flag[:len(prof_flag)] = prof_flag
                    prof_flag = resized_prof_flag
                else:
                    # prof_flag is bigger than the number of depths, so resize the qc
                    prof_flag = prof_flag[:ndeps]

            profile_qc.data['DEPTH' + raw] = dep.astype('float32')
            profile_qc.data['DEPTH' + raw +'_quality_control'] = pd.to_numeric(qc, errors='coerce').astype('int8')
            profile_qc.data[var + raw] = prof.astype('float32')
            profile_qc.data[var + raw + '_quality_control'] = pd.to_numeric(prof_flag, errors='coerce').astype('int8')

    # if DEPTH and DEPTH_RAW are not the same, apply fixes
    if not np.array_equal(profile_qc.data['DEPTH'], profile_qc.data['DEPTH_RAW']):
        # check the depth columns for consistency and match the variables based on DEPTH and DEPTH_RAW matches
        df_raw = profile_qc.data.filter(regex='RAW$', axis=1)
        df_qc = profile_qc.data.filter(regex='^((?!RAW).)*$', axis=1)
        # remove any nan rows from the dataframes
        df_raw = df_raw.dropna(how='all')
        df_qc = df_qc.dropna(how='all')

        # check the lengths of the arrays
        if len(df_raw) != len(df_qc):
            # there might be a couple of reasons for this.
            # 1. There is an extra depth added at 3.7m in the df_qc file and we need to put a nan row in the df_raw file
            # check if there is a 3.7m depth in the df_qc and not in the df_raw could also be depth corrected (3.7 *1.0336)
            tf = ((np.isclose(3.7, df_qc['DEPTH'].values, atol=1e-6).any()
                  and ~np.isclose(3.7, df_raw['DEPTH_RAW'].values, atol=1e-6).any())
                  or (np.isclose(3.7 * 1.0336, df_qc['DEPTH'].values, atol=1e-6).any()
                      and ~np.isclose(3.7 * 1.0336, df_raw['DEPTH_RAW'].values, atol=1e-6).any()))
            if tf:
                # what index is the 3.7m depth at in the df_qc
                idx = df_qc[np.isclose(3.7, df_qc['DEPTH'].values, atol=1e-6) |
                            np.isclose(3.7 * 1.0336, df_qc['DEPTH'].values, atol=1e-6)].index[0]
                # create a row of nans at the location where idx is
                nan_row = pd.DataFrame(np.nan, index=[idx], columns=df_raw.columns)
                # insert the nan row at the correct position
                df_raw = pd.concat([df_raw.iloc[:idx], nan_row, df_raw.iloc[idx:]]).reset_index(drop=True)
                # concatenate the two dataframes
                df = pd.concat([df_raw, df_qc], axis=1)
            # recheck the lengths
            if len(df_raw) != len(df_qc):
                # are there any duplicated depths in the longer dataframe?
                if df_raw['DEPTH_RAW'].duplicated().any():
                    LOGGER.warning('Duplicated DEPTH_RAW found in %s' % profile_qc.XBT_input_filename)
                    # drop the duplicates
                    df_raw = df_raw.drop_duplicates(subset='DEPTH_RAW').reset_index(drop=True)
                if df_qc['DEPTH'].duplicated().any():
                    LOGGER.warning('Duplicated DEPTH found in %s' % profile_qc.XBT_input_filename)
                    # drop the duplicates
                    df_qc = df_qc.drop_duplicates(subset='DEPTH').reset_index(drop=True)
                df = pd.concat([df_qc, df_raw], axis=1)
                # check the lengths again
                if len(df_raw) != len(df_qc):
                    LOGGER.warning('DEPTH_RAW and DEPTH counts are significantly different. Please review %s' % profile_qc.XBT_input_filename)
                    # concatenate the two dataframes with NaNs in the rows that don't match
                    df = pd.concat([df_qc, df_raw], axis=1)
        else:
            # simplest case where the lengths are the same but actual values might be different
            # concatenate the two dataframes
            df = pd.concat([df_qc, df_raw], axis=1)

        # check that the merge has worked
        if len(df) != max(len(df_raw), len(df_qc)):
            LOGGER.error('Dataframes have not been merged correctly. Please review %s' % profile_qc.XBT_input_filename)
            exit(1)

        # check here that the DEPTH and DEPTH_RAW columns are the same or DEPTH_RAW is 1.0336 * DEPTH
        if not np.isclose(df['DEPTH_RAW'].values, df['DEPTH'].values, atol=1e-6).all() and \
                not np.isclose(df['DEPTH_RAW'].values * 1.0336, df['DEPTH'].values, atol=1e-6).all():
            LOGGER.error('DEPTH_RAW and DEPTH values do not match in %s' % profile_qc.XBT_input_filename)
            exit(1)

        # save the dataframe of DEPTH dimensioned data to the profile object
        profile_qc.data = df

    # check for duplicated depths and log if found
    if profile_qc.data['DEPTH'].duplicated().any() or profile_qc.data['DEPTH_RAW'].duplicated().any():
        LOGGER.error('Duplicated DEPTH or DEPTH_RAW found in %s' % profile_qc.XBT_input_filename)

    # Location information
    lat = profile_qc.netcdf_file_obj['latitude'][0].__float__()
    lat_raw = profile_noqc.netcdf_file_obj['latitude'][0].__float__()
    lon = profile_qc.netcdf_file_obj['longitude'][0].__float__()
    lon_raw = profile_noqc.netcdf_file_obj['longitude'][0].__float__()
    # check if scale factor has been applied, shouldn't have a negative longitude:
    if lon < 0:
        if profile_qc.netcdf_file_obj['longitude'].scale:
            LOGGER.info('Scale Factor in ed file longitude attributes, changing longitude value from  %s %s' %
                        (lon, profile_qc.XBT_input_filename))
            lon = lon * -1
        else:
            LOGGER.error('Negative LONGITUDE value with no Scale Factor %s %s' % (lon, profile_qc.XBT_input_filename))

    # Change the 360 degree longitude to degrees_east (0-180, -180 to 0)
    if lon > 180:
        lon = lon - 360

    # Change the 360 degree longitude to degrees_east (0-180, -180 to 0)
    if lon_raw > 180:
        lon_raw = lon_raw - 360

    profile_qc.data['LATITUDE'] = np.round(lat, 6)
    profile_qc.data['LONGITUDE'] = np.round(lon, 6)
    profile_qc.data['LATITUDE_RAW'] = np.round(lat_raw, 6)
    profile_qc.data['LONGITUDE_RAW'] = np.round(lon_raw, 6)

    # position and time QC - check this is not empty. Assume 1 if it is
    q_pos = profile_qc.netcdf_file_obj['Q_Pos'][0]
    if not q_pos or q_pos.ndim == 0:
        # only one value in the array
        q_pos = remove_control_chars(str(decode_bytearray(profile_qc.netcdf_file_obj['Q_Pos'][:])))
        if q_pos:
            q_pos = int(q_pos)
        else:
            q_pos = 1
    else:
        # Apply the function to each element in the masked array
        q_pos = int(np.ma.array([remove_control_chars(str(item)) for item in q_pos.data], mask=q_pos.mask)[0])

    profile_qc.data['LATITUDE_quality_control'] = q_pos
    profile_qc.data['LONGITUDE_quality_control'] = q_pos

    # Date time information
    woce_date = profile_qc.netcdf_file_obj['woce_date'][0]
    woce_time = profile_qc.netcdf_file_obj['woce_time'][0]

    # AW Add Original date_time from the raw .nc - date-time could be changed thru QC
    woce_date_raw = profile_noqc.netcdf_file_obj['woce_date'][0]
    woce_time_raw = profile_noqc.netcdf_file_obj['woce_time'][0]

    q_date_time = profile_qc.netcdf_file_obj['Q_Date_Time'][0]
    # remove control characters from the q_date_time
    if not q_date_time or q_date_time.ndim == 0:
        # only one value in the array
        q_date_time = int(decode_bytearray(profile_qc.netcdf_file_obj['Q_Date_Time'][:]))
    else:
        q_date_time = int(
            np.ma.array([remove_control_chars(str(item)) for item in q_date_time.data], mask=q_date_time.mask)[0])

    # need to be a bit more specific as some times have missing padding at the end, some at the start.
    # could break if hour is 00 and there are no zeros!
    # Let's try padding left and right, then convert to time for both
    rpad = str(woce_time).ljust(6, '0')
    lpad = str(woce_time).zfill(6)

    # get the right date format
    xbt_date = '%sT%s' % (woce_date, rpad)
    xbt_date = convert_time_string(xbt_date, '%Y%m%dT%H%M%S')
    xbt_date2 = '%sT%s' % (woce_date, lpad)
    xbt_date2 = convert_time_string(xbt_date2, '%Y%m%dT%H%M%S')
    # replace NaT with the other date
    if pd.isnull(xbt_date):
        xbt_date = xbt_date2

    # Raw date
    rpad = str(woce_time_raw).ljust(6, '0')
    lpad = str(woce_time_raw).zfill(6)

    # get the right date format
    xbt_date_raw = '%sT%s' % (woce_date_raw, rpad)
    xbt_date_raw = convert_time_string(xbt_date_raw, '%Y%m%dT%H%M%S')
    xbt_date_raw2 = '%sT%s' % (woce_date_raw, lpad)
    xbt_date_raw2 = convert_time_string(xbt_date_raw2, '%Y%m%dT%H%M%S')
    # replace NaT with the other date
    if pd.isnull(xbt_date_raw):
        xbt_date_raw = xbt_date_raw2

    # AW - TIME_RAW is original date-time - set it too
    profile_qc.data['TIME'] = xbt_date
    profile_qc.data['TIME_quality_control'] = q_date_time
    profile_qc.data['TIME_RAW'] = xbt_date_raw

    # drop rows where all NaN values which does happen in these old files sometimes
    profile_qc.data = profile_qc.data.dropna(subset=['TEMP', 'DEPTH', 'TEMP_RAW', 'DEPTH_RAW'], how='all')

    # how many parameters do we have, not including DEPTH?
    profile_qc.nprof = len(profile_qc.prof_type)
    profile_noqc.nprof = len(profile_noqc.prof_type)

    # TODO: handle all the other variables here
    profile_qc, profile_noqc = parse_extra_vars(profile_qc, profile_noqc)

    return profile_qc, profile_noqc


def adjust_position_qc_flags(profile):
    """ When a 'PE' flag is present in the Act_Code, the latitude and longitude qc flags need to be adjusted if not
    already set (applies to data processed with older versions of MQUEST)
    Also, if the temperature QC flags are not set correctly (3 for PER, 2 for PEA), these should be updated.
    """

    # exit this if we don't have a position code
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("LA|LO|PE")]) == 0:
        return profile

    # get the temperature QC codes
    df = profile.data
    if profile.histories['HISTORY_QC_CODE'].str.contains('LAA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if not np.isclose(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LAA'), 'HISTORY_PREVIOUS_VALUE'].values),
                      profile.data['LATITUDE_RAW'], atol=1e-6).all():
            LOGGER.error('LATITUDE_RAW not the same as the PREVIOUS_value! %s' % profile.XBT_input_filename)
        if profile.data['LATITUDE_quality_control'] != 5:
            # PEA on latitude
            profile.data['LATITUDE_quality_control'] = 5
            LOGGER.info('LATITUDE correction (PEA) in original file, changing LATITUDE flag to level 5. %s'
                        % profile.XBT_input_filename)
            # change to flag 2 for temperature for all depths where qc is less than 2
            mask = df['TEMP_quality_control'] < 2
            df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('LOA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LONGITUDE_RAW value within a tolerance
        if not np.isclose(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LOA'), 'HISTORY_PREVIOUS_VALUE'].values),
                      profile.data['LONGITUDE_RAW'], atol=1e-6).all():
            LOGGER.error('LONGITUDE_RAW not the same as the PREVIOUS_value! %s' % profile.XBT_input_filename)
        if profile.data['LONGITUDE_quality_control'] != 5:
            # PEA on longitude
            profile.data['LONGITUDE_quality_control'] = 5
            LOGGER.info('LONGITUDE correction (PEA) in original file, changing LONGITUDE flag to level 5. %s'
                        % profile.XBT_input_filename)
            # change to flag 2 for temperature for all depths where qc is less than 2
            mask = df['TEMP_quality_control'] < 2
            df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('PER').any():
        # PER on longitude and latitude
        profile.data['LONGITUDE_quality_control'] = 3
        profile.data['LATITUDE_quality_control'] = 3
        LOGGER.info('Position Reject (PER) in original file, changing LONGITUDE & LATITUDE flags to level 3.%s'
                    % profile.XBT_input_filename)
        # change to flag 3 for temperature for all depths where qc is less than 3
        mask = df['TEMP_quality_control'] < 3
        df.loc[mask, 'TEMP_quality_control'] = 3

    # update the temperature QC flags
    profile.data = df

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
        LOGGER.info('TIME correction (TEA) in original file, changing TIME flag to level 5.%s'
                    % profile.XBT_input_filename)
        # change to flag 2 for temperature for all depths where qc is less than 2

        profile.data.loc[profile.data['TEMP_quality_control'] < 2, 'TEMP_quality_control'] = 2
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if convert_time_string(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'TEA'), 'HISTORY_PREVIOUS_VALUE'].values, format='%Y%m%d%H%M%S') != \
                profile.data['TIME_RAW']:
            LOGGER.error('TIME_RAW not the same as the PREVIOUS_VALUE! %s'
                         % profile.XBT_input_filename)

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
    profile.data['TEMP_uncertainty'] = ma.empty_like(profile.data['TEMP'])
    profile.data['TEMP_uncertainty'] = tunc[0]
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
    xx = ['XBT_fallrate_equation_coefficients', 'XBT_fallrate_equation_coefficients_RAW']
    ind = 0

    for s in nms:
        if att_name in list(s.global_atts.keys()):
            item_val = s.global_atts[att_name]
            item_val = ''.join(item_val.split())
            if item_val in list(ptyp_list.keys()) and item_val not in list(fre_list.keys()):
                # old PTYP surface code, need to match up PEQ$code
                item_val = ptyp_list[item_val]
            # is it in the PEQ list
            elif item_val in list(peq_list.keys()) and item_val not in list(fre_list.keys()):
                LOGGER.warning('PROBE_TYPE %s is not an XBT type, not converted' % item_val)
                profile_qc.data['PROBE_TYPE'] = ''
                # this is not an XBT
                return profile_qc
            elif item_val not in list(fre_list.keys()):
                # record the orignal value
                profile_qc.global_atts[vv[ind] + '_origname'] = item_val
                # try fuzzy matching here
                imatch = difflib.get_close_matches(item_val[0:4], list(ptyp_list.keys()), n=1, cutoff=0.5)
                if imatch:
                    LOGGER.warning('PROBE_TYPE %s not found in WMO1770, using closest match %s %s'
                                   % (item_val, imatch[0], s.XBT_input_filename))
                    item_val = ptyp_list[imatch[0]]

            if item_val in list(fre_list.keys()):
                probetype = peq_list[item_val]
                coef_a = fre_list[item_val].split(',')[0]
                coef_b = fre_list[item_val].split(',')[1]

                profile_qc.data[vv[ind]] = item_val
                profile_qc.global_atts[vv[ind] + '_name'] = probetype
                profile_qc.global_atts[xx[ind]] = 'a: ' + coef_a + ', b: ' + coef_b
                profile_qc.data['PROBE_TYPE_quality_control'] = 1
            else:
                # Handle case where no good match is found
                profile_qc.global_atts[xx[ind]] = 'Unknown'
                profile_qc.data[vv[ind]] = item_val
                profile_qc.global_atts[vv[ind] + '_name'] = 'Unknown'
                profile_qc.data['PROBE_TYPE_quality_control'] = 0
                LOGGER.warning('PROBE_TYPE %s is unknown in %s' % (item_val, s.XBT_input_filename))
        else:
            profile_qc.global_atts[xx[ind]] = 'Unknown'
            profile_qc.data[vv[ind]] = '1023'
            profile_qc.global_atts[vv[ind] + '_name'] = 'Unknown'
            profile_qc.data['PROBE_TYPE_quality_control'] = 0
            LOGGER.error('PROBE_TYPE, XBT_probetype_fallrate_equation missing from %s' % s.XBT_input_filename)
        ind = ind + 1

    # select a QC flag for the probe type
    # TODO: if the probe types are different in raw and edited, need to handle this.
    #  Has it been changed? what does the data look like? Need to assign 5 to changed profile, include the PR flag
    #  and adjust the QC on the temperature and depth
    if profile_qc.data['PROBE_TYPE'] != profile_qc.data['PROBE_TYPE_RAW']:
        LOGGER.error('PROBE_TYPE are different in ed and raw files. %s' % profile_qc.XBT_input_filename)

    return profile_qc


def parse_histories_nc(profile):
    """ Parse the history records in Mquest files
    """
    # let's use a pandas dataframe with empty columns
    df = pd.DataFrame()
    nhist = int(profile.netcdf_file_obj['Num_Hists'][0].data)

    # for each column, extract the data
    # list the data labels matching the columns in the dataframe
    varname = ['Act_Code', 'Ident_Code', 'Act_Parm', 'PRC_Code', 'PRC_Date', 'Aux_ID', 'Flag_severity', 'Version',
                'Previous_Val']
    for var in varname:
        if var not in profile.netcdf_file_obj.variables:
            LOGGER.warning('Variable %s not found in %s' % (var, profile.XBT_input_filename))
            df[var] = np.nan
            continue
        # test if the data is a byte array or a float
        if np.issubdtype(profile.netcdf_file_obj[var].dtype, np.number):
            vv = profile.netcdf_file_obj[var][0:nhist].data
        else:
            # if this is the Act_Code, check if the nhist is correct
            if var == 'Act_Code':
                vv = [''.join(chr(x) for x in bytearray(xx)).strip()
                      for xx in profile.netcdf_file_obj[var][:].data if bytearray(xx).strip()]
                vv = [remove_control_chars(str(x)) for x in vv]
                # Remove empty strings from the list
                vv = [x for x in vv if x]
                if nhist != len(vv):
                    nhist = len(vv)
                    LOGGER.warning('HISTORY: Updating nhist to match length of history codes. %s' % profile.XBT_input_filename)
            # convert the byte array to a string
            vv = [''.join(chr(x) for x in bytearray(xx)).strip()
                  for xx in profile.netcdf_file_obj[var][0:nhist].data if bytearray(xx).strip()]
            vv = [remove_control_chars(str(x)) for x in vv]
        df[var] = vv
    # rename the columns
    df.columns = ['HISTORY_QC_CODE', 'HISTORY_INSTITUTION', 'HISTORY_PARAMETER', 'HISTORY_SOFTWARE',
                               'HISTORY_DATE', 'HISTORY_START_DEPTH', 'HISTORY_QC_CODE_VALUE',
                               'HISTORY_SOFTWARE_RELEASE', 'HISTORY_PREVIOUS_VALUE']

    # change HISTORY_START_DEPTH and HISTORY_PREVIOUS_VALUE to float64
    df['HISTORY_START_DEPTH'] = df['HISTORY_START_DEPTH'].astype('float32')
    df['HISTORY_PREVIOUS_VALUE'] = df['HISTORY_PREVIOUS_VALUE'].astype('float32')
    # change HISTORY_QC_CODE_VALUE to int32
    df['HISTORY_QC_CODE_VALUE'] = df['HISTORY_QC_CODE_VALUE'].astype('int32')

    if nhist > 0:
        # check that the history codes exist in our list
        # read the set list of codes from the csv files
        qc_df = read_qc_config()
        # make a new column with the first two characters of the qc_df code
        qc_df['code_short'] = qc_df['code'].str[:2]
        # create list of acceptable parameter names
        parm_names = {'DEPH': 'DEPTH', 'DATI': 'DATE, TIME', 'DATE': 'DATE', 'TIME': 'TIME', 'LATI': 'LATITUDE',
                 'LONG': 'LONGITUDE', 'LALO': 'LATITUDE, LONGITUDE', 'TEMP': 'TEMP'}
        # check that the history codes are in the list
        if not df['HISTORY_QC_CODE'].isin(qc_df['code_short']).all():
            missing = df.loc[~df['HISTORY_QC_CODE'].isin(qc_df['code_short']), 'HISTORY_QC_CODE']
            LOGGER.warning('HISTORY_QC_CODE values %s not found in the QC code list. Please review output for this file %s'
                           % (missing.values, profile.XBT_input_filename))
            # remove any codes that are not in the list and where PARAMETER is not in names list
            df = df.loc[df['HISTORY_QC_CODE'].isin(qc_df['code_short']) & df['HISTORY_PARAMETER'].isin(parm_names.keys())]
            # reset nhist to the new length
            nhist = len(df)

        # allow for history dates to be YYYYMMDD or DDMMYYYY
        date1 = convert_time_string(df['HISTORY_DATE'], '%Y%m%d')
        date2 = convert_time_string(df['HISTORY_DATE'],'%d%m%Y')
        df['HISTORY_DATE'] = date1.fillna(date2)
    else:
        # no history records
        profile.histories = df
        return profile

    # append the 'A' or 'R' to each code
    for idx, row in df.iterrows():
        if df.at[idx, 'HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5]:
            df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'A'
        else:
            df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'R'

    # update institute names to be more descriptive
    names = read_section_from_xbt_config('INSTITUTE')
    df['HISTORY_INSTITUTION'] = df['HISTORY_INSTITUTION'].map(lambda x: names[x].split(',')[0] if x in names else x)
    if any(df['HISTORY_INSTITUTION'].isna()):
        # list the institutes that are not defined
        missing = df.loc[df['HISTORY_INSTITUTION'].isna(), 'HISTORY_INSTITUTION']
        LOGGER.warning("HISTORY_INSTITUTION values %s are not defined. Please review output for this file %s"
                     % (missing, profile.XBT_input_filename))

    # get a list of qc_df['code'] values where qc_df['code_short'] only appears once in the dataframe
    # Get the value counts of 'code_short'
    code_short_counts = qc_df['code_short'].value_counts()
    # Filter 'qc_df' to get rows where 'code_short' appears only once
    single_code_short_df = qc_df[qc_df['code_short'].isin(code_short_counts[code_short_counts == 1].index)]

    # if any of the single_qc_codes are in the HISTORY_QC_CODE, change the HISTORY_QC_CODE_VALUE to match the single_qc_code_short_df['tempqc'] value
    for idx, row in single_code_short_df.iterrows():
        mask = df['HISTORY_QC_CODE'].str[:2] == row['code_short']
        if any(mask):
            df.loc[mask, ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE', 'HISTORY_PARAMETER']] = [row['code'] ,row['tempqc'], row['parameter']]
    # this group of changes is here because I have reviewed all our QC codes in the historic databases and I know
    # there are some that are not correct.
    # change ERA to PLA with flag 3 to reduce duplication of flags
    df.loc[
        (df['HISTORY_QC_CODE'].str.contains('ERA')), ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'PLA', 3
    # change URA for BDA and flag 2
    df.loc[
        (df['HISTORY_QC_CODE'].str.contains('URA')), ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'BDA', 2

    # set the software value to 2.1 for CS and PE, RE flags
    df.loc[
        df.HISTORY_QC_CODE.isin(['CSR', 'PEA', 'PER', 'REA']), ['HISTORY_SOFTWARE_RELEASE', 'HISTORY_SOFTWARE']] = '2.1', 'CSCBv2'

    # update software names to be more descriptive
    names = {'CSCB': 'CSIRO Quality control cookbook for XBT data v1.1',
             'CSCBv2': 'Australian XBT Quality Control Cookbook Version 2.1'}
    df['HISTORY_SOFTWARE'] = df['HISTORY_SOFTWARE'].map(names, na_action='ignore')

    # fix any 9999 etc values in HISTORY_PREVIOUS_VALUE where HISTORY_PARAMETER is TEMP to be 99.99 to match the data
    df.loc[(df['HISTORY_PARAMETER'] == 'TEMP') & (df['HISTORY_PREVIOUS_VALUE'] > 99), 'HISTORY_PREVIOUS_VALUE'] = 99.99

    # change CSA to CSR and the flag to 3 to match new format
    df.loc[(df['HISTORY_QC_CODE'].str.contains('CSA')),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'CSR', 3

    # Change the PEA flag to LA or LO and ensure the TEMP_QC_CODE_VALUE is set to 2, not 5
    df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
            (df['HISTORY_PARAMETER'].str.contains('LATITUDE'))),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'LAA', 2
    df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
            (df['HISTORY_PARAMETER'].str.contains('LONGITUDE'))),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'LOA', 2

    # Combine duplicated TEA flags to a single TEA for TIME variable TEMP_QC_CODE_VALUE is set to 2, not 5
    # Also change just DATE TEA flags to TIME
    dfTEA = df[df['HISTORY_QC_CODE'] == 'TEA'].copy()
    if len(dfTEA) > 0:
        # test here for both TIME and DATE in the TEA flags
        if any(dfTEA['HISTORY_PARAMETER'].str.contains('TIME')) & any(dfTEA['HISTORY_PARAMETER'].str.contains('DATE')):
            LOGGER.error('TEA flags contain both TIME and DATE. Please review %s' % profile.XBT_input_filename)
            exit(1)

        # get the date value from the TIME variable
        dtt = profile.data['TIME'].strftime('%Y%m%d')
        # get the TIME value from the TIME variable
        ti = profile.data['TIME'].strftime('%H%M%S')

        # if any of timerows['HISTORY_PREVIOUS_VALUE'] contains a variation with 9's then set to 0
        timeidx = df['HISTORY_PARAMETER'] == 'TIME'
        pattern = re.compile(r'^9{1,5}(\.\d+)?$')
        if timeidx.any():
            if df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'].astype(str).str.contains(pattern).any():
                df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'] = 0
            # include the date information
            df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'] = df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'].apply(
                lambda x: dtt + str(int(x)) + '00').astype(float)

        # now check for any 'DATE' parameter in the TEA flags
        dateidx = df['HISTORY_PARAMETER'] == 'DATE'
        if dateidx.any():
            if df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'].astype(str).str.contains(pattern).any():
                df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'] = 0
            # allow for dates to be YYYYMMDD or DDMMYYYY
            date1 = convert_time_string(df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'].astype(int).astype(str), '%Y%m%d', 'string').astype(float)
            date2 = convert_time_string(df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'].astype(int).astype(str), '%d%m%Y', 'string').astype(float)
            df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'] = date1.fillna(date2)

        # change the 'DATE' label to TIME  and update the TEA PREVIOUS_VALUE to the new datetime value
        df.loc[((df['HISTORY_PARAMETER'].str.contains('DATE') | df['HISTORY_PARAMETER'].str.contains('TIME')) &
                (df['HISTORY_QC_CODE'].str.contains('TEA'))), ['HISTORY_PARAMETER']] = 'TIME'

    # add the QC description information
    df["HISTORY_QC_CODE_DESCRIPTION"] = [''] * nhist
    # map the qc_df['code'] to the df['HISTORY_QC_CODE'] and add the description to the df['HISTORY_QC_CODE_DESCRIPTION']

    # Create a dictionary from qc_df for mapping
    qc_code_to_description = qc_df.set_index('code')['label'].to_dict()

    # Map the 'HISTORY_QC_CODE' to the descriptions and add to 'HISTORY_QC_CODE_DESCRIPTION'
    df['HISTORY_QC_CODE_DESCRIPTION'] = df['HISTORY_QC_CODE'].map(qc_code_to_description)

    if any(df['HISTORY_QC_CODE_DESCRIPTION'].eq('')):
        missing = df.loc[df['HISTORY_QC_CODE_DESCRIPTION'] == '', 'HISTORY_QC_CODE']
        if missing.any():
            LOGGER.warning("HISTORY_QC_CODE \"%s\" is not defined. Please edit xbt_config file. %s"
                           % (missing, profile.XBT_input_filename))

    # remove any duplicated lines for any code
    df = df[~(df.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_START_DEPTH']))]
    # find duplicated codes where one previous value is nan and parameter is TEMP. Remove the nan value
    mask = df.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_START_DEPTH'])
    mask2 = df['HISTORY_PARAMETER'].str.contains('TEMP')
    mask3 = df['HISTORY_PREVIOUS_VALUE'].isna()
    mask4 = mask & mask2 & mask3
    if any(mask4):
        print('Check this is working %s' % profile.XBT_input_filename)
        exit(1)
        df = df[~mask4]

    # sort the flags by depth order to help with finding STOP_DEPTH
    # TODO: will keep the stop depth for now. Consider re-writing to loop over each of the lists of act_code types
    df = df.sort_values('HISTORY_START_DEPTH')
    dfdat = profile.data
    for idx, row in df.iterrows():
        # Ensure start depth is the same as the value in the depth array
        # Find the closest value to the start depth in the histories
        ii = (dfdat['DEPTH'] - row['HISTORY_START_DEPTH']).abs().idxmin()
        df.at[idx, 'HISTORY_START_DEPTH'] = dfdat.at[ii, 'DEPTH']
        # QC,RE, TE, PE and EF etc flag applies to entire profile, stop_depth is deepest depth
        res = row['HISTORY_QC_CODE'] in qc_df.loc[
            qc_df['group_label'].str.contains('ACT_CODES_FULL_PROFILE'), 'code'].values
        if res:
            df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

        # if the flag is in act_code_single_point list, then stop depth is same as start
        res = row['HISTORY_QC_CODE'] in qc_df.loc[
            qc_df['group_label'].str.contains('ACT_CODES_SINGLE_POINT'), 'code'].values
        if res:
            df.at[idx, "HISTORY_STOP_DEPTH"] = df.at[idx, 'HISTORY_START_DEPTH']

        # TODO: surface flags in the act_code_next_flag category need to ignore the CS flags
        # if the flag is in act_code_next_flag, then stop depth is the next depth or bottom
        # find next deepest flag depth
        res = row['HISTORY_QC_CODE'] in qc_df.loc[
            qc_df['group_label'].str.contains('ACT_CODES_TO_NEXT_FLAG'), 'code'].values
        stop_idx = df['HISTORY_START_DEPTH'] > row['HISTORY_START_DEPTH']
        stop_depth = df['HISTORY_START_DEPTH'][stop_idx]
        if any(stop_idx) & res:
            ii = (np.abs(dfdat['DEPTH'] - stop_depth.values[0])).argmin()
            df.at[idx, "HISTORY_STOP_DEPTH"] = dfdat['DEPTH'][ii]
        elif res:  # if there isn't a deeper flag, use deepest depth
            df.at[idx, "HISTORY_STOP_DEPTH"] = profile.global_atts['geospatial_vertical_max']

    # assign the dataframe back to profile at this stage
    profile.histories = df.reset_index(drop=True)

    return profile


def combine_histories(profile_qc, profile_noqc):
    # check for global attributes in the noqc file and update the global atts as required
    # handle the longitude change where data was imported from dataset with a negative longitude where it should
    # have been positive. The *raw.nc previous value and *ed.nc previous value should be the same, update the LONG_RAW.
    #first merge all the histories
    combined_histories = pd.merge(profile_qc.histories, profile_noqc.histories, how='left')
    # check for TER where the date has been corrected and therefore should be a TEA, happens in some badly recorded flags in old files
    if any(combined_histories['HISTORY_QC_CODE'].str.contains('TER')):
        # does the HISTORY_PREVIOUS_VALUE match the TIME_RAW value?
        if not combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
            'HISTORY_PREVIOUS_VALUE'].values == profile_qc.data['TIME_RAW'].timestamp():
            # does the previous value contain 9's?
            if any(combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
                'HISTORY_PREVIOUS_VALUE'].astype(str).str.contains('9{1,5}')):
                LOGGER.warning('HISTORY: Previous value does not match TIME_RAW value. %s' % profile_qc.XBT_input_filename)
                # use the TIME_RAW value and update previous value
                combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
                    'HISTORY_PREVIOUS_VALUE'] = int(profile_qc.data['TIME_RAW'].strftime('%Y%m%d%H%M%S'))
                # now are the TIME and TIME_RAW values the same?
                if profile_qc.data['TIME'] != profile_qc.data['TIME_RAW']:
                    # update TER to TEA and change the flag to 2
                    combined_histories.loc[
                        combined_histories['HISTORY_QC_CODE'].str.contains('TER'), ['HISTORY_QC_CODE',
                                    'HISTORY_QC_CODE_VALUE']] = ['TEA', 2]
    # find rows in combined_histories where the HISTORY_QC_CODE contains PER and HISTORY_PARAMETER is not 'LATITUDE, LONGITUDE':
    if combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('PER') \
                    & ~combined_histories['HISTORY_PARAMETER'].str.contains('LATITUDE, LONGITUDE')].shape[0] > 0:
        # in this case, the PER has to be changed to LAA if 'HISTORY_PARAMETER' is 'LATITUDE' and LOA if 'HISTORY_PARAMETER' is 'LONGITUDE'
        # find the rows where the HISTORY_QC_CODE contains PER and HISTORY_PARAMETER is 'LATITUDE'
        if combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('PER') \
                    & combined_histories['HISTORY_PARAMETER'].str.contains('LATITUDE')].shape[0] > 0:
            # update PER to LAA and change the flag to 2
            combined_histories.loc[
                combined_histories['HISTORY_QC_CODE'].str.contains('PER') & combined_histories['HISTORY_PARAMETER'].str.contains('LATITUDE'),
                ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = ['LAA', 2]
        # find the rows where the HISTORY_QC_CODE contains PER and HISTORY_PARAMETER is 'LONGITUDE'
        if combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('PER') \
                    & combined_histories['HISTORY_PARAMETER'].str.contains('LONGITUDE')].shape[0] > 0:
            # update PER to LOA and change the flag to 2
            combined_histories.loc[
                combined_histories['HISTORY_QC_CODE'].str.contains('PER') & combined_histories['HISTORY_PARAMETER'].str.contains('LONGITUDE'),
                ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = ['LOA', 2]


    # check for duplicated history codes at the same depth so we don't duplicate the QC code in the fft variable
    # this will keep the first value recorded in HISTORY_DATE.
    non_temp_codes = combined_histories[combined_histories['HISTORY_PARAMETER'] != 'TEMP']
    # loop over the unique values in the HISTORY_PARAMETER column
    for vv in non_temp_codes['HISTORY_PARAMETER'].unique():
        var = vv + '_RAW'
        # get the index of duplicated rows for vv in non_temp_codes
        dup_idx = non_temp_codes[non_temp_codes['HISTORY_PARAMETER'] == vv].duplicated(
            subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)
        if dup_idx.any():
            # TODO: if DEPTH is duplicated, check the previous value is the same as the DEPTH_RAW value, will need indexing
            dup_idx = dup_idx.reindex(non_temp_codes.index, fill_value=False)
            if vv not in ['LONGITUDE', 'TIME', 'LATITUDE']:
                if vv in ['DEPTH']:
                    print('HISTORY: Duplicate %s flags found, need to troubleshoot. %s' % (vv, profile_qc.XBT_input_filename))
                    exit(1)
                # will be 'LATITUDE, LONGITUDE' or 'DATE, TIME'
                # find the first flag looking at HISTORY_DATE
                idx = non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                    'HISTORY_DATE'].idxmin()
                if len(idx) > 0:
                    LOGGER.warning('PREVIOUS_VALUE is not the same as the %s value, removed from the dataset %s'
                                   % (var, profile_qc.XBT_input_filename))
                    non_temp_codes = non_temp_codes.drop(idx)
            # else it is TEA
            elif vv == 'TIME':
                # check the previous value is the same as the TIME_RAW value
                # convert the previous value to a datetime object
                prevval = convert_time_string(non_temp_codes[dup_idx]['HISTORY_PREVIOUS_VALUE'], format='%Y%m%d%H%M%S')
                # identify the rows where the previous value is not the same as the TIME_RAW value and remove them
                idx = non_temp_codes[dup_idx][~(prevval == profile_qc.data['TIME_RAW'])].index
                if len(idx) > 0:
                    LOGGER.warning('Duplicated PREVIOUS_VALUE is not the same as the TIME_RAW value, removed %s'
                                   % profile_qc.XBT_input_filename)
                    non_temp_codes = non_temp_codes.drop(idx)
            else:
                # handle any duplicated position flags here
                # keep the earliest LATITUDE or LONGITUDE flag and remove the others
                LOGGER.warning(
                    'HISTORY: Multiple %s flags found in histories and duplicates removed. %s' % (vv, profile_noqc.XBT_input_filename))
                # find the first flag looking at HISTORY_DATE
                idx = non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'] == vv, 'HISTORY_DATE'].idxmin()
                # remove the other LOA flags
                non_temp_codes = non_temp_codes.drop(
                    non_temp_codes.loc[
                        non_temp_codes['HISTORY_PARAMETER'].values == vv].index.difference(
                        [idx]))

        # copy this information to the PARAMETER_RAW value if it isn't the same, check only where the parameter exactly matches vv
        if vv in ['LATITUDE', 'LONGITUDE']:
            if np.round(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
            'HISTORY_PREVIOUS_VALUE'].values, 6) != np.round(
                profile_qc.data[var], 6):
                LOGGER.info('HISTORY: Updating %s_RAW to match the previous value in *raw.nc file. %s'
                               % (vv, profile_qc.XBT_input_filename))
                profile_qc.data[var] = non_temp_codes.loc[
                    non_temp_codes['HISTORY_PARAMETER'].values == vv, 'HISTORY_PREVIOUS_VALUE'].values[0]
        elif vv in ['TIME']:
            # TIME_RAW is in datetime format and HISTORY_PREVIOUS_VALUE is in float format
            # if the HISTORY_PREVIOUS_VALUE is not NaN, then it is a valid date
            if not pd.isna(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                'HISTORY_PREVIOUS_VALUE'].values[0]):
                # convert the HISTORY_PREVIOUS_VALUE to a datetime object
                prevval = convert_time_string(str(int(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                    'HISTORY_PREVIOUS_VALUE'].values[0])), '%Y%m%d%H%M%S', 'string')
                # check the previous value is the same as the TIME_RAW value
                if not prevval == profile_qc.data[var]:
                    LOGGER.info('HISTORY: Updating %s_RAW to match the previous value in *raw.nc file. %s'
                                   % (vv, profile_qc.XBT_input_filename))
                    # for time, keep TIME_RAW as the previous value
                    non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv, 'HISTORY_PREVIOUS_VALUE'] = int(profile_qc.data['TIME_RAW'].strftime('%Y%m%d%H%M%S'))

    # Filter the rows where HISTORY_PARAMETER is TEMP
    temp_codes = combined_histories[combined_histories['HISTORY_PARAMETER'] == 'TEMP']
    # get the index of the rows to drop for TEMP variables only
    idx = temp_codes[(temp_codes.duplicated(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)) &
                        (temp_codes['HISTORY_PREVIOUS_VALUE'] > 90)].index
    if len(idx) > 0:
        LOGGER.warning(
            'HISTORY: Duplicate QC code encountered and removed in create_flag_feature: %s. Please review. %s'
            % (temp_codes.loc[idx, 'HISTORY_QC_CODE'].unique(), profile_qc.XBT_input_filename))
        temp_codes = temp_codes.drop(idx)
    # Concatenate the non-TEMP rows back with the sorted TEMP rows
    combined_histories = pd.concat([non_temp_codes, temp_codes])

    profile_qc.histories = combined_histories
    # check for any duplicated flags that aren't exact matches but occur at the same depth with same previous value and remove them
    profile_qc.histories = profile_qc.histories[~(profile_qc.histories.duplicated(['HISTORY_PARAMETER',
                                                'HISTORY_QC_CODE', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_START_DEPTH']))]

    # are there any duplicates left that we need to investigate?
    if profile_qc.histories.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_START_DEPTH']).any():
        # if the HISTORY_PREVIOUS_VALUE, HISTORY_PARAMETER, HISTORY_QC_CODE AND HISTORY_START_DEPTH are the same, then remove the duplicate
        profile_qc.histories = profile_qc.histories.drop_duplicates(['HISTORY_PARAMETER', 'HISTORY_QC_CODE',
                                                                     'HISTORY_START_DEPTH', 'HISTORY_PREVIOUS_VALUE'])
        # Filter the dataframe for rows where HISTORY_PARAMETER is 'TEMP'
        temp_df = profile_qc.histories[profile_qc.histories['HISTORY_PARAMETER'] == 'TEMP']

        # Find duplicated rows based on HISTORY_QC_CODE and HISTORY_START_DEPTH
        duplicated_rows = temp_df[temp_df.duplicated(['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)]

        # Find rows where HISTORY_PREVIOUS_VALUE is different
        different_previous_value_rows = duplicated_rows[
            duplicated_rows.duplicated(['HISTORY_QC_CODE', 'HISTORY_START_DEPTH', 'HISTORY_PREVIOUS_VALUE'],
                                       keep=False) == False]

        # check these rows to see if the HISTORY_PREVIOUS_VALUE is the same as the TEMP_RAW value
        for idx, row in different_previous_value_rows.iterrows():
            # get the index of the row in the profile data
            ii = np.where(np.round(profile_qc.data['DEPTH'], 2) == np.round(row['HISTORY_START_DEPTH'], 2))[0]
            # check the previous value is the same as the TEMP_RAW value
            if not np.round(row['HISTORY_PREVIOUS_VALUE'], 2).item() == np.round(profile_qc.data['TEMP_RAW'][ii], 2).item():
                # remove this row from the dataframe
                profile_qc.histories = profile_qc.histories.drop(idx)
                # log the error
                LOGGER.warning('HISTORY: Duplicate QC code removed: %s. Please review. %s' % (row['HISTORY_QC_CODE'], profile_qc.XBT_input_filename))
        if profile_qc.histories.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_START_DEPTH']).any():
            LOGGER.warning('HISTORY: Duplicated flags remain in the qc file. Please review. %s' % profile_qc.XBT_input_filename)

    # reset the index
    profile_qc.histories = profile_qc.histories.reset_index(drop=True)
    return profile_qc


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
    df = profile.data
    # find the depths in the profile data
    ind = np.in1d(np.round(df['DEPTH'], 2), np.round(depths, 2)).nonzero()[0]
    # does this profile have a PLA flag? if so, use the previous values to replace the TEMP values
    if 'PLA' in profile.histories['HISTORY_QC_CODE'].values:
        LOGGER.info('Restoring TEMP values for CS flags where PLA exists %s' % profile.XBT_input_filename)
        # update the TEMP values with the previous value
        df.loc[ind, 'TEMP'] = temps
    # makes sure we have the same number of CS flags in the profile data as in the histories before proceeding
    elif (len(ind) > 0) & (len(temps) == len(ind)):
        # temps should be equal to df['TEMP_RAW'][ind], let's check they are equal and there are no missing values
        if (temps != df['TEMP_RAW'][ind]).all() and (temps.max() <= 99) and (df['TEMP_RAW'][ind].max() <= 99):
            # check they are within 0.01 of each other
            if not np.allclose(temps, df['TEMP_RAW'][ind], atol=0.01):
                # check the median difference with a bigger tolerance:
                if np.median(np.abs(temps - df['TEMP_RAW'][ind])) > 0.01:
                    LOGGER.error('TEMP_RAW values do not match the HISTORY_PREVIOUS_VALUE for CS flags %s'
                                 % profile.XBT_input_filename)
                    return profile

        # update the TEMP values with the TEMP_RAW values if they do not contain values > 99
        if not (df['TEMP_RAW'][ind] > 99).any():
            df.loc[ind, 'TEMP'] = df.loc[ind, 'TEMP_RAW']
        # update the TEMP_RAW values with the HISTORY_PREVIOUS_VALUE values if the TEMP_RAW values have values > 99 and the
        # HISTORY_PREVIOUS_VALUE values do not
        elif not (temps > 99).any() and (df['TEMP_RAW'][ind] > 99).any():
            df.loc[ind, 'TEMP_RAW'] = temps
            df.loc[ind, 'TEMP'] = temps
        else:
            LOGGER.error('TEMP_RAW values and HISTORY_PREVIOUS_VALUE values are both > 99 for CS flags. Please review. %s'
                         % profile.XBT_input_filename)
    elif len(depths) == 0:
        LOGGER.warning('No CS flags found in the histories. %s' % profile.XBT_input_filename)
    else:
        # the number of CS flags in the profile data does not match the number of missing temps in the data
        LOGGER.warning('Number of CS flags in the profile data does not match the number of missing temps in the data. %s'
                     % profile.XBT_input_filename)

    # find any depths with 99.99 values that are flagged with SPA or IPA or HFA
    idx = (df['TEMP'] > 99)
    if idx.any() and ind.any():
        # check if there are any SPA, IPA or HFA flags at the same depth
        idx2 = profile.histories['HISTORY_START_DEPTH'].isin(df.loc[idx, 'DEPTH'])
        if idx2.any():
            # get the flags
            flags = profile.histories.loc[idx2, 'HISTORY_QC_CODE']
            # if SPA, IPA or HFA flags are present, update the TEMP values to be 99.99
            if flags.str.contains('SPA|IPA|HFA').any():
                # are these flags adjacent to a CSR flag?
                # get the depths of the SPA, IPA or HFA flags
                depths2 = profile.histories.loc[idx2, 'HISTORY_START_DEPTH'].values
                # find the depths in the profile data
                ind2 = np.in1d(np.round(df['DEPTH'], 2), np.round(depths2, 2)).nonzero()[0]
                temps = profile.histories['HISTORY_PREVIOUS_VALUE'][ind2].values.astype('float')
                # is the first value of ind2 only one different from last value of ind?
                if (ind2[0] - ind[-1]) == 1:
                    LOGGER.info('Restoring 99.99 values for SPA, IPA or HFA flags and changing flag to CSR. %s'
                                % profile.XBT_input_filename)
                    # update the TEMP values with the TEMP_RAW values if they do not contain values > 99
                    if not (df['TEMP_RAW'][ind2] > 99).any():
                        df.loc[ind2, 'TEMP'] = df.loc[ind2, 'TEMP_RAW']
                    # update the TEMP_RAW values with the HISTORY_PREVIOUS_VALUE values if the TEMP_RAW values have values > 99 and the
                    # HISTORY_PREVIOUS_VALUE values do not
                    elif not (temps > 99).any() and (df['TEMP_RAW'][ind2] > 99).any():
                        df.loc[ind2, 'TEMP_RAW'] = temps
                        df.loc[ind2, 'TEMP'] = temps
                    else:
                        LOGGER.error('TEMP_RAW values and HISTORY_PREVIOUS_VALUE values are both > 99 for CS flags %s'
                                     % profile.XBT_input_filename)
                        exit(1)
                    # update the TEMP_quality_control values
                    df.loc[ind2, 'TEMP_quality_control'] = 3
                    # update the TEMP_QC_CODE to CSR
                    profile.histories.loc[idx2, 'HISTORY_QC_CODE'] = 'CSR'
                    # update the TEMP_QC_CODE_VALUE to 3
                    profile.histories.loc[idx2, 'HISTORY_QC_CODE_VALUE'] = 3
                    # if there are any SPA, IPA or HFA flags at the same depth as the CSR flags, remove them
                    # get the location of any SPA, IPA or HFA flags at the same depth as the CSR flags in the profile.histories
                    idx3 = profile.histories['HISTORY_QC_CODE'].str.contains('SPA|IPA|HFA')
                    if idx3.any():
                        LOGGER.info('Removing SPA, IPA or HFA flags at the same depth as CSR flags. %s'
                                    % profile.XBT_input_filename)
                        # Get the depths where HISTORY_QC_CODE is CSR
                        csr_depths = profile.histories.loc[
                            profile.histories['HISTORY_QC_CODE'] == 'CSR', 'HISTORY_START_DEPTH']

                        # Remove rows where HISTORY_QC_CODE is SPA, HFA, or IPA and HISTORY_START_DEPTH is in csr_depths
                        profile.histories = profile.histories[
                            ~((profile.histories['HISTORY_QC_CODE'].isin(['SPA', 'HFA', 'IPA'])) &
                              (profile.histories['HISTORY_START_DEPTH'].isin(csr_depths)))]
                        # reset the index
                        profile.histories = profile.histories.reset_index(drop=True)

    # are there any TEMP values that are still > 99?
    if (df['TEMP'] > 99).any():
        # see if any of the histories have a valid TEMP value for these depths
        idx = df['TEMP'] > 99
        depths = df.loc[idx, 'DEPTH']
        idx2 = (np.isclose(profile.histories['HISTORY_START_DEPTH'], (depths), atol=1e-6) &
                (profile.histories['HISTORY_PARAMETER'].str.contains('TEMP') &
                 (profile.histories['HISTORY_PREVIOUS_VALUE'] < 99)))
        if idx2.any():
            LOGGER.info('Restoring TEMP values for depths where TEMP > 99. %s' % profile.XBT_input_filename)
            # assign the previous_value at idx2 to the TEMP values at idx
            df.loc[idx, 'TEMP'] = profile.histories.loc[idx2, 'HISTORY_PREVIOUS_VALUE'].values
            # assign to TEMP_RAW as well
            if (df['TEMP_RAW'][idx] > 99).any():
                df.loc[idx, 'TEMP_RAW'] = profile.histories.loc[idx2, 'HISTORY_PREVIOUS_VALUE'].values
            # check again if there are any TEMP values that are still > 99
            if (df['TEMP'] > 99).any():
                LOGGER.warning('TEMP values are still > 99 after restoration. %s' % profile.XBT_input_filename)

    # update profile data
    profile.data = df
    return profile


def create_flag_feature(profile):
    """ Take the existing QC code values and turn them into a integer representation. One bit for every code.
    And there are now two variables, one for accept codes, one for reject codes."""

    # create a dataframe with the codes and their integer representation
    df = read_qc_config()
    # make a new column in df with just the first two characters of the code column
    df['code_short'] = df['code'].str[:2]
    df_data = profile.data.copy(deep=True)

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
        LOGGER.warning('TEMP_quality_control values are 0 and TEMP_RAW_quality_control values are not. Updating. %s'
                       % profile.XBT_input_filename)
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
                               'HISTORY_QC_CODE_VALUE': 1,
                               'HISTORY_SOFTWARE_RELEASE': '',
                               'HISTORY_PREVIOUS_VALUE': 0}, ignore_index=True)

    # only continue if there are codes to map
    if codes.empty:
        profile.histories = codes
        return profile

    # first get the quality at each depth and add the information to the history table
    for idx, row in codes.iterrows():
        # get the index of the depth in the data
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        codes.loc[idx, 'tempqc'] = df_data.loc[ii, 'TEMP_quality_control']
    # for CSR flags, replace the tempqc values with the TEMP_quality_control value that is one deeper than the deepest CSR flag
    # get the index of the CS flags
    idx_csr = codes['HISTORY_QC_CODE'].str.contains('CSR')
    # get the depths of the CS flags
    depths = codes.loc[idx_csr, 'HISTORY_START_DEPTH'].values
    # if there are CSR flags
    if len(depths) > 0:
        # find the next deepest depth
        ideps = df_data['DEPTH'] > depths[-1]
        # update any codes['tempqc'] where start_depth == 0
        idx = codes['HISTORY_START_DEPTH'] == df_data['DEPTH'].values[0]
        codes.loc[idx, 'tempqc'] = df_data.loc[ideps, 'TEMP_quality_control'].values[0]
        # special case where CSR was used as a single flag to reject everything below. Let's change this flag to a SPR
        if len(depths) == 1 and df_data.loc[ideps, 'TEMP_quality_control'].values[0] == 3:
            codes.loc[idx_csr, 'HISTORY_QC_CODE'] = 'SPR'
            codes.loc[idx_csr, 'HISTORY_QC_CODE_VALUE'] = 4

    # check the TEMP_quality_control values are the same as the HISTORY_QC_CODE_VALUE values
    for idx, row in codes.iterrows():
        # check here that the TEMP_quality_control value is the same as the tempqc value
        # skip the CSR and position flags as they are handled specifically
        if row['HISTORY_QC_CODE'] not in ['REA','TEA','LAA','LOA','PER','TER','CSR']:
            if row['tempqc'] != row['HISTORY_QC_CODE_VALUE']:
                # get the df['tempqc'] value for the two-character code
                tempqc = df.loc[df['code_short'].str.contains(row['HISTORY_QC_CODE'][:2]), 'tempqc'].values
                # check if the two character code appears more than once in the df['code_short'] column
                if np.size(tempqc) > 1:
                    # if so, then we need to check that the TEMP_quality_control value is in the same category as the tempqc value
                    # where the categories are 1,2,5 and 3,4
                    if ((row['HISTORY_QC_CODE_VALUE'] in [1, 2, 5] and row['tempqc'] in [3 ,4]) or
                            (row['HISTORY_QC_CODE_VALUE'] in [3, 4] and row['tempqc'] in [1, 2, 5])):
                        # update the HISTORY_QC_CODE_VALUE to the tempqc value as the TEMP_quality_control value is in the wrong category
                        if row['tempqc'] in [1, 2, 5]:
                            codes.loc[idx, 'HISTORY_QC_CODE_VALUE'] = tempqc[0]
                            # also change the HISTORY_QC_CODE to A
                            codes.loc[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'][:2] + 'A'
                        else:
                            codes.loc[idx, 'HISTORY_QC_CODE_VALUE'] = tempqc[1]
                            # also change the HISTORY_QC_CODE to R
                            codes.loc[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'][:2] + 'R'
                else:
                    # if the two character code only appears once, then update the tempqc value in the codes table
                    codes.loc[idx, 'HISTORY_QC_CODE_VALUE'] = tempqc
                    if tempqc in [0, 1, 2, 5]:
                        # also change the HISTORY_QC_CODE to A
                        codes.loc[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'][:2] + 'A'
                    else:
                        # also change the HISTORY_QC_CODE to R
                        codes.loc[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'][:2] + 'R'

    # delete the tempqc column in codes, no longer required
    codes = codes.drop(columns=['tempqc'])
    # delete the code_short column in df, no longer required
    df = df.drop(columns=['code_short', 'group_label'])

    # make sure the previous_values are the same as the data['TEMP_RAW'] values and replace missing TEMP values at CS
    profile.histories = codes
    profile.data = df_data
    profile = restore_temp_val(profile)
    codes = profile.histories
    df_data = profile.data

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
        if mapcodes.loc[nan_values, 'HISTORY_QC_CODE'].str.contains("BB|DC|GS|MS").any():
            mapcodes = mapcodes[~nan_values]
        else:
            LOGGER.error('HISTORY: new QC code encountered, please code in the new value. %s %s' % (
                mapcodes.loc[nan_values, 'HISTORY_QC_CODE'].unique(), profile.XBT_input_filename))
            exit(1)

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
        if row['HISTORY_QC_CODE'] in ['SPA', 'HFA', 'IPA', 'EIA']:
            # 2 should have been assigned above, now just overwriting with 5
            tempdf.loc[ii, row['code']] = 5

    # index of the tempdf rows that have a value of 5
    idx = tempdf.eq(5).any(axis=1)
    # calculate the maximum tempqc value for each depth
    tempdf['tempqc'] = tempdf.max(axis=1)
    # overwrite the tempqc value with 5 where there is a 5 in the tempdf
    tempdf.loc[idx, 'tempqc'] = 5

    # find any depths where the tempqc value is less than the TEMP_quality_control value not including the 5 values
    # and ignore where LOA has changed the QC to 2 from 1
    idx = (df_data['TEMP_quality_control'] > tempdf['tempqc']) & (df_data['TEMP_quality_control'] != 5)
    if idx.any() & ~(codes['HISTORY_QC_CODE'].str.contains('LOA')).any():
        LOGGER.warning('TEMP_quality_control values are greater than the tempqc values. %s' % profile.XBT_input_filename)

    # update the TEMP_quality_control field with the tempdf values
    df_data['TEMP_quality_control'] = tempdf['tempqc']

    # Iterate over the history table.
    for idx, row in mapcodes.iterrows():
        # Get depth index
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        # if this is an accept code (QC_Flag = 1, 2, 3, 5) then add it to the accept code array
        if row['HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5]:
            # adding them together - is there a more correct way to do this?
            # Add byte values (masks) for accept codes
            df_data.loc[ii, 'XBT_accept_code'] = df_data.loc[ii, 'XBT_accept_code'] + np.float64(row['byte_value'])
        else:
            # Add byte values (masks) for reject codes
            df_data.loc[ii, 'XBT_reject_code'] = df_data.loc[ii, 'XBT_reject_code'] + np.float64(row['byte_value'])

    # update the histories with the correct tempqc values from mapcodes
    mapcodes['HISTORY_QC_CODE_VALUE'] = mapcodes['tempqc']
    # drop unwanted columns
    mapcodes = mapcodes.drop(columns=['tempqc', 'byte_value', 'label', 'code'])
    df_data = df_data.drop(columns=['tempqc'])

    # update the histories
    profile.histories = mapcodes
    # update the profile data
    profile.data = df_data

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
        LOGGER.error('Profile not processed, No data in the file: %s' % profile.XBT_input_filename)
        return False

    if (data_type != 'XB') and (data_type != 'BA'):  # and data_type != 'BA' and data_type != 'TE':
        LOGGER.error('Profile not processed as it is type %s %s ' % (data_type, profile.XBT_input_filename))
        return False

    if duplicate_flag == 'D':
        LOGGER.error(
            'Profile not processed. Tagged as duplicate profile in original netcdf file %s' % profile.XBT_input_filename)
        return False

    if 'DU' in histcodes:
        LOGGER.error(
            'Profile not processed. Tagged as duplicate profile in original netcdf file %s' % profile.XBT_input_filename)
        return False

    data_vars = temp_prof_info(profile.netcdf_file_obj)
    if 'TEMP' not in data_vars.values():
        LOGGER.error('Profile not processed, no TEMP in file %s' % profile.XBT_input_filename)
        return False

    return True


def make_dataframe(profile_ed, profile_raw, profile_turo):
    # convert the data in profile to a parquet file
    # create a dataframe from the profile data
    df = pd.DataFrame(profile_ed.data)
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


def set_metadata(tbl, tbl_meta):
    """Store table- and column-level metadata as json-encoded byte strings.

    Table-level metadata is stored in the table's schema.
    Column-level metadata is stored in the table columns' fields.

    To update the metadata, first new fields are created for all columns.
    Next a schema is created using the new fields and updated table metadata.
    Finally a new table is created by replacing the old one's schema, but
    without copying any data.

    Args:
        tbl (pandas dataframe): The table to store metadata in
        col_meta: A json-serializable dictionary with column metadata in the form
            {
                'column_1': {'some': 'data', 'value': 1},
                'column_2': {'more': 'stuff', 'values': [1,2,3]}
            }
        tbl_meta: A json-serializable dictionary with table-level metadata.
    """
    # Convert the pandas dataframe to a pyarrow table
    tbl = pa.Table.from_pandas(tbl)

    # Get column metadata
    col_meta, var_list = generate_table_att(os.path.join(os.path.dirname(__file__), 'generate_nc_file_att'))
    # Create updated column fields with new metadata
    if col_meta:
        fields = []
        for col in tbl.schema.names:
            if col in col_meta:
                # Get updated column metadata
                metadata = tbl.field(col).metadata or {}
                for k, v in col_meta[col].items():
                    metadata[k] = json.dumps(v).encode('utf-8')
                # Update field with updated metadata
                fields.append(tbl.field(col).with_metadata(metadata))
            else:
                fields.append(tbl.field(col))

        # Get updated table metadata
        tbl_metadata = tbl.schema.metadata or {}
        for k, v in tbl_meta.items():
            if type(v) == bytes:
                tbl_metadata[k] = v
            else:
                tbl_metadata[k] = json.dumps(v).encode('utf-8')

        # Create new schema with updated field metadata and updated table metadata
        schema = pa.schema(fields, metadata=tbl_metadata)

        # With updated schema build new table (shouldn't copy data)
        # tbl = pa.Table.from_batches(tbl.to_batches(), schema)
        tbl = tbl.cast(schema)

    return tbl


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
        # if f != 61024487:
        #     continue
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
                if not profile_ed:
                    continue
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
            LOGGER.warning('Profile not processed, file %s is in keys file, but does not exist' % f)
    # add table metadata to the dfall dataframe
    dfall = set_metadata(dfall, tbl_meta={'Parent file':keys.dbase_name})
    # write the dataframe to a parquet file
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name), os.path.basename(keys.dbase_name) + '.parquet')
    pq.write_table(dfall, pq_filename)
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name),
                               os.path.basename(keys.dbase_name) + '_histories.parquet')
    dfhist.to_parquet(pq_filename, index=False)
    pq_filename = os.path.join(os.path.dirname(keys.dbase_name), os.path.basename(keys.dbase_name) + '_globals.parquet')
    globsall.to_parquet(pq_filename, index=False)

    print('All done')