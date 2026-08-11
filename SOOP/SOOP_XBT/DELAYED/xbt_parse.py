#!/usr/bin/python3

import argparse
import sys
import tempfile
from netCDF4 import Dataset
import difflib
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from lib.python.imos_logging import IMOSLogging
from lib.python.ship_callsign import ship_callsign_list
from lib.python.xbt_line_vocab import xbt_line_info
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
            profile = XbtProfile(fid) # Reads the profile and metadata.
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
        self.Input_filename = input_filename

        # now read the data and metadata from the file
        # LOGGER.info('Parsing %s' % self.Input_filename)
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
        if filename.endswith('_keys.nc'):
            self.keys_file_path = filename.input_path
            self.dbase_name = self.keys_file_path.replace('_keys.nc', '')
        else:
            self.dbase_name = filename
            self.keys_file_path \
                = '{campaign_path}_keys.nc'.format(campaign_path=filename.rstrip(os.path.sep))

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
            data_type = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['data_t'][:].data
                     if bytearray(xx).strip()]
            # keep only the data_type that are 'XB'
            data_type = np.asarray(data_type)
            ikeep = np.where(data_type == 'XB')[0]

            station_number = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in
                              netcdf_file_obj['stn_num'][:].data if bytearray(xx).strip()]
            # change station number to a numpy array
            station_number = np.asarray(station_number, dtype=np.int32)
            station_number = station_number[ikeep]
            # sort it and keep unique station numbers where sometimes the keys has multiple values
            order = np.argsort(station_number)
            station_number = station_number[order]

            # read in the position information
            latitude = np.round(netcdf_file_obj['obslat'][:].data, 6)
            longitude = np.round(netcdf_file_obj['obslng'][:].data, 6)
            latitude = latitude[ikeep]
            longitude = longitude[ikeep]
            # sort them as per the station number
            latitude = latitude[order]
            longitude = longitude[order]
            # decode date/time information

            # callsign
            calls = [''.join(chr(x) for x in bytearray(xx)) for xx in netcdf_file_obj['callsign'][:].data
                     if bytearray(xx)]
            # remove control characters from the callsign list
            calls = [remove_control_chars(x) for x in calls]
            # replace any calls that are just spaces with "Unknown"
            calls = [x if x.strip() else 'Unknown' for x in calls]
            # strip any leading or trailing spaces from the callsigns
            calls = [x.strip() for x in calls]
            # make a numpy array before indexing with ikeep (which is a numpy array)
            calls = np.asarray(calls)[ikeep]
            # sort the same as station number and convert back to list
            calls = np.asarray(calls)[order].tolist()

            # get the date/time information
            year = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['obs_y'][:].data
                     if bytearray(xx).strip()]
            month = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['obs_m'][:].data
                     if bytearray(xx).strip()]
            day = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['obs_d'][:].data
                     if bytearray(xx).strip()]
            # create a datetime object for each profile
            try:
                date_time = [datetime(int(y), int(m), int(d))
                                  for y, m, d in zip(year, month, day)]
            except ValueError as e:
                # use a default day of 1 with year and month as we are only interested in the year for grouping
                LOGGER.error('Error parsing date/time in keys file %s: %s' % (self.keys_file_path, e))
                date_time = [datetime(int(y), int(m), 1)
                                  for y, m in zip(year, month)]

            # sort the date_time as per station number
            # convert to numpy array before indexing with ikeep
            date_time = np.asarray(date_time)[ikeep]
            date_time = np.asarray(date_time)[order].tolist()
            # create the data dictionary
            self.data = {}
            self.data = {'station_number': [int(x) for x in station_number], 'latitude': [x for x in latitude],
                         'longitude': [x for x in longitude], 'callsign': [x for x in calls], 'date': [x for x in date_time]}

    def group_by_year(self, group_by=False):
        # group the data by year and return a dictionary with years as keys and lists of station numbers as values
        year_dict = {}
        if not group_by:
            year = self.data['date'][0].year
            year_dict[year] = []
            # return a dictionary with all station_numbers sorted by station_number
            for i, station in enumerate(self.data['station_number']):
                year_dict[year].append(station)
        else:
            for i, date in enumerate(self.data['date']):
                year = date.year
                if year not in year_dict:
                    year_dict[year] = []
                year_dict[year].append(self.data['station_number'][i])
        return year_dict

    def group_by_callsign_year(self, year_dictionary=None):
        # get the unique callsigns, group by callsign and year
        callsign_dict = {}
        if year_dictionary is None:
            year_dictionary = self.group_by_year(group_by=True)
        # iterate through the data and group by callsign and year
        for i, callsign in enumerate(self.data['callsign']):
            year = self.data['date'][i].year
            if callsign not in callsign_dict:
                callsign_dict[callsign] = {}
            if year not in callsign_dict[callsign]:
                callsign_dict[callsign][year] = []
            # append the station number to the list for this callsign and year
            callsign_dict[callsign][year].append(self.data['station_number'][i])
        return callsign_dict


def coordinate_data(profile_qc, profile_noqc, profile_raw, station_number):
    # perform checks and adjustments and combine data in preparation for writing out
    profile_qc, profile_noqc = parse_data_nc(profile_qc, profile_noqc, profile_raw, station_number)

    # let's check if there are histories to parse and then handle
    profile_qc = parse_histories_nc(profile_qc)
    if int(profile_noqc.netcdf_file_obj['Num_Hists'][0].data) == 0:
        # if there are no histories in the noqc file, apply an empty dataframe with the same columns and data types as profile_qc
        profile_noqc.histories = pd.DataFrame({col: pd.Series(dtype=profile_qc.histories[col].dtype) for col in profile_qc.histories.columns})
    else:
        # we need to carry the depths information into the history parsing, so copy the data array into profile_noqc
        profile_noqc.data = profile_qc.data
        profile_noqc = parse_histories_nc(profile_noqc)
        # delete the data array from profile_noqc as it is not needed anymore
        del profile_noqc.data
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

    # Probe type goes into a variable with coefficients as attributes, and assign QC to probe types
    profile_qc = get_fallrate_eq_coef(profile_qc, profile_noqc)
    # if probetype is not XBT return empty profile_qc
    if profile_qc.data['PROBE_TYPE'].all() == '':
        return []

    # check that the sums of TEMP and TEMP_RAW and DEPTH and DEPTH_RAW are the same within a tolerance
    # check_sums_of_temp_depth(profile_qc)

    # add uncertainties:
    profile_qc.data = add_uncertainties(profile_qc.data)

    # add Launcher variable and assign 'LM-3A Hand-Held' if the vessel is not l'Astrolabe and date is less than 2020-11-01
    # else assign 'LM-4A Thru-Hull'
    profile_qc.data = add_launcher_variable(profile_qc.data)

    # remove columns that are all NaN
    # profile_qc.data = profile_qc.data.dropna(axis=1, how='all')

    return profile_qc


def get_recorder_type(df):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')
    syst_list = read_section_from_xbt_config('SYST')

    # if df['RECORDER_TYPE'][0] is nan, return '99' and 'Unknown'
    if pd.isna(df['RECORDER_TYPE'][0]):
        LOGGER.warning('RECORDER_TYPE is NaN in xbt_config file, using unknown for recorder. %s' % df['Input_filename'][0])
        item_val = '99'
        return item_val, rct_list[item_val].split(',')[0]
    item_val = str(int(df['RECORDER_TYPE'][0]))
    #        if item_val in list(syst_list.keys()):
    #            item_val = syst_list[item_val].split(',')[0]

    if item_val in list(rct_list.keys()):
        return item_val, rct_list[item_val].split(',')[0]
    else:
        LOGGER.warning(
            '{item_val} missing from recorder type part in xbt_config file, using unknown for recorder. %s'.format(
                item_val=item_val) % df['Input_filename'][0])
        item_val = '99'
        return item_val, rct_list[item_val].split(',')[0]


def parse_extra_vars(profile_qc, profile_noqc):
    """
    retrieve surface codes and some additional variables from input NetCDF file object
    """
    dataf = profile_qc.data.copy()
    # read the variable names from the netcdfVars config file
    vars_list = read_variables_config()
    # separate the dataframe into surface codes and other variables
    # where surface codes are in the mquest column and have either 4 character codes or a ';' separator
    mquest_has_sep = vars_list['Mquest'].astype('string').str.contains(';', na=False)
    mquest_len_4 = vars_list['Mquest'].astype('string').str.len().eq(4)
    srfc_code_list = vars_list[mquest_has_sep | mquest_len_4].copy()
    # where the mquest column contains ';' separate the codes into a list
    # use assign to avoid dtype-assignment issues on older pandas versions
    srfc_code_list = srfc_code_list.assign(Mquest=srfc_code_list['Mquest'].astype(str).str.split(';'))
    # now explode the mquest column to have one row per code
    srfc_code_list = srfc_code_list.explode('Mquest').reset_index(drop=True)

    # other variables are the ones remaining
    vars_list = vars_list[~mquest_has_sep & ~mquest_len_4]

    # transfer the non-surface code variables to the dataf dataframe
    ext = ['','_RAW']
    for ind, profile in enumerate([profile_qc, profile_noqc]):
        for index, row in vars_list.iterrows():
            # var_name is either variable_name or Attribute Name, whichever is not NaN
            var_name = row['variable_name']
            # var is the Mquest code for the variable
            var = row['Mquest']
            if var in list(profile.netcdf_file_obj.variables.keys()):
                vv = decode_bytearray(profile.netcdf_file_obj[var][:])
                if not vv or len(vv) == 0:
                    dataf[var_name  + ext[ind]] = ''
                else:
                    data = remove_control_chars(vv).strip()
                    data = data.replace(' ', '')
                    # if the variable is institution, create a dictionary of the institution codes
                    if var == 'Stream_Ident':
                        institute_list = read_section_from_xbt_config('INSTITUTE')
                        # remove the last two characters from the string
                        data = data[:-2]
                        if data in list(institute_list.keys()):
                            dataf[var_name + ext[ind]] = institute_list[data].split(',')[0]
                            dataf['Institution_code' + ext[ind]] = institute_list[data].split(',')[1]
                        else:
                            LOGGER.warning('Agency_GTS_code code %s is not defined in xbt_config file. Please edit xbt_config %s'
                                           % (data, profile.Input_filename))
                        continue
                    if var == 'Digit_Code' or var == 'Standard':
                        for count in range(profile.nprof):
                            vv = decode_bytearray(profile.netcdf_file_obj[var][count])
                            if not vv or len(vv) == 0:
                                dataf[var_name + '_' + profile.prof_type[count] + ext[ind]] = ''
                            else:
                                dataf[var_name + '_' + profile.prof_type[count] + ext[ind]] = remove_control_chars(vv).strip()
                    else:
                        dataf[var_name + ext[ind]] = data
            else:
                dataf[var_name + ext[ind]] = ''

        # Parse the surface codes into the variables too
        srfc_code_nc = profile.netcdf_file_obj['SRFC_Code'][:]
        srfc_parm = profile.netcdf_file_obj['SRFC_Parm'][:]
        nsrf_codes = int(profile.netcdf_file_obj['Nsurfc'][:])

        # cycle through the surface codes and assign them to the dataframe
        missing_codes = []
        for i in range(nsrf_codes):
            srfc_code_iter = decode_bytearray(srfc_code_nc[i])
            # print(srfc_code_iter)
            # check srfc_code_iter is in the mquest column of the srfc_code_list dataframe
            if srfc_code_iter in srfc_code_list['Mquest'].values:
                # get the index of the srfc_code_iter in the srfc_code_list dataframe where the mquest column can have multiple values per row separated by ';'
                srfc_code_index = srfc_code_list[srfc_code_list['Mquest'] == srfc_code_iter].index[0]
                # now get the attributes for this srfc_code_iter
                att_name = srfc_code_list.loc[srfc_code_index, 'variable_name']
                att_type = srfc_code_list.loc[srfc_code_index, 'variable_type']
                att_val = decode_bytearray(srfc_parm[i])
                # remove control characters from the attribute value
                att_val = remove_control_chars(att_val).strip()
                # if the att_name contains 'date' then convert it to a datetime object
                if 'date' in att_name.lower():
                    # check for characters such as '/' or '-' in the string
                    pattern = re.compile(r'[\-\/]')
                    # build a format string based on the presence of these characters
                    if pattern.search(att_val):
                        format = '%m-%d-%Y' if '-' in att_val else '%m/%d/%Y'
                    else:
                        format = '%m%d%Y'  # assume the date is in the format mmddyyyy
                    # try to convert the string to a datetime object
                    att_val_conv = convert_time_string(att_val, format, 'string', '%Y%m%d')
                    if att_val_conv is None:
                        # try converting with no format, dayfirst = true
                        att_val_conv = pd.to_datetime(att_val,errors='coerce')
                        # check if att_val_conv is NaNT
                        if not pd.isna(att_val_conv):
                            # convert to a string
                            att_val = att_val_conv.strftime('%Y%m%d')
                        else:
                            # if conversion fails, set to deployment date
                            att_val = dataf['TIME'][0].strftime('%Y%m%d')
                            LOGGER.error('"%s = %s" could not be converted to date format. Using deployment date %s' % (
                                att_name, att_val, dataf['TIME'][0].strftime('%Y%m%d')))
                    else:
                        # successfully converted to a string
                        att_val = att_val_conv
                try:
                    if 'float' in att_type:
                        dataf[att_name + ext[ind]] = float(att_val.replace(' ', ''))
                    else:
                        dataf[att_name + ext[ind]] = att_val.replace(' ', '')
                except ValueError:
                    LOGGER.warning(
                        '"%s = %s" could not be converted to %s(). Please review. %s' % (
                        att_name, att_val, att_type.upper(), profile.Input_filename))
                    continue
            else:
                if srfc_code_iter != '' and srfc_code_iter != 'IOTA':
                    # collect the code in a list for the user to review
                    missing_codes.append(srfc_code_iter)

        if missing_codes:
                LOGGER.warning('%s codes not defined in srfc_code in netcdfVars file. Please edit xbt_config %s'
                               % (missing_codes, profile.Input_filename))

        # if the platform code didn't come through, assign unknown type
        if ('Callsign' + ext[ind] not in dataf.columns) or not is_string_or_list_of_strings(dataf['Callsign' + ext[ind]].unique().item()):
            LOGGER.warning('Callsign is missing, GCLL has not been read or is missing. %s' % profile.Input_filename)
            # assign unknown to the platform code
            dataf['Callsign' + ext[ind]] = 'Unknown'
            dataf['Platform_code' + ext[ind]] = 'Unknown'
            dataf['Ship_name' + ext[ind]] = 'Unknown'
            dataf['Ship_IMO' + ext[ind]] = 'Unknown'

        # get the ship details
        # note that the callsign and ship name are filled from the original file values, but will be replaced here if they exist in the AODN vocabulary
        # for these older historical files, the Callsign and Platform_code are the same. In newer files, the platform_code
        # will be the GTSID or SOTID.
        dataf['Platform_code' + ext[ind]] = dataf['Callsign' + ext[ind]]  # set here as can't have duplicate assignments in the config file
        ships = SHIP_CALL_SIGN_LIST
        calls = dataf['Platform_code' + ext[ind]].unique().item()
        if calls in ships:
            dataf['Ship_name' + ext[ind]] = ships[calls][0]
            dataf['Ship_IMO' + ext[ind]] = ships[calls][1]
        elif calls !='Unknown' and difflib.get_close_matches(calls, ships, n=1, cutoff=0.8) != []:
            dataf['Callsign' + ext[ind]] = \
                difflib.get_close_matches(calls, ships, n=1, cutoff=0.8)[0]
            dataf['Ship_name' + ext[ind]] = ships[dataf['Callsign' + ext[ind]].unique().item()][0]
            dataf['Ship_IMO' + ext[ind]] = ships[dataf['Callsign' + ext[ind]].unique().item()][1]
            LOGGER.warning(
                'PLATFORM_CODE: Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s %s' % (
                    dataf['Platform_code' + ext[ind]].unique().item(), dataf['Callsign' + ext[ind]].unique().item(), profile.Input_filename))
        else:
            dataf['Platform_code' + ext[ind]] = 'Unknown'
            dataf['Ship_name' + ext[ind]] = 'Unknown'
            dataf['Ship_IMO' + ext[ind]] = 'Unknown'

        # extract the information and assign correctly
        if 'RECORDER_TYPE' + ext[ind] in dataf.columns:
            recorder_val, recorder_type = get_recorder_type(dataf)
            dataf['RECORDER_TYPE' + ext[ind]] = recorder_val
            dataf['RECORDER_TYPE_name' + ext[ind]] = recorder_type
        else:
            dataf['RECORDER_TYPE_name' + ext[ind]] = 'Unknown'
            dataf['RECORDER_TYPE' + ext[ind]] = '99'

        # check deployment height
        if 'Height_launch_above_water' + ext[ind] in dataf.columns:
            if dataf['Height_launch_above_water_meters'].unique().item() > 50:
                LOGGER.warning('HTL$, xbt launch height attribute seems to be very high. Please review: %s meters %s' %
                               (dataf['Height_launch_above_water_meters'].unique().item(), profile.Input_filename))

        # some files don't have line information
        if 'SOOP_line_label' + ext[ind] in dataf.columns:
            line = dataf['SOOP_line_label' + ext[ind]].unique().item()
            if not line or pd.isna(line):
                line = 'NOLINE'
                dataf['SOOP_line_label' + ext[ind]] = 'NOLINE'
                LOGGER.warning('XBT line is not recorded, assigning NOLINE %s' %
                               profile.Input_filename)
        else:
            line = 'NOLINE'
            dataf['SOOP_line_label' + ext[ind]] = 'NOLINE'
            LOGGER.warning('XBT line is not recorded, assigning NOLINE %s' %
                           profile.Input_filename)

        xbt_line_codes = [s for s in list(XBT_LINE_INFO.keys())]  # IMOS codes taken from vocabulary
        if line in xbt_line_codes:
            xbt_line_att = XBT_LINE_INFO[line]
            dataf['SOOP_line_description' + ext[ind]] = xbt_line_att[1]
        else:
            LOGGER.error(
                'SOOP line : "%s" is not defined in AODN vocabs.ands.org.au(contact AODN) %s' %
                (line, profile.Input_filename))

    # check for differences in the extra variables between the qc and noqc files and remove redundant columns
    # all data is in the dataf dataframe and we need to check if the *_noqc columns are the same as the non-RAW columns
    # if they are, we can remove the *_RAW columns
    for col in dataf.columns:
        # skip if the column contains TIME*, TEMP*, DEPTH*, LATITUDE*, LONGITUDE*, PROBE*
        if re.match(r'TIME|TEMP|PSAL|SSPD|DEPTH|LATITUDE|LONGITUDE|PROBE_TYPE', col):
            continue
        if '_RAW' in col:
            # check if the column exists without the _RAW and if it does, check if the data is the same
            if col.replace('_RAW', '') in dataf.columns:
                if dataf[col].equals(dataf[col.replace('_RAW', '')]):
                    dataf = dataf.drop(col, axis=1)
                else:
                    LOGGER.error('Column %s in *_RAW file is not the same as the non-RAW column. Please review %s' %
                                 (col, profile_qc.Input_filename))

    # split the input filename and remove the _ed.nc or _raw.nc ending
    dataf['Input_filename'] = re.split(r'ed\.nc|raw\.nc', profile_qc.Input_filename)[0]

    # convert the 'PROBE_manufacture_date' to a timestamp
    if 'PROBE_manufacture_date_YYYYMMDD' in dataf.columns:
        date1 = convert_time_string(dataf['PROBE_manufacture_date_YYYYMMDD'], '%Y%m%d')
        date2 = convert_time_string(dataf['PROBE_manufacture_date_YYYYMMDD'], '%m%d%Y')
        # if date1 is not NaT, assign it to the column, otherwise assign date2
        dataf['PROBE_manufacture_date_YYYYMMDD'] = date1 if not date1.isna().all() else date2

    # if the 'Institution_unique_identifier' is nan, assign the 'station_number' to it
    if dataf['Institution_unique_identifier'].isna().all():
        if 'station_number' in dataf.columns:
            dataf['Institution_unique_identifier'] = dataf['station_number'].astype(str)
        else:
            LOGGER.warning('Institution_unique_identifier is NaN and station_number is not available. %s' %
                           profile.Input_filename)
            dataf['Institution_unique_identifier'] = 'Unknown'

    # assign dataf to profile_qc.data
    profile_qc.data = dataf

    return profile_qc, profile_noqc


def parse_data_nc(profile_qc, profile_noqc, profile_raw, station_number):
    """ Parse variable data from all sources into a dictionary attached to the profile_qc structure
    """
    # create column headers from the variable names in generate_nc_file_att file
    variable_names = read_variables_config()
    # remove variable_names in the dataframe that are not in the profile_qc.data
    variable_names = variable_names[~variable_names['variable_name'].str.startswith('HISTORY_')]
    # create profile_qc.data using the variable names as columns
    profile_qc.data = pd.DataFrame(columns=variable_names['variable_name'].tolist())
    # check both raw and edited ndeps and return profile_qc.data with the maximum number of depths
    profile_qc = check_profile_lengths(profile_qc, profile_noqc)
    # record the ndeps in the profile_qc dataframe for later use
    ndeps = profile_qc.data['DEPTH'].size

    # Pressure/depth information from both noqc and qc files
    for s in [profile_qc, profile_noqc]:
        # assign '_RAW' if s is profile_noqc, otherwise assign ''
        raw = '_RAW' if s == profile_noqc else ''
        # cycle through the variables identified in the file, for XBT files, this should only be TEMP:
        data_vars = temp_prof_info(s.netcdf_file_obj)
        # if the data_vars is not 'TEMP' but contains 'TEMP', change to 'TEMP'
        for key, value in data_vars.items():
            if 'TEMP' in value:
                data_vars[key] = 'TEMP'
        # assign the data_vars to the profile object
        s.prof_type = list(data_vars.values())

        if len(data_vars) > 1:
            LOGGER.error('Profile contains %s variables and is not an XBT %s' % (data_vars, s.Input_filename))
            continue
        # should only be one variable, TEMP, but leave as a loop for future proofing
        for ivar, var in data_vars.items():
            # we want the DEPTH to be a single dataset, but read all depths for each variable
            if 'P' in decode_bytearray(s.netcdf_file_obj.variables['D_P_Code'][ivar]):
                LOGGER.error('Pressure data found in %s. This is not a valid XBT file' % s.Input_filename)
                continue
            dep = np.round(s.netcdf_file_obj.variables['Depthpress'][ivar, :], 4)
            # if there are any depths that are less than 0, set them to NaN
            dep[dep < 0] = np.nan
            # eliminate nan depths if there are any
            dep = np.ma.masked_invalid(dep)
            # resize the arrays to eliminate empty values
            dep = np.ma.masked_array(dep.compressed())

            # if the size of the depth array is not the same as ndeps, pad the array or pad the dataframe
            if len(dep) != ndeps:
                if len(dep) < ndeps:
                    LOGGER.warning(
                        'Resizing %s and %s arrays to the number of depths recorded in MQNC file. %s' % (var, var, s.Input_filename))
                    # Create a new array of the desired size filled with NaN
                    resized_dep = np.full(ndeps, np.nan)
                    resized_dep[:len(dep)] = dep
                    dep = resized_dep
                else:
                    # dep is bigger than the number of depths, so resize the profile_qc.data dataframe by adding new rows
                    profile_qc.data = profile_qc.data.reindex(range(len(dep)))
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
            # mask any nan values from the profile
            prof = np.ma.masked_invalid(prof)
            # special case where values == 99.99 or a similar value are invalid where depth is > 4m
            if 'TEMP' in var and (abs(prof) > 90).any():
                # first change any values that might be 99.99 or 99999 or -99.99 or -99999  or similar to 99.99
                # do this by using (abs(prof) > 90)
                ind = np.where(abs(prof) > 90)[0]
                if len(ind) > 0:
                    prof[ind] = 99.99
                # replace values == 99.99 with NaN where they occur after 4 m depth
                # idepth = np.where(dep < 4.0)[0]
                # if len(idepth) > 0:
                #     # if there are depths less than 4m, then replace the values after the last depth less than 4m
                #     prof[idepth[-1]+1:] = np.where(np.isclose(abs(prof[idepth[-1]+1:]), 99.99,rtol=0.01), np.nan, prof[idepth[-1]+1:])
                prof = np.ma.masked_invalid(prof)
            # resize the arrays to eliminate empty values
            prof = np.ma.masked_array(prof.compressed())
            # Is there a mismatch in DEPTH and TEMP lengths?
            if ndeps != len(prof):
                if len(prof) < ndeps:
                    LOGGER.warning(
                        'Resizing %s and %s arrays to the number of depths recorded in MQNC file. %s' % (var, var, s.Input_filename))
                    # Create a new array of the desired size filled with NaN
                    resized_prof = np.full(ndeps, np.nan)
                    resized_prof[:len(prof)] = prof
                    prof = resized_prof
                else:
                    # prof is bigger than the number of depths, but not creating new depths, so resize prof to the number of depths
                    prof = prof[:ndeps]

            # if the size of the variable isn't equal to the number of depths, exit
            if (len(prof) != ndeps):
                LOGGER.error('Profile %s has %s depths but %s values for %s' % (s.Input_filename, ndeps, len(prof), var))
                exit(1)

            prof_flag = s.netcdf_file_obj.variables['ProfQP'][ivar, 0, :, 0, 0].flatten()
            # resize the arrays to eliminate empty values
            prof_flag = np.ma.masked_array(prof_flag.compressed())
            prof_flag = np.ma.masked_array(
                invalid_to_ma_array(prof_flag, fillvalue=99))  # replace masked values for IMOS IODE flags

            # if the size of the array isn't equal to the number of depths, adjust here
            if len(prof_flag) != ndeps:
                if len(prof_flag) < ndeps:
                    LOGGER.warning(
                        'Resizing %s and %s arrays to the number of depths recorded in MQNC file. %s' % (var, var, s.Input_filename))
                    # Create a new array of the desired size filled with NaN
                    resized_prof_flag = np.full(ndeps, np.nan)
                    resized_prof_flag[:len(prof_flag)] = prof_flag
                    prof_flag = resized_prof_flag
                else:
                    # prof_flag is bigger than the number of depths, so resize the qc
                    prof_flag = prof_flag[:ndeps]

            profile_qc.data['DEPTH' + raw] = dep.astype('float32')
            profile_qc.data['DEPTH' + raw +'_quality_control'] = pd.to_numeric(qc, errors='coerce')
            profile_qc.data[var + raw] = prof.astype('float32')
            profile_qc.data[var + raw + '_quality_control'] = pd.to_numeric(prof_flag, errors='coerce')

    # if DEPTH and DEPTH_RAW are not the same, apply fixes
    if not np.array_equal(profile_qc.data['DEPTH'], profile_qc.data['DEPTH_RAW']):
        # check the depth columns for consistency and match the variables based on DEPTH and DEPTH_RAW matches
        raw_columns = [col for col in profile_qc.data.columns if 'RAW' in col]
        df_raw = profile_qc.data[raw_columns]
        df_qc = profile_qc.data[[col for col in profile_qc.data.columns if col not in raw_columns]]
        # remove any nan rows from the dataframes
        df_raw = df_raw.dropna(subset=['DEPTH_RAW', 'TEMP_RAW'], how='all')
        df_qc = df_qc.dropna(subset=['DEPTH', 'TEMP'], how='all')

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
                    LOGGER.warning('Duplicated DEPTH_RAW found in %s' % profile_qc.Input_filename)
                    # drop the duplicates
                    df_raw = df_raw.drop_duplicates(subset='DEPTH_RAW').reset_index(drop=True)
                if df_qc['DEPTH'].duplicated().any():
                    LOGGER.warning('Duplicated DEPTH found in %s' % profile_qc.Input_filename)
                    # drop the duplicates
                    df_qc = df_qc.drop_duplicates(subset='DEPTH').reset_index(drop=True)
                df = pd.concat([df_qc, df_raw], axis=1)
                # check the lengths again
                if len(df_raw) != len(df_qc):
                    LOGGER.warning('DEPTH_RAW and DEPTH counts are significantly different. Please review %s' % profile_qc.Input_filename)
                    # concatenate the two dataframes with NaNs in the rows that don't match
                    df = pd.concat([df_qc, df_raw], axis=1)
        else:
            # simplest case where the lengths are the same but actual values might be different
            # concatenate the two dataframes
            df = pd.concat([df_qc, df_raw], axis=1)

        # check that the merge has worked
        if len(df) != max(len(df_raw), len(df_qc)):
            LOGGER.error('Dataframes have not been merged correctly. Please review %s' % profile_qc.Input_filename)
            exit(1)

        # check here that the DEPTH and DEPTH_RAW columns are the same or DEPTH_RAW is 1.0336 * DEPTH
        if not np.isclose(df['DEPTH_RAW'].values, df['DEPTH'].values, atol=1e-6).all() and \
                not np.isclose(df['DEPTH_RAW'].values * 1.0336, df['DEPTH'].values, atol=1e-6).all():
            LOGGER.error('DEPTH_RAW and DEPTH values do not match in %s' % profile_qc.Input_filename)

        # save the dataframe of DEPTH dimensioned data to the profile object
        profile_qc.data = df

    # check for duplicated depths and log if found
    if profile_qc.data['DEPTH'].duplicated().any() or profile_qc.data['DEPTH_RAW'].duplicated().any():
        LOGGER.error('Duplicated DEPTH or DEPTH_RAW found in %s' % profile_qc.Input_filename)

    # assign the station number to the profile_qc.data
    profile_qc.data['station_number'] = station_number

    # Location information
    lat = profile_qc.netcdf_file_obj['latitude'][0].__float__()
    lat_raw = profile_noqc.netcdf_file_obj['latitude'][0].__float__()
    lon = profile_qc.netcdf_file_obj['longitude'][0].__float__()
    lon_raw = profile_noqc.netcdf_file_obj['longitude'][0].__float__()
    # check if scale factor has been applied, shouldn't have a negative longitude:
    if lon < 0:
        if profile_qc.netcdf_file_obj['longitude'].scale:
            LOGGER.info('Scale Factor in ed file longitude attributes, changing longitude value from  %s %s' %
                        (lon, profile_qc.Input_filename))
            lon = lon * -1
        else:
            LOGGER.error('Negative LONGITUDE value with no Scale Factor %s %s' % (lon, profile_qc.Input_filename))

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
    q_pos = decode_bytearray(profile_qc.netcdf_file_obj['Q_Pos'][:])
    # remove control characters from the q_pos
    q_pos = np.ma.array(remove_control_chars(q_pos), mask=0)
    # convert to int and replace empty strings with '0' assuming that if the value is empty, it is not set
    q_pos = np.ma.array(np.where(q_pos == '', '1', q_pos), mask=q_pos.mask).astype(int)

    profile_qc.data['LATITUDE_quality_control'] = q_pos
    profile_qc.data['LONGITUDE_quality_control'] = q_pos

    # Date time information
    woce_date = profile_qc.netcdf_file_obj['woce_date'][0]
    woce_time = profile_qc.netcdf_file_obj['woce_time'][0]

    # AW Add Original date_time from the raw .nc - date-time could be changed thru QC
    woce_date_raw = profile_noqc.netcdf_file_obj['woce_date'][0]
    woce_time_raw = profile_noqc.netcdf_file_obj['woce_time'][0]

    q_date_time = decode_bytearray(profile_qc.netcdf_file_obj['Q_Date_Time'][:])
    # remove control characters from the q_date_time
    q_date_time = np.ma.array(remove_control_chars(q_date_time), mask=0)
    # convert to int and replace empty strings with '0' assuming that if the value is empty, it is not set
    q_date_time = np.ma.array(np.where(q_date_time == '', '1', q_date_time), mask=q_date_time.mask).astype(int)

    # need to be a bit more specific as some times have missing padding at the end, some at the start.
    # could break if hour is 00 and there are no zeros!
    # Let's try padding left and right, then convert to time for both
    rpad = str(woce_time).ljust(6, '0')
    lpad = str(woce_time).zfill(6)
    # get the right date format using convert_time_string
    xbt_time = convert_time_string(rpad, '%H%M%S')
    xbt_time2 = convert_time_string(lpad, '%H%M%S')
    if not pd.isnull(xbt_time2) and pd.isnull(xbt_time):
        xbt_time = lpad
    elif pd.isnull(xbt_time2) and not pd.isnull(xbt_time):
        xbt_time = rpad
    elif pd.isnull(xbt_time) and pd.isnull(xbt_time2):
        LOGGER.error('Could not convert time %s to a valid time format, using 000000' % (woce_time))
        # if int(rpad) > 240000, assume time is at 00:00:00
        if int(lpad) >= 240000:
            xbt_time ='000000'
    else:
        xbt_time = lpad

    xbt_date = '%s%s' % (woce_date, xbt_time)
    xbt_date = convert_time_string(xbt_date,'%Y%m%d%H%M%S')

    # Raw date
    rpad = str(woce_time_raw).ljust(6, '0')
    lpad = str(woce_time_raw).zfill(6)

    # get the right date format using convert_time_string
    xbt_time_raw = convert_time_string(rpad, '%H%M%S')
    xbt_time_raw2 = convert_time_string(lpad, '%H%M%S')
    if not pd.isnull(xbt_time_raw2) and pd.isnull(xbt_time_raw):
        xbt_time_raw = lpad
    elif pd.isnull(xbt_time_raw2) and not pd.isnull(xbt_time_raw):
        xbt_time = rpad
    elif pd.isnull(xbt_time_raw) and pd.isnull(xbt_time_raw2):
        LOGGER.error('Could not convert time_raw %s to a valid time format, using 000000' % (woce_time))
        # if int(rpad) > 240000, assume time is at 00:00:00
        if int(lpad) >= 240000:
            xbt_time_raw ='000000'
    else:
        xbt_time_raw = lpad
    xbt_date_raw = '%s%s' % (woce_date_raw, xbt_time_raw)
    xbt_date_raw = convert_time_string(xbt_date_raw,'%Y%m%d%H%M%S')

    # AW - TIME_RAW is original date-time - set it too
    profile_qc.data['TIME'] = xbt_date
    profile_qc.data['TIME_quality_control'] = q_date_time
    profile_qc.data['TIME_RAW'] = xbt_date_raw

    # drop rows where all NaN values which does happen in these old files sometimes
    profile_qc.data = profile_qc.data.dropna(subset=['TEMP', 'DEPTH', 'TEMP_RAW', 'DEPTH_RAW'], how='all')

    # sort the data by DEPTH and reset the index
    profile_qc.data = profile_qc.data.sort_values(by='DEPTH').reset_index(drop=True)

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
    if len(profile.histories[profile.histories['HISTORY_QC_CODE'].str.contains("LA|LO|PE|TP")]) == 0:
        return profile

    # get the temperature QC codes
    df = profile.data
    if profile.histories['HISTORY_QC_CODE'].str.contains('LAA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LATITUDE_RAW value
        if not np.isclose(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LAA'), 'HISTORY_PREVIOUS_VALUE'].values[0]),
                      profile.data['LATITUDE_RAW'][0], atol=1e-6):
            LOGGER.error('LATITUDE_RAW not the same as the PREVIOUS_value! %s' % profile.Input_filename)
        if profile.data['LATITUDE_quality_control'].unique() != 5:
            # PEA on latitude
            profile.data['LATITUDE_quality_control'] = 5
            LOGGER.info('LATITUDE correction (PEA) in original file, changing LATITUDE flag to level 5. %s'
                        % profile.Input_filename)
        # if there is no LOA code, make sure the LONGITUDE quality control is set to 1 as QPOS is applied to both lat/long
        if not profile.histories['HISTORY_QC_CODE'].str.contains('LOA').any():
            profile.data['LONGITUDE_quality_control'] = 1

        # change to flag 2 for temperature for all depths where qc is less than 2
        mask = df['TEMP_quality_control'] < 2
        df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('LOA').any():
        # check HISTORY_PREVIOUS_VALUE matches the LONGITUDE_RAW value within a tolerance
        if not np.isclose(float(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'LOA'), 'HISTORY_PREVIOUS_VALUE'].values[0]),
                      profile.data['LONGITUDE_RAW'][0], atol=1e-6):
            LOGGER.error('LONGITUDE_RAW not the same as the PREVIOUS_value! %s' % profile.Input_filename)
        if profile.data['LONGITUDE_quality_control'].unique() != 5:
            # PEA on longitude
            profile.data['LONGITUDE_quality_control'] = 5
            LOGGER.info('LONGITUDE correction (PEA) in original file, changing LONGITUDE flag to level 5. %s'
                        % profile.Input_filename)
        # if there is no LAA code, make sure the LATITUDE quality control is set to 1 as QPOS is applied to both lat/long
        if not profile.histories['HISTORY_QC_CODE'].str.contains('LAA').any():
            profile.data['LATITUDE_quality_control'] = 1

        # change to flag 2 for temperature for all depths where qc is less than 2
        mask = df['TEMP_quality_control'] < 2
        df.loc[mask, 'TEMP_quality_control'] = 2

    if profile.histories['HISTORY_QC_CODE'].str.contains('PER').any():
        # PER on longitude and latitude
        profile.data['LONGITUDE_quality_control'] = 3
        profile.data['LATITUDE_quality_control'] = 3
        LOGGER.info('Position Reject (PER) in original file, changing LONGITUDE & LATITUDE flags to level 3.%s'
                    % profile.Input_filename)
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
    if profile.histories['HISTORY_QC_CODE'].str.contains('TEA').any() and profile.data['TIME_quality_control'][0] != 5:
        # TEA
        profile.data['TIME_quality_control'] = 5
        LOGGER.info('TIME correction (TEA) in original file, changing TIME flag to level 5.%s'
                    % profile.Input_filename)
        # change to flag 2 for temperature for all depths where qc is less than 2

        profile.data.loc[profile.data['TEMP_quality_control'] < 2, 'TEMP_quality_control'] = 2
        # check HISTORY_PREVIOUS_VALUE matches the TIME_RAW value
        if convert_time_string(profile.histories.loc[
                              profile.histories['HISTORY_QC_CODE'].str.contains(
                                  'TEA'), 'HISTORY_PREVIOUS_VALUE'].values[0], '%Y%m%d%H%M%S', 'datetime') != \
                profile.data['TIME_RAW'][0]:
            LOGGER.error('TIME_RAW not the same as the PREVIOUS_VALUE! %s'
                         % profile.Input_filename)

    return profile


def get_fallrate_eq_coef(profile_qc, profile_noqc):
    """return probe type name, coef_a, coef_b as defined in WMO1770"""
    fre_list = read_section_from_xbt_config('FRE')
    peq_list = read_section_from_xbt_config('PEQ$')
    ptyp_list = read_section_from_xbt_config('PTYP')

    nms = [profile_qc, profile_noqc]
    vv = ['', '_RAW']

    for ind in range(vv.__len__()):
        item_val = profile_qc.data['PROBE_TYPE' + vv[ind]].unique().item()
        if item_val == '':
            profile_qc.data['PROBE_TYPE' + vv[ind]] = '1023'
            profile_qc.data['PROBE_TYPE_name' + vv[ind]]  = 'Unknown'
            profile_qc.data['PROBE_TYPE_coefficient_a' + vv[ind]] = np.nan
            profile_qc.data['PROBE_TYPE_coefficient_b' + vv[ind]] = np.nan
            if ind == 0:
                profile_qc.data['PROBE_TYPE_quality_control'] = 0
            LOGGER.error('PROBE_TYPE is missing from %s' % profile_qc.Input_filename)
            return profile_qc
        # if histories is not empty, check for TPR code
        if not profile_qc.histories.empty and \
                profile_qc.histories['HISTORY_QC_CODE'].str.contains('TPR').any():
            # assign a test probe type
            item_val = '104'
        if item_val in list(ptyp_list.keys()) and item_val not in list(fre_list.keys()):
            # old PTYP surface code, need to match up PEQ$code
            item_val = ptyp_list[item_val]
        # is it in the PEQ list
        elif item_val in list(peq_list.keys()) and item_val not in list(fre_list.keys()):
            LOGGER.warning('PROBE_TYPE %s is not an XBT type, not converted' % item_val)
            profile_qc.data['PROBE_TYPE'] = ''
            # this is not an XBT
            return profile_qc
        elif item_val in list(peq_list.keys()):
            # record the original value
            profile_qc.data['PROBE_TYPE_original_name'] = item_val
        else:
            # try fuzzy matching here
            imatch = difflib.get_close_matches(item_val[0:4], list(ptyp_list.keys()), n=1, cutoff=0.5)
            if imatch:
                LOGGER.warning('PROBE_TYPE %s not found in WMO1770, using closest match %s %s'
                               % (item_val, imatch[0], profile_qc.Input_filename))
                item_val = ptyp_list[imatch[0]]

        # use the code we have extracted to get the fall rate equation and name of probe
        if item_val in list(fre_list.keys()):
            probetype = peq_list[item_val]
            coef_a = float(fre_list[item_val].split(',')[0])
            coef_b = float(fre_list[item_val].split(',')[1]) * 0.001

            profile_qc.data['PROBE_TYPE' + vv[ind]] = item_val
            profile_qc.data['PROBE_TYPE_name' + vv[ind]] = probetype
            profile_qc.data['PROBE_TYPE_coefficient_a' + vv[ind]] = coef_a
            profile_qc.data['PROBE_TYPE_coefficient_b' + vv[ind]] = coef_b
            if ind == 0:
                profile_qc.data['PROBE_TYPE_quality_control'] = 1
        else:
            profile_qc.data['PROBE_TYPE' + vv[ind]] = '1023'
            profile_qc.data['PROBE_TYPE_name' + vv[ind]]  = 'Unknown'
            profile_qc.data['PROBE_TYPE_coefficient_a' + vv[ind]] = np.nan
            profile_qc.data['PROBE_TYPE_coefficient_b' + vv[ind]] = np.nan
            if ind == 0:
                profile_qc.data['PROBE_TYPE_quality_control'] = 0
            LOGGER.error('PROBE_TYPE, PROBE_TYPE fallrate equation missing from %s' % profile_qc.Input_filename)

    # select a QC flag for the probe type
    # TODO: if the probe types are different in raw and edited, need to handle this.
    #  Has it been changed? what does the data look like? Need to assign 5 to changed profile, include the PR flag
    #  and adjust the QC on the temperature and depth
    if profile_qc.data['PROBE_TYPE'].unique() != profile_qc.data['PROBE_TYPE_RAW'].unique():
        LOGGER.error('PROBE_TYPE are different in ed and raw files. %s' % profile_qc.Input_filename)

    return profile_qc


def parse_histories_nc(profile):
    """ Parse the history records in Mquest files
    """
    # let's use a pandas dataframe with empty columns
    df = pd.DataFrame()
    nhist = int(profile.netcdf_file_obj['Num_Hists'][0].data)
    # set code_strings to a list of indices from 0 to nhist
    code_strings = list(range(nhist))

    # for each column, extract the data
    # list the data labels matching the columns in the dataframe
    varname = ['Act_Code', 'Ident_Code', 'Act_Parm', 'PRC_Code', 'PRC_Date', 'Aux_ID', 'Flag_severity', 'Version',
                'Previous_Val']
    for var in varname:
        if var not in profile.netcdf_file_obj.variables:
            LOGGER.warning('Variable %s not found in %s' % (var, profile.Input_filename))
            df[var] = np.nan
            continue
        # test if the data is a byte array or a float
        if np.issubdtype(profile.netcdf_file_obj[var].dtype, np.number):
            vv = profile.netcdf_file_obj[var][:].data
            # use code_strings to resize vv
            vv = [vv[i] for i in code_strings]
        else:
            # if this is the Act_Code, check if the nhist is correct
            if var == 'Act_Code':
                # first get all the history codes including empty ones
                vv = [''.join(chr(x) for x in bytearray(xx))
                      for xx in profile.netcdf_file_obj[var][:].data if bytearray(xx)]
                vv = [remove_control_chars(str(x)) for x in vv]
                # identify the locations of the empty stings in vv
                code_strings = [i for i, x in enumerate(vv) if x.strip() != '']
                # use code_strings to set nhist and resize vv
                if code_strings:
                    vv = [vv[i] for i in code_strings]
                    nhist = len(vv)
            else:
                # convert the byte array to a string
                vv = [''.join(chr(x) for x in bytearray(xx))
                      for xx in profile.netcdf_file_obj[var][:].data if bytearray(xx)]
                vv = [remove_control_chars(str(x)) for x in vv]
                # use code_strings to resize vv
                vv = [vv[i] for i in code_strings]
        df[var] = vv
    # rename the columns
    df.columns = ['HISTORY_QC_CODE', 'HISTORY_INSTITUTION', 'HISTORY_PARAMETER', 'HISTORY_SOFTWARE',
                               'HISTORY_DATE', 'HISTORY_START_DEPTH', 'HISTORY_QC_CODE_VALUE',
                               'HISTORY_SOFTWARE_RELEASE', 'HISTORY_PREVIOUS_VALUE']

    # change HISTORY_START_DEPTH and HISTORY_PREVIOUS_VALUE to float64
    df['HISTORY_START_DEPTH'] = df['HISTORY_START_DEPTH'].astype('float32')
    df['HISTORY_PREVIOUS_VALUE'] = df['HISTORY_PREVIOUS_VALUE'].astype('string')
    # change HISTORY_QC_CODE_VALUE to int32
    df['HISTORY_QC_CODE_VALUE'] = df['HISTORY_QC_CODE_VALUE'].astype('int8')

    if nhist == 0:
        # no history records
        profile.histories = df
        return profile

    # check that the history codes exist in our list
    # read the set list of codes from the csv files
    qc_dfa, qc_dfr = read_flag_quality_table(True)
    # combine the two dataframes
    qc_df = pd.concat([qc_dfa, qc_dfr], ignore_index=True)

    # create list of acceptable parameter names
    parm_names = {'DEPH': 'DEPTH', 'DATI': 'DATE, TIME', 'DATE': 'DATE', 'TIME': 'TIME', 'LATI': 'LATITUDE',
             'LONG': 'LONGITUDE', 'LALO': 'LATITUDE, LONGITUDE', 'TEMP': 'TEMP'}
    # check that the history codes are in the list
    if not df['HISTORY_QC_CODE'].isin(qc_df['code']).all():
        missing = df.loc[~df['HISTORY_QC_CODE'].isin(qc_df['code']), 'HISTORY_QC_CODE']
        LOGGER.warning('HISTORY_QC_CODE values %s not found in the QC code list. Please review output for this file %s'
                       % (missing.values, profile.Input_filename))
        # remove any codes that are not in the list and where PARAMETER is not in names list
        df = df.loc[df['HISTORY_QC_CODE'].isin(qc_df['code']) & df['HISTORY_PARAMETER'].isin(parm_names.keys())]
        # reset nhist to the new length
        nhist = len(df)
    # Check for errors in the HISTORY_PARAMETER where the HISTORY_QC_CODE does not match the parameter in qc_df
    # match the HISTORY_QC_CODE with qc_df['code'] and check the parameter is correct
    for idx, row in df.iterrows():
        # get the code and parameter
        code = row['HISTORY_QC_CODE']
        parm = row['HISTORY_PARAMETER']
        # does row['HISTORY_QC_CODE'] match the qc_df['code']?
        if parm != qc_df.loc[qc_df['code'] == code, 'Parameter'].values[0] and code not in ['PE', 'TE']:
            LOGGER.warning('HISTORY_QC_CODE %s has incorrect HISTORY_PARAMETER %s and has been corrected. Please review output for this file. %s'
                           % (code, parm, profile.Input_filename))
            # if the qc_df['parameter'] is not 'PE' OR 'TE', then set the HISTORY_PARAMETER to the qc_df['parameter']
            df.at[idx, 'HISTORY_PARAMETER'] = qc_df.loc[qc_df['code'] == code, 'Parameter'].values[0]

    # allow for history dates to be YYYYMMDD or DDMMYYYY
    date1 = convert_time_string(df['HISTORY_DATE'], '%Y%m%d')
    date2 = convert_time_string(df['HISTORY_DATE'],'%d%m%Y')
    df['HISTORY_DATE'] = date1.fillna(date2)

    # append the 'A' or 'R' to each code
    for idx, row in df.iterrows():
        if df.at[idx, 'HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5]:
            df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'A'
        else:
            df.at[idx, 'HISTORY_QC_CODE'] = row['HISTORY_QC_CODE'] + 'R'

    # this group of changes is here because I have reviewed all our QC codes in the historic databases and I know
    # there are some that are not correct.
    # change ERA to PLA with flag 3 to reduce duplication of flags
    df.loc[
        (df['HISTORY_QC_CODE'].str.contains('ERA')), ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'PLA', np.int8(3)
    # change URA for BDA and flag 2
    df.loc[
        (df['HISTORY_QC_CODE'].str.contains('URA')), ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'BDA', np.int8(2)

    # change CSA to CSR and the flag to 3 to match new format
    df.loc[(df['HISTORY_QC_CODE'].str.contains('CSA')),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'CSR', np.int8(3)

    # Change the PEA flag to LA or LO and ensure the HISTORY_QC_CODE_VALUE is set to 5
    df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
            (df['HISTORY_PARAMETER'].str.contains('LATI'))),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'LAA', np.int8(5)
    df.loc[((df['HISTORY_QC_CODE'].str.contains('PEA')) &
            (df['HISTORY_PARAMETER'].str.contains('LONG'))),
    ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE']] = 'LOA', np.int8(5)

    # set the software value to 2.1 for CS and PE, RE flags
    df.loc[
        df.HISTORY_QC_CODE.isin(['CSR', 'LAA', 'LOA', 'PER', 'REA']), ['HISTORY_SOFTWARE_RELEASE', 'HISTORY_SOFTWARE']] = '2.1', 'CSCBv2'

    # update software names to be more descriptive
    names = {'CSCB': 'CSIRO Quality control cookbook for XBT data v1.1',
             'CSCBv2': 'Australian XBT Quality Control Cookbook Version 2.1'}
    df['HISTORY_SOFTWARE'] = df['HISTORY_SOFTWARE'].map(names, na_action='ignore')

    # Combine duplicated TEA flags to a single TEA for TIME variable TEMP_QC_CODE_VALUE is set to 2, not 5
    # Also change just DATE TEA flags to TIME
    dfTEA = df[df['HISTORY_QC_CODE'] == 'TEA'].copy()
    if len(dfTEA) > 0:
        # check that the history parameter is TIME or DATE
        if not dfTEA['HISTORY_PARAMETER'].isin(['TIME', 'DATE']).all():
            LOGGER.warning('HISTORY_PARAMETER for TEA flags incorrect and has been reset to TIME or DATE. Please review %s' % profile.Input_filename)
            # test the HISTORY_PREVIOUS_VALUE by converting to time string
            for idx, row in dfTEA.iterrows():
                dati = convert_time_string(row['HISTORY_PREVIOUS_VALUE'], '%Y%m%d', 'string')
                if dati is None:
                    # assume this is a TIME variable and set the HISTORY_PARAMETER to TIME
                    dfTEA.at[idx, 'HISTORY_PARAMETER'] = 'TIME'
                else:
                    # assume this is a DATE variable and set the HISTORY_PARAMETER to DATE
                    dfTEA.at[idx, 'HISTORY_PARAMETER'] = 'DATE'

        # first tidy up the TIME variable and DATE variable
        # if any of timerows['HISTORY_PREVIOUS_VALUE'] contains a variation with 9's then set to 0
        pattern = re.compile(r'^9{1,5}(?:\.\d+)?$')
        timeidx = df['HISTORY_PARAMETER'] == 'TIME'
        if timeidx.any():
            if df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'].str.contains(pattern).any():
                df.loc[timeidx, 'HISTORY_PREVIOUS_VALUE'] = '000000'
            # remove any colons in the HISTORY_PREVIOUS_VALUE
            dfTEA['HISTORY_PREVIOUS_VALUE'] = dfTEA['HISTORY_PREVIOUS_VALUE'].str.replace(':', '', regex=False)
            # use zfill to ensure the HISTORY_PARAMETER=='TIME' is 6 digits long
            dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'] = \
                dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'].str.zfill(6)

        # tidy the DATE variable
        dateidx = df['HISTORY_PARAMETER'] == 'DATE'
        if dateidx.any():
            # if any of date rows['HISTORY_PREVIOUS_VALUE'] contains a variation with 9's then set to 0
            if df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'].astype(str).str.contains(pattern).any():
                df.loc[dateidx, 'HISTORY_PREVIOUS_VALUE'] = '00000000'
            else:
                # allow for dates to be YYYYMMDD or DDMMYYYY
                date1 = convert_time_string(
                    dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'].values[0],
                    '%Y%m%d', 'string')
                date2 = convert_time_string(
                    dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'].values[0],
                    '%d%m%Y', 'string')
                # use date1 if it is not None, otherwise use date2
                if date1 is not None:
                    df.loc[df['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = date1
                    dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = date1
                elif date2 is not None:
                    df.loc[df['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = date2
                    dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = date2
                else:
                    LOGGER.error('DATE format not recognised in %s. Please review the file.' % profile.Input_filename)
                    df.loc[df['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = '00000000'
                    dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = '00000000'
        # test here for both TIME and DATE in the TEA flags
        if any(dfTEA['HISTORY_PARAMETER'].str.contains('TIME')) & any(dfTEA['HISTORY_PARAMETER'].str.contains('DATE')):
            # if there are more than two rows, break with error for now
            if len(dfTEA) > 2:
                LOGGER.error('Multiple TEA flags found for TIME and DATE in %s. Please review the file.' % profile.Input_filename)
                # exit(1)
            # if both TIME and DATE are present, combine them into a single row for TIME
            # combine the HISTORY_PREVIOUS_VALUE TIME with DATE into a single string called dati
            dati = dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'].values[0] + \
                   dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'].values[0]
            # convert the dati to a date string in the format YYYYMMDDHHMMSS
            dati = convert_time_string(dati, '%Y%m%d%H%M%S', 'string')
            # if dati is None, put '000000' in the TIME row
            if dati is None:
                dati = '00000000000000'
            # replace the TIME row
            # replace the TIME and DATE rows with a single row for TIME
            df = df.loc[df['HISTORY_PARAMETER'] != 'DATE']
            df.loc[df['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'] = dati
            # reset index
            df = df.reset_index(drop=True)
            # change the nhist to the new length
            nhist = len(df)
        else:
            # get the date value from the TIME variable
            dtt = profile.data['TIME'].dt.strftime('%Y%m%d')[0]
            # get the TIME value from the TIME variable
            ti = profile.data['TIME'].dt.strftime('%H%M%S')[0]

            # if there is a TIME variable, add the dtt to the HISTORY_PREVIOUS_VALUE and convert to datetime
            if any(dfTEA['HISTORY_PARAMETER'] == 'TIME'):
                # get the TIME value from the TIME variable
                ti = dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'].values[0]
                ti = convert_time_string(ti, '%H%M%S', 'string')
                # combine the dtt and ti into a single string called dati
                dati = str(dtt) + str(ti)
                # convert the dati to a date string in the format YYYYMMDDHHMMSS
                dati = convert_time_string(dati, '%Y%m%d%H%M%S', 'string')
                # if dati is None, put '000000' in the TIME row
                if dati is None:
                    dati = '000000'
                # replace the TIME row with the new datetime value
                df.loc[df['HISTORY_PARAMETER'] == 'TIME', 'HISTORY_PREVIOUS_VALUE'] = dati

            # now check for any 'DATE' parameter in the TEA flags
            if any(dfTEA['HISTORY_PARAMETER'] == 'DATE'):
                # get the DATE value from the DATE variable
                dt = dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'].values[0]
                dt = convert_time_string(dt, '%Y%m%d', 'string')
                # if the date is not in the correct format, try DDMMYYYY
                if dt is None:
                    dt = dfTEA.loc[dfTEA['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'].values[0]
                    dt = convert_time_string(dt, '%d%m%Y', 'string')
                # combine the dtt and dt into a single string called dati
                dati = str(dt) + str(ti)
                # convert the dati to a date string in the format YYYYMMDDHHMMSS
                dati = convert_time_string(dati, '%Y%m%d%H%M%S', 'string')
                # if dati is None, put '00000000' in the DATE row
                if dati is None:
                    dati = '00000000'
                # replace the DATE row with the new datetime value
                df.loc[df['HISTORY_PARAMETER'] == 'DATE', 'HISTORY_PREVIOUS_VALUE'] = dati

            # change the 'DATE' label to TIME  and update the TEA PREVIOUS_VALUE to the new datetime value
            df.loc[((df['HISTORY_PARAMETER'].str.contains('DATE') | df['HISTORY_PARAMETER'].str.contains('TIME')) &
                    (df['HISTORY_QC_CODE'].str.contains('TEA'))), ['HISTORY_PARAMETER']] = 'TIME'

    # update institute names to be more descriptive
    names = read_section_from_xbt_config('INSTITUTE')
    df['HISTORY_INSTITUTION'] = df['HISTORY_INSTITUTION'].map(lambda x: names[x].split(',')[0] if x in names else x)
    if any(df['HISTORY_INSTITUTION'].isna()):
        # list the institutes that are not defined
        missing = df.loc[df['HISTORY_INSTITUTION'].isna(), 'HISTORY_INSTITUTION']
        LOGGER.warning("HISTORY_INSTITUTION values %s are not defined. Please review output for this file %s"
                       % (missing, profile.Input_filename))

    # get a list of qc_df['code'] values where qc_df['code'] only appears once in the dataframe
    # Get the value counts of 'code'
    code_short_counts = qc_df['code'].value_counts()
    # Filter 'qc_df' to get rows where 'code' appears only once
    single_code_short_df = qc_df[qc_df['code'].isin(code_short_counts[code_short_counts == 1].index)]

    # if any of the single_qc_codes are in the HISTORY_QC_CODE, change the HISTORY_QC_CODE_VALUE to match the single_qc_code_short_df['tempqc'] value
    for idx, row in single_code_short_df.iterrows():
        mask = df['HISTORY_QC_CODE'].str[:2] == row['code']
        if any(mask):
            df.loc[mask, ['HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE', 'HISTORY_PARAMETER']] = [row['full_code'],
                                                                                               row['TEMP_quality_control'],
                                                                                               row['Parameter']]
    # add the QC description information
    df["HISTORY_QC_CODE_DESCRIPTION"] = [''] * nhist
    # map the qc_df['code'] to the df['HISTORY_QC_CODE'] and add the description to the df['HISTORY_QC_CODE_DESCRIPTION']

    # Create a dictionary from qc_df for mapping
    qc_code_to_description = qc_df.set_index('full_code')['name'].to_dict()

    # Map the 'HISTORY_QC_CODE' to the descriptions and add to 'HISTORY_QC_CODE_DESCRIPTION'
    df['HISTORY_QC_CODE_DESCRIPTION'] = df['HISTORY_QC_CODE'].map(qc_code_to_description)

    if any(df['HISTORY_QC_CODE_DESCRIPTION'].eq('')) or any(df['HISTORY_QC_CODE_DESCRIPTION'].isna()):
        missing = df.loc[df['HISTORY_QC_CODE_DESCRIPTION'] == '', 'HISTORY_QC_CODE']
        if missing.any():
            LOGGER.warning("HISTORY_QC_CODE \"%s\" is not defined. Please edit xbt_config file. %s"
                           % (missing, profile.Input_filename))

    # remove any duplicated lines for any code
    df = df[~(df.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_PREVIOUS_VALUE', 'HISTORY_START_DEPTH']))]
    # find duplicated codes where one previous value is nan and parameter is TEMP. Remove the nan value
    mask = df.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_START_DEPTH'])
    mask2 = df['HISTORY_PARAMETER'].str.contains('TEMP')
    mask3 = df['HISTORY_PREVIOUS_VALUE'].isna()
    mask4 = mask & mask2 & mask3
    if any(mask4):
        print('Check this is working %s' % profile.Input_filename)
        exit(1)
        df = df[~mask4]

    # remove any rows where HISTORY_QC_CODE is 'WBR' and has a HISTORY_START_DEPTH of NAN
    df.dropna(subset=['HISTORY_START_DEPTH'], inplace=True)

    # change the start_depth values to the closest depth value in the profile data if profile.input_filename ends with ed.nc
    if profile.Input_filename.endswith('ed.nc'):
        # reset the index
        df = df.reset_index(drop=True)
        for idx, row in df.iterrows():
            # find the closest depth value in the profile data
            closest_depth = profile.data['DEPTH'].sub(row['HISTORY_START_DEPTH']).abs().idxmin()
            # if the closest depth is > 2m, log warning
            if abs(profile.data['DEPTH'][closest_depth] - row['HISTORY_START_DEPTH']) > 2:
                LOGGER.warning('HISTORY: Closest depth for HISTORY_START_DEPTH %s is %s, which is more than 2m away from the profile depth. %s'
                                % (row['HISTORY_START_DEPTH'], profile.data['DEPTH'][closest_depth],
                                    profile.Input_filename))
            # update the HISTORY_START_DEPTH value to the closest depth value
            df.at[idx, 'HISTORY_START_DEPTH'] = profile.data['DEPTH'].iloc[closest_depth]
            # check for values that cannot be converted to float in HISTORY_PREVIOUS_VALUE
            if row['HISTORY_PARAMETER'] == 'TEMP':
                try:
                    float(row['HISTORY_PREVIOUS_VALUE'])
                except ValueError:
                    LOGGER.error('HISTORY: HISTORY_PREVIOUS_VALUE %s cannot be converted to float at depth %s. %s'
                                 % (row['HISTORY_PREVIOUS_VALUE'], row['HISTORY_START_DEPTH'], profile.Input_filename))
                    # set the HISTORY_PREVIOUS_VALUE to '99.99' if it cannot be converted to float
                    df.at[idx, 'HISTORY_PREVIOUS_VALUE'] = '99.99'
                    # re-read the row
                    row = df.iloc[idx]
            # if there are any 9999 type values in the HISTORY_PREVIOUS_VALUE and the HISTORY_PARAMETER is TEMP,
            # change the HISTORY_PREVIOUS_VALUE to the TEMP_RAW value at the closest depth
            pattern = re.compile(r'^9{1,5}(?:\.\d+)?$')
            if (row['HISTORY_PARAMETER'] == 'TEMP' and pd.isna(row['HISTORY_PREVIOUS_VALUE'])) or (row['HISTORY_PARAMETER'] == 'TEMP' and \
                    re.match(pattern, row['HISTORY_PREVIOUS_VALUE'])) :
                # update the HISTORY_PREVIOUS_VALUE to the TEMP_RAW value at the closest depth
                df.at[idx, 'HISTORY_PREVIOUS_VALUE'] = \
                    profile.data['TEMP_RAW'].iloc[closest_depth].astype(str)
                # re-read the row
                row = df.iloc[idx]

            # if the HISTORY_PREVIOUS_VALUE does not match the TEMP_RAW value at the closest depth, update it if the depth is within 2m
            if row['HISTORY_PARAMETER'] == 'TEMP' and \
                    np.isclose(float(row['HISTORY_PREVIOUS_VALUE']),
                                   profile.data['TEMP_RAW'].iloc[closest_depth], atol=5e-1):
                # update the HISTORY_PREVIOUS_VALUE to the TEMP_RAW value at the closest depth
                df.at[idx, 'HISTORY_PREVIOUS_VALUE'] = \
                    profile.data['TEMP_RAW'].iloc[closest_depth].astype(str)
            elif row['HISTORY_PARAMETER'] == 'TEMP' and \
                    not np.isclose(float(row['HISTORY_PREVIOUS_VALUE']),
                                   profile.data['TEMP_RAW'].iloc[closest_depth], atol=5e-1):
                LOGGER.warning('HISTORY: HISTORY_PREVIOUS_VALUE %s does not match TEMP_RAW value %s at depth %s. %s'
                               % (row['HISTORY_PREVIOUS_VALUE'], profile.data['TEMP_RAW'].iloc[closest_depth],
                                  profile.data['DEPTH'].iloc[closest_depth], profile.Input_filename))


    # assign the dataframe back to profile at this stage
    profile.histories = df.reset_index(drop=True)

    return profile


def combine_histories(profile_qc, profile_noqc):
    # check for global attributes in the noqc file and update the global atts as required
    # handle the longitude change where data was imported from dataset with a negative longitude where it should
    # have been positive. The *raw.nc previous value and *ed.nc previous value should be the same, update the LONG_RAW.
    #first merge all the histories
    combined_histories = pd.merge(profile_qc.histories, profile_noqc.histories, how='left')
    # if the combined_histories is empty, return the profile_qc
    if combined_histories.empty:
        return profile_qc
    # check for TER where the date has been corrected and therefore should be a TEA, happens in some badly recorded flags in old files
    if any(combined_histories['HISTORY_QC_CODE'].str.contains('TER')):
        # does the HISTORY_PREVIOUS_VALUE match the TIME_RAW value?
        if not combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
            'HISTORY_PREVIOUS_VALUE'].values == profile_qc.data['TIME_RAW'][0].strftime('%Y%m%d%H%M%S'):
            # does the previous value contain 9's?
            if any(combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
                'HISTORY_PREVIOUS_VALUE'].astype(str).str.contains('0{1,5}')):
                LOGGER.warning('HISTORY: Previous value does not match TIME_RAW value. %s' % profile_qc.Input_filename)
                # use the TIME_RAW value and update previous value
                combined_histories.loc[combined_histories['HISTORY_QC_CODE'].str.contains('TER'),
                    'HISTORY_PREVIOUS_VALUE'] = profile_qc.data['TIME_RAW'][0].strftime('%Y%m%d%H%M%S')
                # now are the TIME and TIME_RAW values the same?
                if profile_qc.data['TIME'][0] != profile_qc.data['TIME_RAW'][0]:
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
                    print('HISTORY: Duplicate %s flags found, need to troubleshoot. %s' % (vv, profile_qc.Input_filename))
                    exit(1)
                # will be 'LATITUDE, LONGITUDE' or 'DATE, TIME'
                # find the first flag looking at HISTORY_DATE
                idx = non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                    'HISTORY_DATE'].idxmin()
                if len(idx) > 0:
                    LOGGER.warning('PREVIOUS_VALUE is not the same as the %s value, removed from the dataset %s'
                                   % (var, profile_qc.Input_filename))
                    non_temp_codes = non_temp_codes.drop(idx)
            # else it is TEA
            elif vv == 'TIME':
                # check the previous value is the same as the TIME_RAW value
                # convert the previous value to a datetime object
                prevval = convert_time_string(non_temp_codes[dup_idx]['HISTORY_PREVIOUS_VALUE'], format='%Y%m%d%H%M%S')
                # identify the rows where the previous value is not the same as the TIME_RAW value and remove them
                idx = non_temp_codes[dup_idx][~(prevval == profile_qc.data['TIME_RAW'][0])].index
                if len(idx) > 0:
                    LOGGER.warning('Duplicated PREVIOUS_VALUE is not the same as the TIME_RAW value, removed %s'
                                   % profile_qc.Input_filename)
                    non_temp_codes = non_temp_codes.drop(idx)
            else:
                # handle any duplicated position flags here
                # keep the earliest LATITUDE or LONGITUDE flag and remove the others
                LOGGER.warning(
                    'HISTORY: Multiple %s flags found in histories and duplicates removed. %s' % (vv, profile_noqc.Input_filename))
                # find the first flag looking at HISTORY_DATE
                idx = non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'] == vv, 'HISTORY_DATE'].idxmin()
                # remove the other LOA flags
                non_temp_codes = non_temp_codes.drop(
                    non_temp_codes.loc[
                        non_temp_codes['HISTORY_PARAMETER'].values == vv].index.difference(
                        [idx]))

        # copy this information to the PARAMETER_RAW value if it isn't the same, check only where the parameter exactly matches vv
        if vv in ['LATITUDE', 'LONGITUDE']:
            # is the previous_value within 0.01 of the LATITUDE or LONGITUDE_RAW value?
            if not np.allclose(np.round(float(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                'HISTORY_PREVIOUS_VALUE'].values[0]), 6), np.round(profile_qc.data[var][0], 6), atol=0.01):
                # if the previous value is the negative of the raw value, then it is a valid value, change the profile_qc.data[var] to the previous value
                if np.allclose(abs(float(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                        'HISTORY_PREVIOUS_VALUE'].values[0])), profile_qc.data[var][0], 6, atol=0.01):
                    profile_qc.data[var] = float(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                        'HISTORY_PREVIOUS_VALUE'].values[0])
                    LOGGER.info('HISTORY: Updating %s to match the *raw.nc file. %s'
                                   % (var, profile_qc.Input_filename))
                else:
                    LOGGER.error('HISTORY: Previous value for %s is not the same as the %s_RAW value, Updating previous value to match. %s'
                                   % (vv, vv, profile_qc.Input_filename))
                    # if the previous value is not within 0.01 of the LATITUDE or LONGITUDE_RAW value, then update the previous value to match the raw value
                    non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv, 'HISTORY_PREVIOUS_VALUE'] = \
                        str(round(profile_qc.data[var][0], 6))
        elif vv in ['TIME']:
            # TIME_RAW is in datetime format and HISTORY_PREVIOUS_VALUE is in string format
            # if the HISTORY_PREVIOUS_VALUE is not zeros, then it is a valid date
            if int(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                'HISTORY_PREVIOUS_VALUE'].values[0]) == 0:
                # convert the HISTORY_PREVIOUS_VALUE to a datetime object if it is not already
                prevval = convert_time_string(non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv,
                    'HISTORY_PREVIOUS_VALUE'].values[0], '%Y%m%d%H%M%S', 'datetime')
                # check the previous value is the same as the TIME_RAW value
                if not prevval == profile_qc.data[var][0]:
                    LOGGER.info('HISTORY: Updating previous value to match the %s_RAW value in *raw.nc file. %s'
                                   % (vv, profile_qc.Input_filename))
                    # for time, keep TIME_RAW as the previous value
                    non_temp_codes.loc[non_temp_codes['HISTORY_PARAMETER'].values == vv, 'HISTORY_PREVIOUS_VALUE'] = profile_qc.data['TIME_RAW'][0].strftime('%Y%m%d%H%M%S')

    # Filter the rows where HISTORY_PARAMETER is TEMP
    temp_codes = combined_histories[combined_histories['HISTORY_PARAMETER'] == 'TEMP']
    # index any temp_codes['HISTORY_PREVIOUS_VALUE'].values that cannot be converted to float32
    bad_idx = temp_codes[pd.to_numeric(temp_codes['HISTORY_PREVIOUS_VALUE'], errors='coerce').isna()].index
    if len(bad_idx) > 0:
        # replace the bad values with None
        temp_codes.loc[bad_idx, 'HISTORY_PREVIOUS_VALUE'] = None

    # get the index of the rows to drop for TEMP variables only
    idx = temp_codes[(temp_codes.duplicated(subset=['HISTORY_QC_CODE', 'HISTORY_START_DEPTH'], keep=False)) &
                        (temp_codes['HISTORY_PREVIOUS_VALUE'].values.astype('float32') > 90)].index
    if len(idx) > 0:
        LOGGER.warning(
            'HISTORY: Duplicate QC code encountered and removed in create_flag_feature: %s. Please review. %s'
            % (temp_codes.loc[idx, 'HISTORY_QC_CODE'].unique(), profile_qc.Input_filename))
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
            ii = np.where(np.isclose(np.round(profile_qc.data['DEPTH'], 2), np.round(row['HISTORY_START_DEPTH'], 2),
                            atol=0.01))[0]
            # check the previous value is the same as the TEMP_RAW value
            if not np.isclose(round(float(row['HISTORY_PREVIOUS_VALUE']), 2), np.round(profile_qc.data['TEMP_RAW'][ii], 2).item(), atol=0.01):
                # remove this row from the dataframe
                profile_qc.histories = profile_qc.histories.drop(idx)
                # log the error
                LOGGER.warning('HISTORY: Duplicate QC code removed: %s. Please review. %s' % (row['HISTORY_QC_CODE'], profile_qc.Input_filename))
        # reset the index
        profile_qc.histories = profile_qc.histories.reset_index(drop=True)
        if profile_qc.histories.duplicated(['HISTORY_PARAMETER', 'HISTORY_QC_CODE', 'HISTORY_START_DEPTH']).any():
            LOGGER.warning('HISTORY: Duplicated flags remain in the qc file. Please review. %s' % profile_qc.Input_filename)

    # reset the index
    profile_qc.histories = profile_qc.histories.reset_index(drop=True)
    return profile_qc

def check_profile_lengths(profile_qc, profile_noqc):
    # Check the lengths of the edited and raw profiles and amend the qc profile if necessary

    if profile_qc.netcdf_file_obj.variables['No_Depths'][:][0] != profile_noqc.netcdf_file_obj.variables['No_Depths'][:][0]:
        LOGGER.warning('Raw and edited profiles are different lengths.')
        # make the dataframe the length of the maximum number of depths in either profile
        max_depths = max(profile_qc.netcdf_file_obj.variables['No_Depths'][:][0],
                         profile_noqc.netcdf_file_obj.variables['No_Depths'][:][0])
        profile_qc.data = profile_qc.data.reindex(index=range(max_depths))
    else:
        # return the profile.data dataframe with rows indexed to the same length as the qc profile
        profile_qc.data = profile_qc.data.reindex(index=range(profile_qc.netcdf_file_obj.variables['No_Depths'][:][0]))

    return profile_qc

def restore_temp_val(profile):
    """
    Restore the temperature values that are associated with
    the 'CS' (surface spike removed) flag. That means identifying them, putting them back into the
    TEMP field, then putting a flag of 3 (probably bad) on them. The values can also stay
    in the HISTORY_PREVIOUS_VALUE field. This process would need to apply to both the TEMP
    and TEMP_RAW (from the *raw.nc file).
    """

    df = profile.data
    # index of CS flags in histories:
    idx = profile.histories['HISTORY_QC_CODE'] == 'CSR'
    depths = profile.histories['HISTORY_START_DEPTH'][idx].values.astype('float32')
    temps = profile.histories['HISTORY_PREVIOUS_VALUE'][idx].values.astype('float32')
    # get the depths where the TEMP values are greater than 90 and the depth is less than 4
    missing_depths = df.loc[(np.isclose(df['TEMP'], 99.99,rtol=0.01)) & (df['DEPTH'] < 4), 'DEPTH'].values.astype('float32')

    # check if the temperature values are missing & replace with previous value if they are:
    # do for both TEMP and TEMP_RAW
    # find the depths in the profile data
    ind = np.isin(np.round(df['DEPTH'], 2), np.round(depths, 2)).nonzero()[0]
    # does this profile have a PLA flag? if so, use the previous values to replace the TEMP values
    if 'PLA' in profile.histories['HISTORY_QC_CODE'].values:
        LOGGER.info('Restoring TEMP values for CS flags where PLA exists %s' % profile.Input_filename)
        # update the TEMP values with the previous value
        df.loc[ind, 'TEMP'] = temps
    # makes sure we have the same number of CS flags in the profile data as in the histories before proceeding
    elif (len(ind) > 0) & (len(temps) == len(ind)):
        # check the depths and missing_depths are the same
        if (len(depths) != len(missing_depths)) & len(missing_depths) > 0:
            LOGGER.error('Depths in CS flags do not match the missing depths in the profile data. Updating CS flags with missing depths. %s'
                         % profile.Input_filename)
            # if there are more depths than missing depths, exit with error
            if len(depths) < len(missing_depths):
                # update the profile.histories with the missing depths by adding another row to the histories
                for depth in missing_depths:
                    # check if the depth is already in the histories
                    if not np.isin(np.round(depth, 2), np.round(profile.histories['HISTORY_START_DEPTH'], 2)):
                        # add a new row to the histories with the depth and previous value
                        new_row = {
                            'HISTORY_START_DEPTH': depth,
                            'HISTORY_PREVIOUS_VALUE': str(df.loc[df['DEPTH'] == depth, 'TEMP_RAW'].values[0]),
                            'HISTORY_QC_CODE': 'CSR',
                            'HISTORY_QC_CODE_DESCRIPTION': 'surface_transient',
                            'HISTORY_QC_CODE_VALUE': np.int8(3),
                            'HISTORY_PARAMETER': 'TEMP',
                            'HISTORY_DATE': pd.Timestamp.now().strftime('%Y-%m-%d'),
                            'HISTORY_INSTITUTION': profile.histories['HISTORY_INSTITUTION'].values[0],
                            'HISTORY_SOFTWARE_RELEASE': '2.1',
                            'HISTORY_SOFTWARE': 'Australian XBT Quality Control Cookbook Version 2.1'
                        }
                        df_new_row = pd.DataFrame([new_row])
                        df_new_row['HISTORY_PREVIOUS_VALUE'] = df_new_row['HISTORY_PREVIOUS_VALUE'].astype('string')
                        df_new_row['HISTORY_DATE'] = pd.to_datetime(df_new_row['HISTORY_DATE'])
                        df_new_row['HISTORY_QC_CODE_VALUE'] = df_new_row['HISTORY_QC_CODE_VALUE'].astype('int8')
                        # append the new row to the histories
                        profile.histories = pd.concat([profile.histories, df_new_row], ignore_index=True)
                        # reset the index
                        profile.histories = profile.histories.reset_index(drop=True)
                        # re-get the depths and temps
                        idx = profile.histories['HISTORY_QC_CODE'] == 'CSR'
                        depths = profile.histories['HISTORY_START_DEPTH'][idx].values.astype('float32')
                        temps = profile.histories['HISTORY_PREVIOUS_VALUE'][idx].values.astype('float32')
                        ind = np.isin(np.round(df['DEPTH'], 2), np.round(depths, 2)).nonzero()[0]

        # temps should be equal to df['TEMP_RAW'][ind], let's check they are equal and there are no missing values
        if (temps != df['TEMP_RAW'][ind]).all() and (temps.max() <= 99) and (df['TEMP_RAW'][ind].max() <= 99):
            # check they are within 0.01 of each other
            if not np.allclose(temps, df['TEMP_RAW'][ind], atol=0.01):
                # check the median difference with a bigger tolerance:
                if np.median(np.abs(temps - df['TEMP_RAW'][ind])) > 0.01:
                    LOGGER.error('TEMP_RAW values do not match the HISTORY_PREVIOUS_VALUE for CS flags %s'
                                 % profile.Input_filename)
                    return profile

        # update the TEMP values with the TEMP_RAW values if they do not contain values == 99.99
        if not (np.isclose(df['TEMP_RAW'][ind], 99.99, rtol=0.01)).any():
            df.loc[ind, 'TEMP'] = df.loc[ind, 'TEMP_RAW']
        # update the TEMP_RAW values with the HISTORY_PREVIOUS_VALUE values if the TEMP_RAW values have values > 99 and the
        # HISTORY_PREVIOUS_VALUE values do not
        elif not (np.isclose(temps, 99.99,rtol=0.01)).any() and (np.isclose(df['TEMP_RAW'][ind], 99.99,rtol=0.01)).any():
            LOGGER.info('Restoring TEMP and TEMP_RAW missing data from histories %s' % profile.Input_filename)
            df.loc[ind, 'TEMP_RAW'] = temps
            df.loc[ind, 'TEMP'] = temps
        else:
            LOGGER.error('TEMP_RAW values and HISTORY_PREVIOUS_VALUE values are both > 99 for CS flags. Please review. %s'
                         % profile.Input_filename)
    elif len(depths) == 0:
        LOGGER.warning('No CS flags found in the histories. %s' % profile.Input_filename)
    else:
        # the number of CS flags in the profile data does not match the number of missing temps in the data
        LOGGER.warning('Number of CS flags in the profile data does not match the number of missing temps in the data. %s'
                     % profile.Input_filename)

    # find any depths with 99.99 values that are flagged with SPA or IPA or HFA
    idx_temp = (df['TEMP'] > 90)
    if idx_temp.any() and ind.any():
        # check if there are any SPA, IPA or HFA flags at the same depth
        idx2 = profile.histories['HISTORY_START_DEPTH'].isin(df.loc[idx_temp, 'DEPTH'])
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
                temps = profile.histories['HISTORY_PREVIOUS_VALUE'][ind2].values.astype('float32')
                # is the first value of ind2 only one different from last value of ind?
                if (ind2[0] - ind[-1]) == 1:
                    LOGGER.info('Restoring 99.99 values for SPA, IPA or HFA flags and changing flag to CSR. %s'
                                % profile.Input_filename)
                    # update the TEMP values with the TEMP_RAW values if they do not contain values == 99.99
                    if not (np.isclose(df['TEMP_RAW'][ind2], 99.99, rtol=0.01)).any():
                        df.loc[ind2, 'TEMP'] = df.loc[ind2, 'TEMP_RAW']
                    # update the TEMP_RAW values with the HISTORY_PREVIOUS_VALUE values if the TEMP_RAW values have values == 99.99 and the
                    # HISTORY_PREVIOUS_VALUE values do not
                    elif not (np.isclose(temps, 99.99, rtol=0.01)).any() and (np.isclose(df['TEMP_RAW'][ind2], 99.99, rtol=0.01)).any():
                        df.loc[ind2, 'TEMP_RAW'] = temps
                        df.loc[ind2, 'TEMP'] = temps
                    else:
                        LOGGER.error('TEMP_RAW values and HISTORY_PREVIOUS_VALUE values are both == 99.99 for CS flags %s'
                                     % profile.Input_filename)
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
                                    % profile.Input_filename)
                        # Get the depths where HISTORY_QC_CODE is CSR
                        csr_depths = profile.histories.loc[
                            profile.histories['HISTORY_QC_CODE'] == 'CSR', 'HISTORY_START_DEPTH']

                        # Remove rows where HISTORY_QC_CODE is SPA, HFA, or IPA and HISTORY_START_DEPTH is in csr_depths
                        profile.histories = profile.histories[
                            ~((profile.histories['HISTORY_QC_CODE'].isin(['SPA', 'HFA', 'IPA'])) &
                              (profile.histories['HISTORY_START_DEPTH'].isin(csr_depths)))]
                        # reset the index
                        profile.histories = profile.histories.reset_index(drop=True)

    # are there any TEMP values that are still == 99.99?
    if (df['TEMP'] > 90).any():
        # see if any of the histories have a valid TEMP value for these depths
        idx_temp = np.isclose(df['TEMP'], 99.99, rtol=0.01)
        depths = df.loc[idx_temp, 'DEPTH']
        # find each depth in the histories where the HISTORY_START_DEPTH matches the depths
        # and the HISTORY_PARAMETER is TEMP and the HISTORY_PREVIOUS_VALUE is not 99.
        for depth in depths:
            # find the index of the depth in the histories
            idx2 = (np.isclose(profile.histories['HISTORY_START_DEPTH'], depth, atol=0.01) &
                    (profile.histories['HISTORY_PARAMETER'] == 'TEMP') &
                    (profile.histories['HISTORY_PREVIOUS_VALUE'] != '99.99'))
            # if there are any histories with a valid TEMP value for this depth
            if idx2.any():
                # get the previous value for this depth
                previous_value = profile.histories.loc[idx2, 'HISTORY_PREVIOUS_VALUE'].values[0]
                # update the TEMP value in the profile data
                df.loc[idx_temp & (df['DEPTH'] == depth), 'TEMP'] = float(previous_value)
                # update the TEMP_RAW value in the profile data
                if (np.isclose(df['TEMP_RAW'][idx_temp], 99.99, rtol=0.01)).any():
                    df.loc[idx_temp & (df['DEPTH'] == depth), 'TEMP_RAW'] = float(previous_value)
                LOGGER.info('Restoring TEMP values for depths where TEMP == 99.99. %s' % profile.Input_filename)

    # update profile data
    profile.data = df
    return profile


def create_flag_feature(profile):
    """ Take the existing QC code values and turn them into a integer representation. One bit for every code.
    And there are now two variables, one for accept codes, one for reject codes."""

    # get the flag quality table with all the historic codes
    # create a dataframe with the codes and their integer representation
    dfa, dfr = read_flag_quality_table(all=True)
    # combine the two dataframes, combining the 'QC_accept_code' and 'QC_reject_code' columns into one column labelled 'byte_value'
    df = pd.concat([dfa, dfr], ignore_index=True)
    # combine the 'QC_accept_code' and 'QC_reject_code' columns into one column labelled 'byte_value'
    df['byte_value'] = df['QC_accept_code'].fillna(0).astype(int) + df['QC_reject_code'].fillna(0).astype(int)

    df_data = profile.data.copy(deep=True)

    # set the fields to zeros to start
    df_data['QC_accept_code'] = 0
    df_data['QC_reject_code'] = 0
    # where the TEMP is NaN, set the QC_accept_code and QC_reject_code to NaN
    df_data.loc[df_data['TEMP'].isna(), ['QC_accept_code', 'QC_reject_code']] = np.nan

    # perform the flag mapping on the original flags and create the two new variables
    codes = profile.histories
    # if the TEMP_quality_control values are 0 and the TEMP_RAW_quality_control values are not, update the TEMP_quality_control
    # values to be the same as the TEMP_RAW_quality_control values
    idx = (df_data['TEMP_quality_control'] == 0).all() & (df_data['TEMP_RAW_quality_control'] != 0).all()
    if idx.any():
        LOGGER.warning('TEMP_quality_control values are 0 and TEMP_RAW_quality_control values are not. Updating. %s'
                       % profile.Input_filename)
        df_data.loc[idx, 'TEMP_quality_control'] = df_data.loc[idx, 'TEMP_RAW_quality_control']
        # add QCA to the history
        codes = codes._append({'HISTORY_INSTITUTION': profile.data['Institution'],
                               'HISTORY_QC_CODE': 'QCA',
                               'HISTORY_PARAMETER': 'TEMP',
                               'HISTORY_SOFTWARE': 'Australian XBT Quality Control Cookbook Version 2.1',
                               'HISTORY_DATE': pd.datetime(pd.Timestamp.now()).strftime('%Y-%m-%d'),
                               'HISTORY_START_DEPTH': df_data['DEPTH'].values[0],
                               'HISTORY_QC_CODE_DESCRIPTION': 'scientific_qc_applied',
                               'HISTORY_QC_CODE_VALUE': np.int8(1),
                               'HISTORY_SOFTWARE_RELEASE': '2.1',
                               'HISTORY_PREVIOUS_VALUE': '0'}, ignore_index=True)

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
        # if there are no depths deeper than the last CSR flag, then assume the last CSR flag is at the bottom of the profile and remove the CSR flag
        if not ideps.any():
            LOGGER.warning('No depths deeper than the last CSR flag, removing CSR flag from bottom of profile. %s' % profile.Input_filename)
            # remove the last CSR flag from the histories
            last_csr_idx = codes.loc[idx_csr].index[-1]
            codes = codes.drop(last_csr_idx)
            # reset the index
            codes = codes.reset_index(drop=True)
            # make sure the df_data['TEMP_quality_control'] is updated to be the same as the previous depth
            df_data.loc[df_data['DEPTH'] == depths[-1], 'TEMP_quality_control'] = df_data.loc[(df_data['DEPTH'] < depths[-1]).index[-2], 'TEMP_quality_control']
        else:
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
                tempqc = df.loc[df['code'].str.contains(row['HISTORY_QC_CODE'][:2]), 'TEMP_quality_control'].values
                # check if the two character code appears more than once in the df['code'] column
                if np.size(tempqc) > 1:
                    # if the value is 5
                    if row['tempqc'] in [5]:
                        # check the TEMP_quality_control values in df_data at the depths deeper than row['HISTORY_START_DEPTH'] and get the next value that is not 5
                        deeper_tempqc = df_data.loc[df_data['DEPTH'] > row['HISTORY_START_DEPTH'], 'TEMP_quality_control'].values
                        deeper_tempqc = deeper_tempqc[deeper_tempqc != 5][0]
                        if deeper_tempqc.size == 0:
                            deeper_tempqc = 5
                    else:
                        deeper_tempqc = row['tempqc']
                    # if so, then we need to check that the TEMP_quality_control value is in the same category as the tempqc value
                    # where the categories are 1,2,5 and 3,4
                    if ((row['HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5] and deeper_tempqc in [3 ,4]) or
                            (row['HISTORY_QC_CODE_VALUE'] in [3, 4] and deeper_tempqc in [0, 1, 2, 5])):
                        # update the HISTORY_QC_CODE_VALUE to the tempqc value as the TEMP_quality_control value is in the wrong category
                        if row['HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5] and deeper_tempqc in [0, 1, 2]:
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
    df = df.drop(columns=['code', 'QC_accept_code', 'QC_reject_code'])

    # make sure the previous_values are the same as the data['TEMP_RAW'] values and replace missing TEMP values at CS
    profile.histories = codes
    profile.data = df_data
    profile = restore_temp_val(profile)
    codes = profile.histories
    df_data = profile.data

    # merge the codes with the flag codes
    mapcodes = pd.merge(df, codes, how='right', left_on='full_code', right_on='HISTORY_QC_CODE')

    if mapcodes.empty:
        profile.data['qc_completed'] = 'no'
        return profile
    else:
        # adjust global attribute to say we have done scientific QC
        profile.data['qc_completed'] = 'yes'

    # update the HISTORY_QC_CODE_DESCRIPTION to the df label
    mapcodes['HISTORY_QC_CODE_DESCRIPTION'] = mapcodes['name']

    # update the HISTORY_PARAMETER to the parameter column in df
    mapcodes['HISTORY_PARAMETER'] = mapcodes['Parameter']

    # any flags not included? check for nan in the label column
    nan_values = mapcodes['name'].isna()
    if nan_values.any():
        # we have an extra flag that we haven't coded
        # if any of the flags are in this list which I know about, remove them
        if mapcodes.loc[nan_values, 'HISTORY_QC_CODE'].str.contains("BB|DC|GS|MS").any():
            mapcodes = mapcodes[~nan_values]
        else:
            LOGGER.error('HISTORY: new QC code encountered, please code in the new value. %s %s' % (
                mapcodes.loc[nan_values, 'HISTORY_QC_CODE'].unique(), profile.Input_filename))
            exit(1)

    # now need to assign the codes to the correct depths.
    # code only added in one location at the start depth, QC flags indicate the quality applied
    # for each code, need an array of values same size as DEPTH, then add them all together
    # also check the TEMP_QC_CODE_VALUE is the same as the actual flag in the flag array

    # create a df with the same number of columns as the number of rows in the mapcodes table and number of rows is number of depths
    tempdf = pd.DataFrame(np.zeros((len(df_data), len(mapcodes))) * np.zeros(len(mapcodes)), columns=mapcodes['full_code'])
    # do the same for DEPTH
    depdf = tempdf.copy()
    # iterate over the mapcodes table and fill a column in tempdf with QC values from the tempqc field
    for idx, row in mapcodes.iterrows():
        # get the index of the depth in the data
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        # if this is a CSR flag, just fill the depth with the tempqc value
        if row['HISTORY_QC_CODE'] == 'CSR':
            tempdf.loc[ii, row['full_code']] = row['TEMP_quality_control']
            depdf.loc[ii, row['full_code']] = row['DEPTH_quality_control']
        else:
            # fill the tempdf from the depth index to the maximum index
            tempdf.loc[ii:, row['full_code']] = row['TEMP_quality_control']
            depdf.loc[ii:, row['full_code']] = row['DEPTH_quality_control']
        # for flags that have been interpolated or filtered, these are 5 and 2 deeper. Change the flag at these depths to 5
        if row['HISTORY_QC_CODE'] in ['SPA', 'HFA', 'IPA', 'EIA']:
            # 2 should have been assigned above, now just overwriting with 5
            tempdf.loc[ii, row['full_code']] = 5

    # index of the tempdf rows that have a value of 5
    idx = tempdf.eq(5).any(axis=1)
    # calculate the maximum tempqc value for each depth
    tempdf['tempqc'] = tempdf.max(axis=1)
    # overwrite the tempqc value with 5 where there is a 5 in the tempdf
    tempdf.loc[idx, 'tempqc'] = 5

    # repeat for the depdf dataframe
    idx = depdf.eq(5).any(axis=1)
    # calculate the maximum depthqc value for each depth
    depdf['depthqc'] = depdf.max(axis=1)
    # overwrite the depthqc value with 5 where there is a 5 in the depdf
    depdf.loc[idx, 'depthqc'] = 5

    # find any depths where the tempqc value is less than the TEMP_quality_control value not including the 5 values
    # and ignore where LOA has changed the QC to 2 from 1
    idx = (df_data['TEMP_quality_control'] > tempdf['tempqc']) & (df_data['TEMP_quality_control'] != 5)
    # if there are any idx and there are values other than 'LOA' in the HISTORY_QC_CODE column, log a warning
    if idx.any():
        if not mapcodes.loc[mapcodes['HISTORY_QC_CODE'].isin(['LOA'])].empty:
            LOGGER.warning('TEMP_quality_control values are greater than the tempqc values. %s' % profile.Input_filename)
    idx = (df_data['DEPTH_quality_control'] > depdf['depthqc']) & (df_data['DEPTH_quality_control'] != 5)
    if idx.any():
        LOGGER.warning('DEPTH_quality_control values are greater than the depthqc values. %s' % profile.Input_filename)

    # update the TEMP_quality_control field with the tempdf values
    df_data['TEMP_quality_control'] = tempdf['tempqc']
    # update the DEPTH_quality_control field with the depdf values
    df_data['DEPTH_quality_control'] = depdf['depthqc']

    # get the maximum value of all the *_quality_control columns except for TEMP_quality_control
    max_qc = mapcodes.filter(like='_quality_control').max(axis=0)
    # loop through the max_qc values and update the df_data with the maximum value
    for col in max_qc.index:
        if col != 'TEMP_quality_control' and col != 'DEPTH_quality_control' and pd.notna(max_qc[col]):
            # update the df_data with the maximum value of the column
            df_data[col] = max_qc[col]


    # Iterate over the history table.
    for idx, row in mapcodes.iterrows():
        # Get depth index
        ii = (np.abs(df_data['DEPTH'] - row['HISTORY_START_DEPTH'])).argmin()
        # if this is an accept code (QC_Flag = 1, 2, 3, 5) then add it to the accept code array
        if row['HISTORY_QC_CODE_VALUE'] in [0, 1, 2, 5]:
            # adding them together - is there a more correct way to do this?
            # Add byte values (masks) for accept codes
            df_data.loc[ii, 'QC_accept_code'] = df_data.loc[ii, 'QC_accept_code'] + np.float64(row['byte_value'])
        else:
            # Add byte values (masks) for reject codes
            df_data.loc[ii, 'QC_reject_code'] = df_data.loc[ii, 'QC_reject_code'] + np.float64(row['byte_value'])

    # update the histories with the correct tempqc values from mapcodes
    mapcodes['HISTORY_QC_CODE_VALUE'] = mapcodes['TEMP_quality_control']
    # drop unwanted columns from mapcodes keeping only 'HISTORY*' columns
    mapcodes = mapcodes.filter(like='HISTORY_')

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
    woce_date = profile.netcdf_file_obj['woce_date'][0]
    # turn the woce_date into a string that is 8 characters long with spaces filled with zeros
    woce_date = str(woce_date).zfill(8)
    # transform the date to a datetime object
    date1 = convert_time_string(woce_date, '%Y%m%d')
    date2 = convert_time_string(woce_date, '%d%m%Y')
    year = (date1 if not pd.isna(date1) else date2).year

    if np.sum(~depth.mask) == 0:
        LOGGER.error('Profile not processed, No data in the file: %s' % profile.Input_filename)
        return False

    if (data_type != 'XB') and (data_type != 'BA'):  # and data_type != 'BA' and data_type != 'TE':
        LOGGER.error('Profile not processed as it is type %s %s ' % (data_type, profile.Input_filename))
        return False

    # if the date of the profile is before 1950, skip it
    if year < 1950:
        LOGGER.error('Profile not processed, date is before 1950: %s' % profile.Input_filename)
        return False

    if duplicate_flag == 'D':
        LOGGER.error(
            'Profile not processed. Tagged as duplicate profile in original netcdf file %s' % profile.Input_filename)
        return False

    if 'DU' in histcodes:
        LOGGER.error(
            'Profile not processed. Tagged as duplicate profile in original netcdf file %s' % profile.Input_filename)
        return False

    data_vars = temp_prof_info(profile.netcdf_file_obj)
    if not any('TEMP' in str(v) for v in data_vars.values()):
        LOGGER.error('Profile not processed, no TEMP in file %s' % profile.Input_filename)
        return False

    return True


def make_dataframe(profile_ed, profile_raw, profile_turo):
    # convert the data in profile to a parquet file
    # profile_ed.data is already a dataframe, but check the columns against the netcdfVars.csv file
    vars = read_variables_config()
    # get a list of variables from the variable_name column
    varslist = vars['variable_name'].tolist()

    # get the columns from the profile_ed.data
    columns = profile_ed.data.columns.tolist()
    # check if the columns in profile_ed.data are in the varslist
    missing_vars = [var for var in columns if var not in varslist]
    if missing_vars:
        # if the variable name ends in _RAW and the values are the same as the variable name without _RAW, then remove the _RAW
        for var in missing_vars:
            if var.endswith('_RAW'):
                var_no_raw = var[:-4]
                if var_no_raw in columns and profile_ed.data[var].equals(profile_ed.data[var_no_raw]):
                    # remove the _RAW column
                    profile_ed.data = profile_ed.data.drop(columns=[var])
                    # LOGGER.info('Removing %s from the dataframe as it is the same as %s' % (var, var_no_raw))
            elif var.endswith('_quality_control'):
                # if the variable name ends with _quality_control and the values are zero or the same as the variable name without _quality_control, then remove the _quality_control
                var_no_qc = var[:-16]  # remove _quality_control
                if var_no_qc in columns and (profile_ed.data[var].eq(0).all() or profile_ed.data[var].equals(profile_ed.data[var_no_qc])):
                    # remove the _quality_control column
                    profile_ed.data = profile_ed.data.drop(columns=[var])
                    # LOGGER.info('Removing %s from the dataframe as it is the same as %s' % (var, var_no_qc))

    # save the final dataframe to a
    df = profile_ed.data.copy(deep=True)

    return df


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
    parser.add_argument('-i', '--input-xbt-campaign-path', type=str, nargs='+',
                        help="paths to one or more database folders")
    parser.add_argument('-o', '--output-folder', nargs='?', default=1,
                        help="output directory of generated files")
    parser.add_argument('-l', '--log-file', nargs='?', default=1,
                        help="log directory")
    parser.add_argument('-g', '--group-by-year', action='store_true',)
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

    for i, path in enumerate(vargs.input_xbt_campaign_path):
        if not os.path.exists(path):
            msg = '%s not a valid path' % path
            print(msg, file=sys.stderr)
            # remove the path from the list
            vargs.input_xbt_campaign_path.pop(i)
    # if no input paths are provided, raise an error
    if not vargs.input_xbt_campaign_path:
        msg = 'No valid input paths provided. Please provide at least one valid path.'
        print(msg, file=sys.stderr)
        exit(1)


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
    ./xbt_dm_imos_conversion.py -i XBT/GTSPPmer2017/GTSPPmer2017MQNC_keys.nc -o /tmp/xb -g 
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
    group_by_year=False
    '''
    for input_path in vargs.input_xbt_campaign_path:
        keysall = XbtKeys(input_path)
        # group the keys by year and process each year separately
        calls = keysall.group_by_callsign_year(vargs.group_by_year)
        # process each year separately
        for callsign in calls:
            for year in calls[callsign]:
                # get the subset of keys for the year
                stations = calls[callsign][year]
                # if there are no keys for this year, skip it
                if stations is None:
                    LOGGER.warning('No keys found for year %s in %s' % (year, keysall.dbase_name))
                    continue

                print('Processing database %s for year %s with callsign %s' % (keysall.dbase_name, year, callsign))

                # read all the variables from the netcdfVars.csv file
                vars = read_variables_config()
                # create dfall with the variables from the netcdfVars.csv file that do not start with 'HISTORY_'
                dfall = pd.DataFrame(columns=vars[vars['variable_name'].str.startswith('HISTORY_') == False]['variable_name'].tolist())
                # add the station_number column to dfall
                dfall['station_number'] = pd.Series(dtype='int64')
                # create dfhist with the variables from the netcdfVars.csv file that start with 'HISTORY_'
                dfhist = pd.DataFrame(columns=vars[vars['variable_name'].str.startswith('HISTORY_')]['variable_name'].tolist())
                # add the station_number column to dfhist
                dfhist['station_number'] = pd.Series(dtype='int64')

                for f in stations:
                    # if f != 61013321:
                    #     continue
                    fpath = '/'.join(re.findall('..?', str(f))) + 'ed.nc'
                    fname = os.path.join(keysall.dbase_name, fpath)
                    # make input_filename here
                    input_filename = os.path.join(os.path.basename(keysall.dbase_name), fpath)

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
                            print('Processing profile %s' % f)
                            # for example where depths are different, metadata is different etc between the ed and raw files.
                            profile_ed = coordinate_data(profile_ed, profile_raw, profile_turo, f)
                            if not profile_ed:
                                continue
                            profile_df = make_dataframe(profile_ed, profile_raw, profile_turo)
                            # add the station number to the dataframe
                            profile_df['station_number'] = f
                            # drop all columns in profile_df that are all NaN
                            profile_df = profile_df.dropna(axis=1, how='all')
                            # add to the big dataframes
                            dfall = pd.concat([dfall, profile_df], ignore_index=True)
                            # add station number to the histories
                            profile_ed.histories['station_number'] = f
                            # check for HISTORY_DATE datatype here
                            if profile_ed.histories['HISTORY_DATE'].dtype != 'datetime64[ns]':
                                profile_ed.histories['HISTORY_DATE'] = pd.to_datetime(profile_ed.histories['HISTORY_DATE'], errors='coerce')
                            # add the histories to the big dataframe
                            dfhist = pd.concat([dfhist, profile_ed.histories], ignore_index=True)
                    else:
                        LOGGER.warning('Profile not processed, file %s is in keys file, but does not exist' % f)

                if dfall.empty:
                    LOGGER.warning('No profiles found in %s' % keysall.dbase_name)
                    continue
                # Drop columns labelled *_RAW_quality_control if they contain all 0s
                dfall = dfall.loc[:, ~(dfall.columns.str.contains('_RAW_quality_control') & (dfall == 0).all())]

                # add table metadata to the dfall dataframe
                dfall = set_metadata(dfall, tbl_meta={'Parent file':keysall.dbase_name})
                # write the dataframe to a parquet file
                if not vargs.group_by_year:
                    pq_filename = os.path.join(vargs.output_folder, os.path.basename(keysall.dbase_name) + '.parquet')
                else:
                    pq_filename = os.path.join(vargs.output_folder, os.path.basename(keysall.dbase_name) + '_' + callsign + str(year) + '.parquet')
                pq.write_table(dfall, pq_filename)
                if not vargs.group_by_year:
                    pq_filename = os.path.join(vargs.output_folder,
                                               os.path.basename(keysall.dbase_name) + '_histories.parquet')
                else:
                    pq_filename = os.path.join(vargs.output_folder,
                                           os.path.basename(keysall.dbase_name)  + '_' + callsign + str(year) + '_histories.parquet')

                dfhist.to_parquet(pq_filename, index=False)

    print('All done')
