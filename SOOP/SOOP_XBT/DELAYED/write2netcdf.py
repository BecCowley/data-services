# import a parquet file and write it to a netcdf file
import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import strftime, gmtime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num

from xbt_utils import make_transect_id
from xbt_utils import read_flag_quality_table, read_variables_config, read_globals_config

CSIRO_INSTITUTION = "Australia Commonwealth Scientific and Industrial Research Organization (CSIRO)"
BOM_INSTITUTION = "Australia Bureau of Meteorology (BoM)"

def create_filename_output(output_folder, prof, hist, imosformat=True, profile_raw=False):
    if imosformat:
        if profile_raw:
            fv = 'FV00'
        else:
            fv = 'FV01'

        filename = 'IMOS_SOOP-XBT_T_%s_%s_%s_%s' % (
            prof['TIME'].iloc[0].strftime('%Y%m%dT%H%M%SZ'), prof['SOOP_line_label'].iloc[0], fv,
            prof['Cruise_ID'].iloc[0])

        # if profile histories contains TP, add 'TEST' to the filename
        if 'TPR' in hist['HISTORY_QC_CODE'].values:
            filename = filename + 'TEST'
        filename = os.path.join(output_folder, filename + '.nc')
        # if the filename already exists, add a '-RX' where X is the number of times the file has been created, starting with 1, until a unique filename is found
        if os.path.exists(filename):
            count = 1
            while os.path.exists(filename):
                filename = os.path.join(output_folder, 'IMOS_SOOP-XBT_T_%s_%s_%s_%s-R%s.nc' % (
                    prof['TIME'].iloc[0].strftime('%Y%m%dT%H%M%SZ'), prof['SOOP_line_label'].iloc[0], fv,
                    prof['Cruise_ID'].iloc[0], count))
                count += 1
    else:
        # format is VNHF_202507231645_D_001.nc
        # where VNHF is the ship Callsign, 202507231645 is the time of the profile,
        # <D, R> is <delayed mode, real time>,
        # and 001 is optional: number of deployment at this date/time and location from the same vessel
        """create the filename for the output netcdf file"""

        # if the profile histories contains TP, do not create the file and return None
        if 'TPR' in hist['HISTORY_QC_CODE'].values:
            print("Profile contains TP, not creating netcdf file")
            return None

        # get the Callsign and time from the profile
        xbt_callsign = prof['Callsign'].iloc[0].strip()
        xbt_time = prof['TIME'].iloc[0].strftime('%Y%m%d%H%M')

        # check for QC flags on temperature, assume the data is from a delayed mode profile
        if any(prof['TEMP_quality_control'] > 0):
            qc_flag = 'D'
        else:
            qc_flag = 'R'

        # create the filename
        filename = os.path.join(output_folder, f"{xbt_callsign}_{xbt_time}_{qc_flag}.nc")

        # if the filename already exists, add a '-RX' where X is the number of times the file has been created, starting with 1, until a unique filename is found
        if os.path.exists(filename):
            count = 1
            while os.path.exists(filename):
                filename = os.path.join(output_folder, f"{xbt_callsign}_{xbt_time}_{qc_flag}-R{count}.nc")
                count += 1

    return filename


def _append_suffix_to_filename(path, suffix):
    """Return a sibling file path by appending suffix before the extension."""
    pth = Path(path)
    return str(pth.with_name(f"{pth.stem}{suffix}{pth.suffix}"))


def build_globals_file_paths(globals_input_file):
    """Build expected globals CSV file paths from the base IMOS globals file."""
    return {
        'default': globals_input_file,
        'csiro_pre2016': _append_suffix_to_filename(globals_input_file, "_pre2016"),
        'bom': _append_suffix_to_filename(globals_input_file, "_BOM"),
        'oceantrax': _append_suffix_to_filename(globals_input_file, "_OceanTraX")
    }


def preload_globals_configs(globals_input_file):
    """Read globals CSV files once and keep them in memory for profile-level selection."""
    globals_paths = build_globals_file_paths(globals_input_file)
    globals_configs = {}

    for key, path in globals_paths.items():
        if os.path.exists(path):
            globals_configs[key] = read_globals_config(path)
        elif key in ['default', 'oceantrax']:
            raise FileNotFoundError(f"Required globals file not found: {path}")
        else:
            print(f"Optional globals file not found: {path}. Using defaults when needed.")

    return globals_configs


def select_imos_globals_config(globals_configs, profile):
    """Select preloaded IMOS globals config for a profile."""
    institution = profile['Institution'].iloc[0] if 'Institution' in profile.columns else None
    latest_time = profile['TIME'].max() if 'TIME' in profile.columns else None

    if institution == CSIRO_INSTITUTION and pd.notna(latest_time) and latest_time < datetime(2017, 1, 1):
        if 'csiro_pre2016' in globals_configs:
            return globals_configs['csiro_pre2016']
    elif institution == BOM_INSTITUTION:
        if 'bom' in globals_configs:
            return globals_configs['bom']

    return globals_configs['default']

def write_output_nc(output_folder, profile, history, globals_attrs=None, globals_file_path='netcdfGlobalAtts.csv', profile_raw=False, historic_flags=False, imosformat=True):
    """output the data to the IMOS format netcdf version
    :param output_folder: the folder to write the netcdf file to
    :param profile: the profile DataFrame
    :param history: the history DataFrame
    :param globals_attrs: optional preloaded globals attribute dictionary
    :param profile_raw: if True, the create a FV00 file, if False create a FV01 file, default is False
    :param imosformat: if True, create a file in IMOS format, otherwise create a file in OceanTrax format
    """

    # now begin write out to new format
    netcdf_filepath = create_filename_output(output_folder, profile, history, imosformat, profile_raw)
    # if netcdf_filepath is None:
    if netcdf_filepath is None:
        print("No netcdf file created for this %s profile due to TP in history or other conditions %s." % (imosformat, profile['station_number'].iloc[0]))
        return
    print('Creating output %s' % netcdf_filepath)

    # reset the index of the profile DataFrame
    profile = profile.reset_index(drop=True)
    # reset the index of the history DataFrame
    history = history.reset_index(drop=True)
    # read the variables config file
    vars = read_variables_config()
    # if probe_type_raw is optional: if the PROBE_TYPE*_RAW columns are the same as the PROBE_TYPE* columns, put None in the PROBE_TYPE*_RAW columns
    if vars.loc[vars['variable_name'] == 'PROBE_TYPE_RAW', 'variable optional/required (1=required 0=optional)'].values[0] == 0:
        probe_types = [col for col in profile.columns if col.startswith('PROBE_TYPE') and '_RAW' not in col]
        for pt in probe_types:
            raw_col = pt + '_RAW'
            if raw_col in profile.columns and profile[pt].equals(profile[raw_col]):
                profile[raw_col] = None

    # add the WIGOS_ID to the profile DataFrame and set it to '0-22000-0-' + profile['SOT_ID'].astype(str)
    if 'SOT_ID' in profile.columns and not profile['SOT_ID'].isnull().all():
            profile['WIGOS_ID'] = '0-22000-0-' + profile['SOT_ID'].astype(str)
    else:
        profile['SOT_ID'] = None  # default value if SOT_ID is not present
        profile['WIGOS_ID'] = None  # default value if SOT_ID is not present
    # read global attributes config, or use preloaded attributes
    if globals_attrs is not None:
        globals_list = globals_attrs.copy()
    else:
        globals_list = read_globals_config(globals_file_path)
    # first get a list of the attributes attached to the variables
    extra_atts = vars[vars['is_var_att_global'] == 'att']
    # get a list of the global attributes
    extra_globals = vars[vars['is_var_att_global'] == 'global']
    # get a list of the variable attributes
    vars = vars[vars['is_var_att_global'] == 'var']
    # Identify attribute columns starting with 'att_'
    att_cols = [col for col in vars.columns if col.startswith('att_')]
    # remove the 'att_' prefix from the attribute columns
    att_labels = [col.replace('att_', '') for col in att_cols]

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(profile['DEPTH']))
        output_netcdf_obj.createDimension('N_HISTORY', 0) #make this unlimited
        output_netcdf_obj.createDimension('str3', 3)  # for three character strings
        output_netcdf_obj.createDimension('str20', 20)  # for twenty character strings
        output_netcdf_obj.createDimension('str60', 60)  # for sixty character strings
        output_netcdf_obj.createDimension('str150', 150)  # for one hundred and fifty character strings

        # Create the variables from the vars Dataframe by looping through the rows
        for index, row in vars.iterrows():
            vv = row['variable_name']
            # print(vv)
            # check if there is data in the profile DataFrame for this variable. Required variables must be kept in the output netcdf file
            if vv not in profile.columns and vv not in history.columns and row['variable optional/required (1=required 0=optional)'] == 0:
                continue
            # if vv is optional and all values in profile[vv] are NaN, skip it
            if vv in profile.columns and profile[vv].isnull().all() and row['variable optional/required (1=required 0=optional)'] == 0:
                continue
            # if vv is optional and all values in history[vv] are NaN, skip it
            if vv in history.columns and history[vv].isnull().all() and row['variable optional/required (1=required 0=optional)'] == 0:
                continue
            # get the datatype as specified in the att_variable_type column
            dt = row['variable_type']
            # get the fill value as specified in the att_fillValue column
            fillvalue = row['fillValue']
            # get the dimensions as specified in the Dimensions column
            dimensions = row['Dimensions']

            # if dimensions is NaN create the variable without dimensions or fill value
            if pd.isna(dimensions) and pd.isna(fillvalue):
                output_netcdf_obj.createVariable(vv, datatype=dt)
            elif pd.isna(dimensions) and not pd.isna(fillvalue):
                # create the variable in the netcdf file with fill value
                output_netcdf_obj.createVariable(vv, datatype=dt, fill_value=fillvalue)
            elif not pd.isna(dimensions) and pd.isna(fillvalue):
                # create the variable in the netcdf file with dimensions
                output_netcdf_obj.createVariable(vv, datatype=dt, dimensions=dimensions.split(','))
            else:
                # create the variable in the netcdf file with dimensions and fill value
                output_netcdf_obj.createVariable(vv, datatype=dt, fill_value=fillvalue, dimensions=dimensions.split(','))
            # set the attributes for the variable
            for att in att_labels:
                # the column is labelled att_<att_name> in the vars DataFrame
                att_name = 'att_' + att
                if att_name in row and pd.notna(row[att_name]):
                    # if the attribute is a validMax, validMin change it to the data type specified in the variable_type column
                    if att in ['valid_max', 'valid_min']:
                        # convert the value to the data type specified in the dt variable where types are int8, int64, int32, float32, float64
                        if dt == 'int8':
                            row[att_name] = np.int8(row[att_name])
                        elif dt == 'int64':
                            row[att_name] = np.int64(row[att_name])
                        elif dt == 'int32':
                            row[att_name] = np.int32(row[att_name])
                        elif dt == 'float32':
                            row[att_name] = np.float32(row[att_name])
                        elif dt == 'float64':
                            row[att_name] = np.float64(row[att_name])

                    if att in 'flag_values':
                        # convert to a byte array
                        if isinstance(row[att_name], str):
                            # if the attribute is a string, convert it to a list of bytes
                            row[att_name] = np.array([np.byte(x.strip().strip(',')) for x in row[att_name].split(' ')])
                    # set the attribute on the variable
                    setattr(output_netcdf_obj.variables[vv], att, row[att_name])
        # read the flag quality tables
        dfa, dfr = read_flag_quality_table(historic_flags)

        # add the accept and reject code attributes:
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'valid_max', int(dfa['QC_accept_code'].values.sum()))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_masks', dfa['QC_accept_code'].values.astype(np.int64))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_meanings', ' '.join(dfa['name'].values))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_codes', ' '.join(dfa['code'].values))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'valid_max', int(dfr['QC_reject_code'].values.sum()))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_masks', dfr['QC_reject_code'].values.astype(np.int64))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_meanings', ' '.join(dfr['name'].values))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_codes', ' '.join(dfr['code'].values))

        # if SAMPLE_TIME is in the output_netcdf_obj, add the units based on the TIME variable
        if 'SAMPLE_TIME' in output_netcdf_obj.variables:
            year_value = profile['TIME'].dt.year.astype(int).values[0]
            dt = datetime(year_value, 1, 1, 0, 0, 0)
            setattr(output_netcdf_obj.variables['SAMPLE_TIME'], 'units', 'milliseconds since ' +
                    dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
        # append the data to the file
        # qc'd
        for v in list(output_netcdf_obj.variables):
            if v not in list(profile) and v not in list(history):
                # if the variable is not in the profile or history or global attributes, skip it, keep fill value
                print(f"Variable {v} not found in profile or history data, skipping.")
                continue
            if v in ['TIME', 'TIME_RAW', 'SAMPLE_TIME']:
                # if the profile[v] is None or contains a string, skip it
                if not ((profile[v].isnull().all()) or (isinstance(profile[v].values[0], str))):
                    time_val_dateobj = date2num(pd.to_datetime(profile[v].values[0]), output_netcdf_obj[v].units,
                                                output_netcdf_obj[v].calendar)
                    output_netcdf_obj[v][:] = time_val_dateobj
                if v == 'TIME':
                    # set the time_coverage_start and time_coverage_end
                    output_netcdf_obj.time_coverage_start = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
                    output_netcdf_obj.time_coverage_end = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
            elif v in list(profile):
                # if all the values of profile[v] are NaN, output the fill value
                if not profile[v].isnull().all():
                    # Check the dimensions of the NetCDF variable
                    var_dims = output_netcdf_obj[v].dimensions

                    # for variables that are dimensioned by DEPTH, output the full array
                    if 'DEPTH' in var_dims:
                        if v not in ['DEPTH']:
                            # fill any NaN values with the fill value for this variable
                            data = profile[v].fillna(output_netcdf_obj[v]._FillValue)
                        else:
                            # for DEPTH, we can directly assign the values
                            data = profile[v].values
                        output_netcdf_obj[v][:] = data
                    else:
                        # just outputting the first value of profile[v] to the netcdf variable
                        # test the shape of the profile[v][0] and the shape of the netcdf variable
                        if not isinstance(profile[v][0], str):
                            # if the profile[v] is a 1D array and the netcdf variable is also a 1D array, assign the first value
                            output_netcdf_obj[v][:] = profile[v].values[0]
                        else:
                            # this is a 1D string variable pad profile[v][0] with empty spaces to match the shape of the netcdf variable
                            padded_shape = output_netcdf_obj[v].shape
                            padded_array = np.full(padded_shape, '', dtype=output_netcdf_obj[v].dtype)
                            # fill the padded array with the profile[v][0] values
                            padded_array[:len(profile[v][0])] = list(profile[v][0])
                            # assign the padded values to the variable
                            output_netcdf_obj[v][:] = padded_array
            elif v in list(history):
                # histories
                if v == 'HISTORY_DATE':
                    # fix history date time field
                    count = 0
                    for ii in history[v]:
                        # if ii is NaN or None, replace with the fill value
                        if pd.isna(ii):
                            output_netcdf_obj[v][count] = output_netcdf_obj[v]._FillValue
                            count += 1
                            continue
                        history_date_obj = date2num(datetime.strptime(str(ii), '%Y-%m-%d %H:%M:%S'),
                                                    output_netcdf_obj['HISTORY_DATE'].units,
                                                    output_netcdf_obj['HISTORY_DATE'].calendar)
                        output_netcdf_obj[v][count] = history_date_obj
                        count += 1
                else:
                    # if the number of dimensions of the variable is 1, we can directly assign the values
                    if len(output_netcdf_obj[v].shape) == 1:
                        output_netcdf_obj[v][:] = history[v].values
                    else:
                        # reshape the history[v] to match the output_netcdf_obj[v] shape and pad with empty spaces if necessary
                        # get the shape of the variable in the netcdf file
                        var_shape = output_netcdf_obj[v].shape
                        # create a padded array with the fill value
                        padded_shape = (len(history[v]),) + var_shape[1:]  # keep the first dimension as the length of history[v]
                        padded_array = np.full(padded_shape, '', dtype=output_netcdf_obj[v].dtype)
                        # fill the padded array with the history[v] values
                        for i, s in enumerate(history[v].values):
                            # if the history[v][i] is None, create an empty string
                            if pd.isna(s):
                                s = ''
                            padded_array[i, :len(s)] = list(s)
                            # assign the padded values to the variable
                        output_netcdf_obj[v][:] = padded_array

            # if v is in extra_atts['attached_var'] then we need to add the attributes to the variable
            if v in extra_atts['attached_var'].values:
                # get the attributes for this variable from the extra_atts DataFrame
                att_extras = extra_atts[extra_atts['attached_var'] == v]
                # loop through the attributes and set them on the variable
                for ind, row in att_extras.iterrows():
                    att_name = row['variable_name']
                    # if the attribute is not in the profile DataFrame, skip it
                    val = profile[att_name][0]
                    if row['variable optional/required (1=required 0=optional)'] == 0 and (
                        pd.isna(val) or val == '' or (isinstance(val, str) and val.strip() == '')):
                        continue
                    if att_name not in profile.columns or pd.isna(profile[att_name].values[0]):
                        setattr(output_netcdf_obj.variables[v], att_name, '')
                    else:
                        # if the profile[att_name] is a timestamp, convert it to a string
                        if pd.api.types.is_datetime64_any_dtype(profile[att_name]):
                            att_value = pd.to_datetime(profile[att_name].values[0]).strftime("%Y%m%d")
                        else:
                            att_value = profile[att_name].values[0]
                        # set the attribute on the variable
                        setattr(output_netcdf_obj.variables[v], att_name, att_value)

        # add geospatial information to global attributes dictionary
        globals_list['geospatial_lat_max'] = profile['LATITUDE'][0]
        globals_list['geospatial_lat_min'] = profile['LATITUDE'][0]
        globals_list['geospatial_lon_max'] = profile['LONGITUDE'][0]
        globals_list['geospatial_lon_min'] = profile['LONGITUDE'][0]
        globals_list['geospatial_vertical_max'] = max(profile['DEPTH'])
        globals_list['geospatial_vertical_min'] = min(profile['DEPTH'])
        # add time coverage information to global attributes dictionary
        globals_list['time_coverage_start'] = profile['TIME'][0].strftime("%Y-%m-%dT%H:%M:%SZ")
        globals_list['time_coverage_end'] = profile['TIME'][0].strftime("%Y-%m-%dT%H:%M:%SZ")
        # add transect_id to global attributes dictionary
        globals_list['transect_id'] = profile['transect_id'][0]

        # add extra global attributes from the extra_globals list
        for ind, row in extra_globals.iterrows():
            att_name = row['variable_name']
            # if the attribute is not in the profile DataFrame, skip it
            val = profile[att_name][0]
            if row['variable optional/required (1=required 0=optional)'] == 0 and (
                    pd.isna(val) or val == '' or (isinstance(val, str) and val.strip() == '')):
                continue
            # if the attribute is in the globals_list, use that value
            if att_name in profile.columns:
                # if the attribute is in the profile DataFrame, use that value
                globals_list[att_name] = profile[att_name].values[0]

        # Add date created to the global attributes
        utctime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        globals_list['date_created'] = utctime

        # set the global attributes where the index is the attribute name
        for att_name, att_value in globals_list.items():
            # if the global_att[att_name] is None, replace with 'Unknown'
            if pd.isna(att_value):
                att_value = 'Unknown'
            output_netcdf_obj.setncattr(att_name, att_value)

# main function
if __name__ == '__main__':
    """
    Example: python write2netcdf.py -i /path/to/input/ -o /path/to/output/ -g /path/to/globals_file
    """
    # parse the arguments
    parser = argparse.ArgumentParser(description="Convert XBT data to IMOS format netcdf")
    parser.add_argument("-i", "--input", help="Path to the input folder", required=True)
    parser.add_argument("-o", "--output", help="Path to the output folder", required=True)
    # add globals file input argument
    parser.add_argument("-g", "--globals",help="Path to globals information", required=True)
    args = parser.parse_args()

    # get the input and output folders
    input_folder = args.input
    output_folder = args.output
    globals_input_file = args.globals
    globals_configs = preload_globals_configs(globals_input_file)
    # add subscript '_oceantrax' to the output folder for oceantrax format files
    output_folder_oceantrax = output_folder.rstrip('/') + '_oceantrax'

    # locate the parquet files in the input folder not including the *histories.parquet and *globals.parquet files
    parquet_data = glob.glob(os.path.join(input_folder, "*.parquet"))
    parquet_data = [f for f in parquet_data if "_globals" not in f]
    parquet_data = [f for f in parquet_data if "_histories" not in f]

    # and the history parquet files
    parquet_history = glob.glob(os.path.join(input_folder, "*histories.parquet"))

    # Track successful profile exports per SOOP line label.
    successful_exports_by_line = defaultdict(int)

    # write the output netcdf files
    for data_file in parquet_data:
        print("Processing file %s" % data_file)
        # read the parquet file
        profiles = pd.read_parquet(data_file)
        histories = pd.read_parquet(data_file.replace(".parquet", "_histories.parquet"))
        # remove the HISTORY_PREVIOUS_VALUE column from the histories dataframe if it exists as it is not needed for the netcdf output
        if 'HISTORY_PREVIOUS_VALUE' in histories.columns:
            histories = histories.drop(columns=['HISTORY_PREVIOUS_VALUE'])
        # put a fix in here for already made parquet files where we have changed the column name from PROBE_manufacture_date to PROBE_manufacture_date_YYYYMMDD
        if 'PROBE_manufacture_date' in profiles.columns:
            profiles = profiles.rename(columns={'PROBE_manufacture_date': 'PROBE_manufacture_date_YYYYMMDD'})
        if 'PROBE_manufacture_date_YYYY-MM-DD' in profiles.columns:
            profiles = profiles.rename(columns={'PROBE_manufacture_date_YYYY-MM-DD': 'PROBE_manufacture_date_YYYYMMDD'})
        # sort the dataframes by line label and TIME
        profiles = profiles.sort_values(by=['SOOP_line_label', 'TIME', 'DEPTH']).reset_index(drop=True)
        # get the station_number order from profiles and apply it to histories so that they are in the same order
        station_number_order = profiles['station_number'].unique()
        histories['station_number'] = pd.Categorical(histories['station_number'], categories=station_number_order, ordered=True)
        histories = histories.sort_values('station_number').reset_index(drop=True)
        # make an empty transect_id column in the profiles dataframe
        profiles['transect_id'] = None
        # create a transect_id column in the profiles dataframe by concatenating the SOOP_line_label and the year of the TIME column
        for soop_line_label in profiles['SOOP_line_label'].unique():
            # use groupby to group the profiles by Cruise_ID, keeping them in the same order and assign a transect_id to each group based on the SOOP_line_label and the year of the TIME column
            count = 0
            for cruise_id, group in profiles[profiles['SOOP_line_label'] == soop_line_label].groupby('Cruise_ID', sort=False):
                count += 1
                transect_id = make_transect_id(soop_line_label, group['TIME'].iloc[0], count)
                profiles.loc[group.index, 'transect_id'] = transect_id
        # there are multiple profiles in the profiles dataframe, loop through unique station numbers
        for station in profiles['station_number'].unique():
            # get the profile and history data for this station
            profile = profiles[profiles['station_number'] == station].reset_index()
            profile_histories = histories[histories['station_number'] == station].reset_index()

            # add some paths to the output_folder based on the 'SOOP_line_label' and year of the profile time
            line_label = profile['SOOP_line_label'][0]
            year = profile['TIME'][0].year
            profile_globals_attrs = select_imos_globals_config(globals_configs, profile)

            # write the profile to the netcdf file in imos format, then in oceantrax format
            for output_format in ['imos', 'oceantrax']:
                if output_format == 'imos':
                    # output folder is output_folder/line_label/year
                    output_folder_line_year = os.path.join(output_folder, line_label, str(year))
                    if not os.path.exists(output_folder_line_year):
                        os.makedirs(output_folder_line_year)
                    write_output_nc(output_folder_line_year, profile, profile_histories, globals_attrs=profile_globals_attrs, profile_raw=False, historic_flags=True, imosformat=True)
                elif output_format == 'oceantrax':
                    continue  # skip oceantrax output for now, as it is not needed for the current use case
                    # output folder is output_folder/line_label/year
                    output_folder_line_year = os.path.join(output_folder_oceantrax, line_label, str(year))
                    if not os.path.exists(output_folder_line_year):
                        os.makedirs(output_folder_line_year)
                    write_output_nc(output_folder_line_year, profile, profile_histories, globals_attrs=globals_configs['oceantrax'], profile_raw=False, historic_flags=True, imosformat=False)
                else:
                    raise ValueError(f"Unknown output format: {output_format}")
            successful_exports_by_line[line_label] += 1

    print("\nSuccessful profile exports by SOOP line:")
    if successful_exports_by_line:
        for line_label in sorted(successful_exports_by_line):
            print(f"{line_label}: {successful_exports_by_line[line_label]}")
    else:
        print("No profiles were exported.")