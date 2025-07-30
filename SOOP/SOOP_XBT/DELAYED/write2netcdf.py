# import a parquet file and write it to a netcdf file
import argparse
import glob
import os
from datetime import datetime
from time import strftime, gmtime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num

from xbt_parse import read_section_from_xbt_config
from xbt_utils import read_flag_quality_table, read_variables_config, read_globals_config

def create_filename_output(prof, hist, profile_raw=False):
    if profile_raw:
        fv = 'FV00'
    else:
        fv = 'FV01'

    filename = 'XBT_T_%s_%s_%s_ID-%s' % (
        prof['TIME'].strftime('%Y%m%dT%H%M%SZ'), prof['SOOP_line_label'], fv,
        prof['Institution_unique_identifier'])

    # decide what prefix is required
    names = read_section_from_xbt_config('VARIOUS')
    str = names['FILENAME']
    if str == 'Cruise_ID':
        str = prof['Cruise_ID']
        filename = '{}-{}'.format(str, filename)
    else:
        if prof['TIME'] > datetime(2008, 0o1, 0o1):
            filename = 'IMOS_SOOP-{}'.format(filename)

    # if profile histories contains TP, change the filename
    if 'TPR' in hist['HISTORY_QC_CODE'].values:
        filename = filename.replace('XBT', 'TEST')

    return filename


def write_output_nc(output_folder, profile, history, profile_raw=False, historic_flags=False):
    """output the data to the IMOS format netcdf version
    :param output_folder: the folder to write the netcdf file to
    :param profile: the profile DataFrame
    :param history: the history DataFrame
    :param profile_raw: if True, the create a FV00 file, if False create a FV01 file, default is False
    """

    # now begin write out to new format
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(profile.iloc[0], history, profile_raw))
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
    # read the global attributes config file
    globals_list = read_globals_config()
    # first get a list of the attributes attached to the variables
    extra_atts = vars[vars['is_var_att_global'] == 'att']
    # get a list of the global attributes
    extra_globals = vars[vars['is_var_att_global'] == 'global']
    # create a list from extra_globals['variable_name'] to use as a list of global attributes
    extra_globals = extra_globals['variable_name'].tolist()
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
            if v in ['TIME', 'TIME_RAW','PROBE_manufacture_date', 'SAMPLE_TIME']:
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
                    # Check the shape of the NetCDF variable
                    var_shape = output_netcdf_obj[v].shape

                    # Ensure the data from profile[v] matches the shape of the NetCDF variable
                    if profile[v].shape == var_shape:
                        # fill any NaN values with the fill value for this variable
                        data = profile[v].fillna(output_netcdf_obj[v]._FillValue)
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
                    if row['variable optional/required (1=required 0=optional)'] == 0 and profile[att_name].isnull().all():
                        continue
                    if att_name not in profile.columns or pd.isna(profile[att_name].values[0]):
                        setattr(output_netcdf_obj.variables[v], att_name, '')
                    else:
                        # if the profile[att_name] is a timestamp, convert it to a string
                        if pd.api.types.is_datetime64_any_dtype(profile[att_name]):
                            att_value = pd.to_datetime(profile[att_name].values[0]).strftime("%Y-%m-%d")
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

        # add extra global attributes from the extra_globals list
        for att_name in extra_globals:
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
    Example: python write2netcdf.py -i /path/to/input/ -o /path/to/output/
    """
    # parse the arguments
    parser = argparse.ArgumentParser(description="Convert XBT data to IMOS format netcdf")
    parser.add_argument("-i", "--input", help="Path to the input folder", required=True)
    parser.add_argument("-o", "--output", help="Path to the output folder", required=True)
    args = parser.parse_args()

    # get the input and output folders
    input_folder = args.input
    output_folder = args.output

    # if output folder doesn't exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # locate the parquet files in the input folder not including the *histories.parquet and *globals.parquet files
    parquet_data = glob.glob(os.path.join(input_folder, "*.parquet"))
    parquet_data = [f for f in parquet_data if "_globals" not in f]
    parquet_data = [f for f in parquet_data if "_histories" not in f]

    # and the history parquet files
    parquet_history = glob.glob(os.path.join(input_folder, "*histories.parquet"))

    # write the output netcdf files
    for data_file in parquet_data:
        # read the parquet file
        profiles = pd.read_parquet(data_file)
        histories = pd.read_parquet(data_file.replace(".parquet", "_histories.parquet"))

        # there are multiple profiles in the profiles dataframe, loop through unique station numbers
        for station in profiles['station_number'].unique():
            # get the profile and history data for this station
            profile = profiles[profiles['station_number'] == station]
            profile_histories = histories[histories['station_number'] == station]

            # write the profile to the netcdf file
            write_output_nc(output_folder, profile, profile_histories,profile_raw=False, historic_flags=True)