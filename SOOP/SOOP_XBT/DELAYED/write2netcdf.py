# import a parquet file and write it to a netcdf file
import argparse
import glob
import os
from datetime import datetime
from time import strftime, gmtime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num

from generate_netcdf_att import get_imos_parameter_info, generate_netcdf_att
from xbt_parse import read_section_from_xbt_config
from xbt_utils import read_flag_quality_table, read_variables_config, read_globals_config

def create_filename_output(prof, hist):
    filename = 'XBT_T_%s_%s_FV01_ID-%s' % (
        prof['TIME'].strftime('%Y%m%dT%H%M%SZ'), prof['SOOP_line'],
        prof['Institution_uniqueid'])

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
    if 'TP' in hist['HISTORY_QC_CODE'].values:
        filename = filename.replace('XBT', 'TESTPROBE')

    return filename


def write_output_nc(output_folder, profile, history, global_atts, profile_raw=None):
    """output the data to the IMOS format netcdf version"""

    # now begin write out to new format
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(profile.iloc[0], history))
    print('Creating output %s' % netcdf_filepath)

    # read the variables config file
    vars = read_variables_config()
    # Identify attribute columns starting with 'att_'
    att_cols = [col for col in vars.columns if col.startswith('att_')]
    # remove the 'att_' prefix from the attribute columns
    att_labels = [col.replace('att_', '') for col in att_cols]

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(profile['DEPTH']))
        output_netcdf_obj.createDimension('N_HISTORY', 0) #make this unlimited

        # Create the variables from the vars Dataframe by looping through the rows
        for index, row in vars.iterrows():
            vv = row['variable_name']
            # print(vv)
            # check if there is data in the profile DataFrame for this variable
            if vv not in profile.columns and vv not in history.columns:
                # if not, skip this variable
                print(f"Variable {vv} not found in profile data, skipping.")
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
                output_netcdf_obj.createVariable(vv, datatype=dt, dimensions=dimensions)
            else:
                # create the variable in the netcdf file with dimensions and fill value
                output_netcdf_obj.createVariable(vv, datatype=dt, fill_value=fillvalue, dimensions=dimensions)
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
                            row[att_name] = np.array([np.byte(int(x.strip())) for x in row[att_name].split(' ')])
                    # set the attribute on the variable
                    setattr(output_netcdf_obj.variables[vv], att, row[att_name])

        # read the flag quality tables
        dfa, dfr = read_flag_quality_table()

        # add the accept and reject code attributes:
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'valid_max', int(dfa['QC_accept_code'].values.sum()))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_masks', dfa['QC_accept_code'].values.astype(np.uint64))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_meanings', ' '.join(dfa['name'].values))
        setattr(output_netcdf_obj.variables['QC_accept_code'], 'flag_codes', ' '.join(dfa['code'].values))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'valid_max', int(dfr['QC_reject_code'].values.sum()))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_masks', dfr['QC_reject_code'].values.astype(np.uint64))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_meanings', ' '.join(dfr['name'].values))
        setattr(output_netcdf_obj.variables['QC_reject_code'], 'flag_codes', ' '.join(dfr['code'].values))

        # if SAMPLE_TIME is in the output_netcdf_obj, add the units based on the TIME variable
        if 'SAMPLE_TIME' in output_netcdf_obj.variables:
            year_value = profile['TIME'].dt.year.astype(int).values[0]
            dt = datetime.datetime(year_value, 1, 1, 0, 0, 0)
            setattr(output_netcdf_obj.variables['SAMPLE_TIME'], 'units', 'milliseconds since ' +
                    dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # add the global attributes
        global_list = read_globals_config()
        for index, row in global_list.iterrows():
            if pd.notna(row['Attribute Value']):
                setattr(output_netcdf_obj, row['Attribute Name'], row['Attribute Value'])
            else:
                # check for information in the global_atts DataFrame
                if row['Attribute Name'] in global_atts.columns:
                    setattr(output_netcdf_obj, row['Attribute Name'], global_atts[row['Attribute Name']].values[0])
                # check for information in the profile DataFrame
                # check for a profile.columns name match including upper and lower case
                elif row['Attribute Name'].lower() in profile.columns.str.lower().tolist():
                    # get the first match
                    matched_col = profile.columns[profile.columns.str.lower() == row['Attribute Name'].lower()][0]
                    setattr(output_netcdf_obj, row['Attribute Name'], profile[matched_col].values[0])
                # print a warning if the attribute is not found
                else:
                    # if this is date_created, set it to the current time
                    if row['Attribute Name'] == 'date_created':
                        setattr(output_netcdf_obj, row['Attribute Name'], strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))
                    else:
                        print(f"Global attribute {row['Attribute Name']} not found in profile or global attributes, skipping.")

        # append the data to the file
        # qc'd
        for v in list(output_netcdf_obj.variables):
            if v not in list(profile) and v not in list(history) and v not in list(global_atts):
                print("Variable not written: \"%s\". Please check!!" % v)
                continue
            if v in ['TIME', 'TIME_RAW','PROBE_manufacture_date', 'SAMPLE_TIME']:
                time_val_dateobj = date2num(pd.to_datetime(profile[v].values[0]), output_netcdf_obj[v].units,
                                            output_netcdf_obj[v].calendar)
                output_netcdf_obj[v][:] = time_val_dateobj
                if v == 'TIME':
                    # set the time_coverage_start and time_coverage_end
                    output_netcdf_obj.time_coverage_start = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
                    output_netcdf_obj.time_coverage_end = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
            elif v in list(profile):
                # Check the shape of the NetCDF variable
                var_shape = output_netcdf_obj[v].shape

                # Ensure the data from profile[v] matches the shape of the NetCDF variable
                if profile[v].shape == var_shape:
                        output_netcdf_obj[v][:] = profile[v]
                else:
                    if isinstance(output_netcdf_obj[v][:], str):
                        output_netcdf_obj[v][0] = str(profile[v].values[0])
                    else:
                        output_netcdf_obj[v][:] = profile[v].values[0]
            else:
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
                    output_netcdf_obj[v][:] = history[v].values

        # first remove all the columns in global_atts that end in '_RAW' as these are not required
        global_atts = global_atts.loc[:, ~global_atts.columns.str.contains('_RAW')]
        # and remove the station_number column
        global_atts = global_atts.drop(columns='station_number')

        # write out the extra global attributes we have collected
        for key, item in global_atts.items():
            if item.values[0] is not None:
                setattr(output_netcdf_obj, key, item.values[0])
        # Add date created
        utctime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        output_netcdf_obj.date_created = utctime

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
        global_atts = pd.read_parquet(data_file.replace(".parquet", "_globals.parquet"))

        # there are multiple profiles in the profiles dataframe, loop through unique station numbers
        for station in profiles['station_number'].unique():
            # get the profile and history data for this station
            profile = profiles[profiles['station_number'] == station]
            profile_histories = histories[histories['station_number'] == station]
            profile_global_atts = global_atts[global_atts['station_number'] == station]
            # write the profile to the netcdf file
            write_output_nc(output_folder, profile, profile_histories, profile_global_atts)