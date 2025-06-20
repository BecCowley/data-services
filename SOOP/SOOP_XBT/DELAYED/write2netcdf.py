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

def create_filename_output(prof, hist, profile_raw=False):
    if profile_raw:
        fv = 'FV00'
    else:
        fv = 'FV01'

    filename = 'XBT_T_%s_%s_%s_ID-%s' % (
        prof['TIME'].strftime('%Y%m%dT%H%M%SZ'), prof['SOOP_line'], fv,
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
    if 'TPR' in hist['HISTORY_QC_CODE'].values:
        filename = filename.replace('XBT', 'TEST')

    return filename


def write_output_nc(output_folder, profile, history, global_atts, profile_raw=False, historic_flags=False):
    """output the data to the IMOS format netcdf version
    :param output_folder: the folder to write the netcdf file to
    :param profile: the profile DataFrame
    :param history: the history DataFrame
    :param global_atts: the global attributes dictionary
    :param profile_raw: if True, the create a FV00 file, if False create a FV01 file, default is False
    """

    # now begin write out to new format
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(profile.iloc[0], history, profile_raw))
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
            # check if there is data in the profile DataFrame for this variable. Required variables must be kept in the output netcdf file
            if vv not in profile.columns and vv not in history.columns and row['variable optional/required (1=required 0=optional)'] == 0:
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
            if v not in list(profile) and v not in list(history) and v not in list(global_atts.keys()):
                # if the variable is not in the profile or history or global attributes, skip it, keep fill value
                print(f"Variable {v} not found in profile or history data, skipping.")
                continue
            if v in ['TIME', 'TIME_RAW','PROBE_manufacture_date', 'SAMPLE_TIME']:
                # if the profile[v] is None, skip it
                if profile[v].isnull().all():
                    continue
                time_val_dateobj = date2num(pd.to_datetime(profile[v].values[0]), output_netcdf_obj[v].units,
                                            output_netcdf_obj[v].calendar)
                output_netcdf_obj[v][:] = time_val_dateobj
                if v == 'TIME':
                    # set the time_coverage_start and time_coverage_end
                    output_netcdf_obj.time_coverage_start = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
                    output_netcdf_obj.time_coverage_end = pd.to_datetime(profile[v].values[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
            elif v in list(profile):
                # if all the values of profile[v] are NaN, skip it
                if profile[v].isnull().all():
                    continue
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

        # Add date created to the global attributes
        utctime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        global_atts['date_created'] = utctime

        # set the global attributes where the index is the attribute name
        for att_name, att_value in global_atts.items():
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
        global_atts = pd.read_parquet(data_file.replace(".parquet", "_globals.parquet"))

        # there are multiple profiles in the profiles dataframe, loop through unique station numbers
        for station in profiles['station_number'].unique():
            # get the profile and history data for this station
            profile = profiles[profiles['station_number'] == station]
            profile_histories = histories[histories['station_number'] == station]
            profile_global_atts = global_atts[global_atts['station_number'] == station]
            # and remove the station_number column
            profile_global_atts = profile_global_atts.drop(columns='station_number')
            # convert the global attributes to a dictionary
            profile_global_atts = profile_global_atts.to_dict(orient='records')[0]

            # write the profile to the netcdf file
            write_output_nc(output_folder, profile, profile_histories, profile_global_atts,profile_raw=False, historic_flags=True)