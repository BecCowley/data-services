# import a parquet file and write it to a netcdf file
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num

from generate_netcdf_att import get_imos_parameter_info, generate_netcdf_att
from xbt_parse import read_section_from_xbt_config

def create_filename_output(prof, hist):
    filename = 'XBT_T_%s_%s_FV01_ID-%s' % (
        prof['TIME'].strftime('%Y%m%dT%H%M%SZ'), prof['XBT_line'],
        prof['XBT_uniqueid'])

    # decide what prefix is required
    names = read_section_from_xbt_config('VARIOUS')
    str = names['FILENAME']
    if str == 'Cruise_ID':
        str = prof['XBT_cruise_ID']
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

    # TODO: create groups in the netcdf file. Group of data, group of histories and group of variables that were previously in the global attributes

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(profile['DEPTH']))
        output_netcdf_obj.createDimension('N_HISTORY', 0) #make this unlimited

        # Create the variables, no dimensions:
        # varslist = ["TIME", "LATITUDE", "LONGITUDE", "PROBE_TYPE"]
        varslist = [key for key in profile.keys()]
        for vv in varslist:
            # first check if this variable is in the imosParameters.txt file
            dt = get_imos_parameter_info(vv, '__data_type')
            fillvalue = get_imos_parameter_info(vv, '_FillValue')
            if fillvalue == '':
                fillvalue = None
            if dt:
                if vv in ['TIME', 'LATITUDE', 'LONGITUDE', 'PROBE_TYPE']:
                    output_netcdf_obj.createVariable(vv, datatype=dt, fill_value=fillvalue)
                    # and associated QC variables:
                    output_netcdf_obj.createVariable(vv + "_quality_control", "b", fill_value=99)
                    # and the *_RAW variables:
                    output_netcdf_obj.createVariable(vv + "_RAW", datatype=dt, fill_value=fillvalue)
                # create dimensioned variables:
                if vv in ['XBT_accept_code', 'XBT_reject_code']:
                    output_netcdf_obj.createVariable(vv, datatype=dt, dimensions=('DEPTH',), fill_value=fillvalue)
                if vv in ['DEPTH', 'TEMP', 'PSAL', 'COND', 'RESISTANCE', 'SAMPLE_TIME']:
                    output_netcdf_obj.createVariable(vv, datatype=dt, dimensions=('DEPTH',), fill_value=fillvalue)
                    # and associated QC variables:
                    output_netcdf_obj.createVariable(vv + "_quality_control", "b", dimensions=('DEPTH',), fill_value=99)
                    # and the *_RAW variables:
                    output_netcdf_obj.createVariable(vv + "_RAW", datatype=dt,
                                                 dimensions=('DEPTH',), fill_value=fillvalue)
                    if vv in ['TEMP', 'DEPTH', 'PSAL']:
                        # add the uncertainty variable
                        output_netcdf_obj.createVariable(vv + "_uncertainty", datatype=dt, dimensions=('DEPTH',),
                                                        fill_value=fillvalue)
                # test if the output_netCDF_obj already has the variable created
                if vv not in output_netcdf_obj.variables:
                    output_netcdf_obj.createVariable(vv, datatype=dt, fill_value=fillvalue)
            elif vv not in output_netcdf_obj.variables:
                print("Variable skipped: \"%s\". Please check!!" % vv)

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
        output_netcdf_obj.createVariable("HISTORY_QC_CODE_VALUE", "f", 'N_HISTORY')

        # write attributes from the generate_nc_file_att file, now that we have added the variables:
        conf_file = os.path.join(os.path.dirname(__file__), 'generate_nc_file_att')
        generate_netcdf_att(output_netcdf_obj, conf_file, conf_file_point_of_truth=True)

        # set up a dataframe of the codes and their values
        # codes from the new cookbook, read from csv file
        # Specify the file path
        a_file_path = os.path.join(os.path.dirname(__file__), 'xbt_accept_code.csv')
        r_file_path = os.path.join(os.path.dirname(__file__), 'xbt_reject_code.csv')

        # Read the CSV file and convert it to a DataFrame
        dfa = pd.read_csv(a_file_path)
        dfr = pd.read_csv(r_file_path)

        # add the accept and reject code attributes:
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'valid_max', int(dfa['byte_value'].values.sum()))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_masks', dfa['byte_value'].values.astype(np.uint64))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_meanings', ' '.join(dfa['label'].values))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_codes', ' '.join(dfa['code'].values))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'valid_max', int(dfr['byte_value'].values.sum()))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_masks', dfr['byte_value'].values.astype(np.uint64))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_meanings', ' '.join(dfr['label'].values))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_codes', ' '.join(dfr['code'].values))

        # if SAMPLE_TIME is in the output_netcdf_obj, add the units
        if 'SAMPLE_TIME' in output_netcdf_obj.variables:
            year_value = profile['TIME'].dt.year.astype(int).values[0]
            dt = datetime.datetime(year_value, 1, 1, 0, 0, 0)
            setattr(output_netcdf_obj.variables['SAMPLE_TIME'], 'units', 'milliseconds since ' +
                    dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # append the data to the file
        # qc'd
        for v in list(output_netcdf_obj.variables):
            if v not in list(profile) and v not in list(history) and v not in list(global_atts):
                print("Variable not written: \"%s\". Please check!!" % v)
                continue
            if v in ['TIME', 'TIME_RAW','XBT_manufacturer_date', 'SAMPLE_TIME']:
                time_val_dateobj = date2num(pd.to_datetime(profile[v].values[0]), output_netcdf_obj[v].units,
                                            output_netcdf_obj[v].calendar)
                output_netcdf_obj[v][:] = time_val_dateobj
            elif v in list(profile):
                # Check the shape of the NetCDF variable
                var_shape = output_netcdf_obj[v].shape

                # Ensure the data from profile[v] matches the shape of the NetCDF variable
                if profile[v].shape == var_shape:
                        output_netcdf_obj[v][:] = profile[v]
                else:
                    if isinstance(output_netcdf_obj[v][:], str):
                        output_netcdf_obj[v][len(profile[v])] = str(profile[v].values[0])
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