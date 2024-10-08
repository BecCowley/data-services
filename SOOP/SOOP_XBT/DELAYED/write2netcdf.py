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

def create_filename_output(prof, hist, global_atts):
    filename = 'XBT_T_%s_%s_FV01_ID-%s' % (
        prof['TIME'].strftime('%Y%m%dT%H%M%SZ'), global_atts['XBT_line'].values[0],
        global_atts['XBT_uniqueid'].values[0])

    # decide what prefix is required
    names = read_section_from_xbt_config('VARIOUS')
    str = names['FILENAME']
    if str == 'Cruise_ID':
        str = global_atts['XBT_cruise_ID']
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
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(profile.iloc[0], history, global_atts))
    print('Creating output %s' % netcdf_filepath)

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(profile['DEPTH']))
        output_netcdf_obj.createDimension('N_HISTORY', 0) #make this unlimited

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
        varslist = [key for key in profile.keys() if ('_quality_control' not in key and 'RAW' not in key
                                                           and 'TUDE' not in key and 'XBT' not in key
                                                           and 'TIME' not in key and 'uncertainty' not in key
                                                           and 'PROBE' not in key and 'station_number' not in key)]
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
            if profile[vv + '_RAW_quality_control'].any() > 0:
                print("QC values have been written to file for \"%s\"_RAW variable. Review." % vv)
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

        accept_codes = output_netcdf_obj.createVariable("XBT_accept_code", "u8", dimensions=('DEPTH',),
                                                  fill_value=0)
        reject_codes = output_netcdf_obj.createVariable("XBT_reject_code", "u8", dimensions=('DEPTH',),
                                                    fill_value=0)

        # # If the turo profile is handed in:
        # if profile_raw is not None:
        #     output_netcdf_obj.createVariable("RESISTANCE", "f", dimensions=('DEPTH',), fill_value=float("nan"))
        #     output_netcdf_obj.createVariable("SAMPLE_TIME", "f", dimensions=('DEPTH',), fill_value=float("nan"))

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

        # set up a dataframe of the codes and their values
        # codes from the new cookbook, read from csv file
        # Specify the file path
        a_file_path = os.path.join(os.path.dirname(__file__), 'xbt_accept_code.csv')
        r_file_path = os.path.join(os.path.dirname(__file__), 'xbt_reject_code.csv')

        # Read the CSV file and convert it to a DataFrame
        dfa = pd.read_csv(a_file_path)
        dfr = pd.read_csv(r_file_path)

        # add the accept and reject code attributes:
        setattr(accept_codes, 'valid_max', int(dfa['byte_value'].values.sum()))
        setattr(accept_codes, 'flag_masks', dfa['byte_value'].values.astype(np.uint64))
        setattr(accept_codes, 'flag_meanings', ' '.join(dfa['label'].values))
        setattr(accept_codes, 'flag_codes', ' '.join(dfa['code'].values))
        setattr(reject_codes, 'valid_max', int(dfr['byte_value'].values.sum()))
        setattr(reject_codes, 'flag_masks', dfr['byte_value'].values.astype(np.uint64))
        setattr(reject_codes, 'flag_meanings', ' '.join(dfr['label'].values))
        setattr(reject_codes, 'flag_codes', ' '.join(dfr['code'].values))

        # write coefficients out to the attributes. In the PROBE_TYPE, PROBE_TYPE_RAW, DEPTH, DEPTH_RAW
        varnames = ['PROBE_TYPE', 'DEPTH']
        for v in varnames:
            setattr(output_netcdf_obj.variables[v], 'XBT_fallrate_coefficients',
                    global_atts['XBT_fallrate_equation_coefficients'].values[0])
            setattr(output_netcdf_obj.variables[v], 'probe_type_name', global_atts['PROBE_TYPE_name'].values[0])

        varnames = ['PROBE_TYPE_RAW', 'DEPTH_RAW']
        for v in varnames:
            setattr(output_netcdf_obj.variables[v], 'fallrate_coefficients',
                    global_atts['XBT_fallrate_equation_coefficients_RAW'].values[0])
            setattr(output_netcdf_obj.variables[v], 'probe_type_name', global_atts['PROBE_TYPE_RAW_name'].values[0])

        # append the data to the file
        # qc'd
        for v in list(output_netcdf_obj.variables):
            if v not in list(profile) and v not in list(history) and v not in list(global_atts):
                print("Variable not written: \"%s\". Please check!!" % v)
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
                        output_netcdf_obj[v][len(profile[v])] = profile[v].values[0]
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