# Converts XBT profile recorded by Turo XBT to standardised netCDF format ready for QC with PYQUEST
# Rebecca Cowley, CSIRO, February, 2025
# Adapted from code by A. Walsh V2 4/10/22

import argparse
import difflib

####Usage####
# python TuroXBT2IMOSnc.py -i xbtdata_raw_folder -o xbtdata_output_folder
# xbtdata_raw = input folder holding raw files from Turo XBT - dropXXX.nc
# xbtdata_output_folder = output folder to hold files produced by this script

# Example:
# python /path/to/data-services/SOOP/SOOP_XBT/DELAYED/TuroXBT2IMOSnc.py -i RD3203 -o IMOSformatnc

import os
import re
import sys
import tempfile

import xarray as xr
from netCDF4 import Dataset
import datetime
from time import localtime, gmtime, strftime
from netCDF4 import date2num
import numpy as np
import glob
import pandas as pd

from xbt_line_vocab import xbt_line_info
from xbt_parse import read_section_from_xbt_config
from generate_netcdf_att import generate_netcdf_att, get_imos_parameter_info
from ship_callsign import ship_callsign_list
from imos_logging import IMOSLogging
from xbt_utils import read_qc_config


def args():
    """ define input argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-xbt-path', type=str,
                        help="path to Turo netcdf files")
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

    if not os.path.exists(vargs.input_xbt_path):
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

    # ship details from AODN vocabs
    global SHIPS
    SHIPS = ship_callsign_list()

    # get the probe type and fre list from the config file
    global fre_list, peq_list
    fre_list = read_section_from_xbt_config('FRE')
    peq_list = read_section_from_xbt_config('PEQ$')

def create_out_filename(profile, line, crid, n, test):
    # create the unique ID from the crid, time and drop number formatted to three digits
    uniqueid = crid + '_' + profile.time.dt.strftime('%Y%m%d%H%M%S').values[0] + '_' + str(n).zfill(3)

    if test:
        filename = 'XBTTEST_T_%s_%s_FV01_ID-%s.nc' % (line, profile.time.dt.strftime('%Y%m%d%H%M%SZ').values[0], uniqueid)
    else:
        filename = 'IMOS_SOOP-XBT_T_%s_%s_FV01_ID-%s.nc' % (line, profile.time.dt.strftime('%Y%m%d%H%M%SZ').values[0], uniqueid)

    return filename, uniqueid


def create_flag_feature():
    """ Take the existing QC code values and turn them into a integer representation. One bit for every code."""

    # set up a dataframe of the codes and their values
    # codes from the new cookbook, read from csv file
    # Specify the file path
    a_file_path = 'xbt_accept_code.csv'
    r_file_path = 'xbt_reject_code.csv'

    # Read the CSV file and convert it to a DataFrame
    dfa = pd.read_csv(os.path.join(os.path.dirname(__file__),a_file_path))
    dfr = pd.read_csv(os.path.join(os.path.dirname(__file__),r_file_path))

    # remove nan values
    dfa = dfa.dropna(subset=['byte_value'])
    # remove the tempqc column
    dfa = dfa.drop(columns=['tempqc'])
    # remove nan values
    dfr = dfr.dropna(subset=['byte_value'])
    # remove the tempqc column
    dfr = dfr.drop(columns=['tempqc'])

    return dfa, dfr


def add_uncertainties(nco):
    """ return the profile with added uncertainties"""

    # use standard uncertainties assigned by IQuOD procedure:
    # XBT manufacturers other than Sippican and TSK and unknown manufacturer / type:  0.2;  <= 230m: 4.6m; > 230 m: 2%
    # XBT deployed from submarines or Tsurumi - Seiki Co(TSK) manufacturer 0.15;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT Sippican manufacturer 0.1;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT deployed from aircraft 0.056
    # XCTD(pre - 1998) 0.06; 4 %
    # XCTD(post - 1998) 0.02; 2 %

    pt = int(nco.Code)
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
        year_value = nco.time.dt.year.astype(int).values[0]
        dti = datetime.datetime(year_value, 1, 1, 0, 0, 0)
        if dti < datetime.datetime.strptime('1998-01-01', '%Y-%m-%d'):
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
    temp_uncertainty = np.ma.empty_like(nco.temperature)
    temp_uncertainty[:] = tunc
    # depth uncertainties:
    unc = np.ma.MaskedArray(nco.depth * dunc[0], mask=False)
    if len(dunc) > 1:
        unc[nco.depth <= 230] = dunc[1]
    depth_uncertainty = np.round(unc, 2)

    return temp_uncertainty, depth_uncertainty


def get_recorder_type(nco):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')

    item_val = str(int(nco.InterfaceCode))
    if item_val in list(rct_list.keys()):
        return item_val, rct_list[item_val].split(',')[0]
    else:
        LOGGER.warning(
            '{item_val} missing from recorder type part in xbt_config file, using unknown for recorder'.format(
                item_val=item_val))
        item_val = '99'
        return item_val, rct_list[item_val].split(',')[0]


def netCDFout(nco, n, crid, callsign, ship_IMO, ship_name, line_info, raw_netCDF_file):

    # create the output file name
    test = False
    if nco.TestCanister == 'yes':
        test = True
    outfile, unique_id = create_out_filename(nco, line_info[0], crid, n, test)
    outfile = os.path.join(vargs.output_folder, outfile)

    # First, get a list of variables mapped between nco and output_netcdf_obj
    varslist = read_section_from_xbt_config('Turo_variables')
    # create a ncobject to write out to new format
    with Dataset(outfile, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(nco.depth))
        output_netcdf_obj.createDimension('N_HISTORY', 0)  # make this unlimited

        # Create the variables, no dimensions:
        for vv in list(varslist.values()):
            # get the variable attributes from imosParameters.txt
            dttyp = get_imos_parameter_info(vv, '__data_type')
            fillvalue = get_imos_parameter_info(vv, '_FillValue')
            if fillvalue == '':
                fillvalue = None
            if dttyp:
                if vv in ['TIME', 'LATITUDE', 'LONGITUDE', 'PROBE_TYPE']:
                    output_netcdf_obj.createVariable(vv, datatype=dttyp, fill_value=fillvalue)
                    # and associated QC variables:
                    output_netcdf_obj.createVariable(vv + "_quality_control", "b", fill_value=99)
                    # and the *_RAW variables:
                    output_netcdf_obj.createVariable(vv + "_RAW", datatype=dttyp, fill_value=fillvalue)
                if vv in ['XBT_recorder_type', 'PROBE_TYPE']:
                    # add the *_name variable
                    output_netcdf_obj.createVariable(vv + "_name", "str", fill_value=fillvalue)
                    # for PROBE_TYPE also add PROBE_TYPE_RAW_name, *_coef_a, *_coef_b
                    if vv == 'PROBE_TYPE':
                        output_netcdf_obj.createVariable(vv + "_RAW_name", "str", fill_value=fillvalue)
                        output_netcdf_obj.createVariable(vv + "_coef_a", "f", fill_value=fillvalue)
                        output_netcdf_obj.createVariable(vv + "_coef_b", "f", fill_value=fillvalue)
                        output_netcdf_obj.createVariable(vv + "_RAW_coef_a", "f", fill_value=fillvalue)
                        output_netcdf_obj.createVariable(vv + "_RAW_coef_b", "f", fill_value=fillvalue)
                if vv == 'Institute_code':
                    output_netcdf_obj.createVariable(vv, "str", fill_value=fillvalue)
                    # create a variable for the institute name
                    output_netcdf_obj.createVariable('Institute_name', "str", fill_value=fillvalue)
                if vv == 'XBT_line':
                    output_netcdf_obj.createVariable(vv, "str", fill_value=fillvalue)
                    # create a variable for the line description
                    output_netcdf_obj.createVariable('XBT_line_description', "str", fill_value=fillvalue)
                # create dimensioned variables:
                if vv in ['XBT_accept_code', 'XBT_reject_code']:
                    output_netcdf_obj.createVariable(vv, datatype=dttyp, dimensions=('DEPTH',), fill_value=fillvalue)
                if vv in ['DEPTH', 'TEMP', 'PSAL']:
                    output_netcdf_obj.createVariable(vv, datatype=dttyp, dimensions=('DEPTH',), fill_value=fillvalue)
                    # and associated QC variables:
                    output_netcdf_obj.createVariable(vv + "_quality_control", "b", dimensions=('DEPTH',), fill_value=99)
                    if vv in ['TEMP', 'DEPTH', 'PSAL']:
                        # add the uncertainty variable
                        output_netcdf_obj.createVariable(vv + "_uncertainty", datatype=dttyp, dimensions=('DEPTH',),
                                                         fill_value=fillvalue)
                    # and the *_RAW variables:
                    output_netcdf_obj.createVariable(vv + "_RAW", datatype=dttyp,
                                                     dimensions=('DEPTH',), fill_value=fillvalue)
                if vv in ['COND', 'RESISTANCE', 'SAMPLE_TIME', 'TEMP_RECORDING_SYSTEM']:
                    output_netcdf_obj.createVariable(vv, datatype=dttyp, dimensions=('DEPTH',), fill_value=fillvalue)
                    if vv in ['TEMP_RECORDING_SYSTEM']:
                        output_netcdf_obj.createVariable(vv + "_quality_control", "b", dimensions=('DEPTH',), fill_value=99)
                # test if the output_netCDF_obj already has the variable created
                if vv not in output_netcdf_obj.variables:
                    output_netcdf_obj.createVariable(vv, datatype=dttyp, fill_value=fillvalue)
            else:
                # if not TEMP_RECORDING_SYSTEM_quality_control, print a warning
                if vv != 'TEMP_RECORDING_SYSTEM_quality_control':
                    print("Variable skipped: \"%s\". Please check!!" % vv)

        # Add the XBT_accept_code and XBT_reject_code variables and size to same size as TEMP
        output_netcdf_obj.createVariable('XBT_accept_code', "int64", fill_value=99, dimensions=('DEPTH',))
        output_netcdf_obj.createVariable('XBT_reject_code', "int64", fill_value=99, dimensions=('DEPTH',))

        # set the sample time units
        year_value = nco.time.dt.year.astype(int).values[0]
        dti = datetime.datetime(year_value, 1, 1, 0, 0, 0)
        setattr(output_netcdf_obj.variables['SAMPLE_TIME'], 'units', 'milliseconds since ' +
                dti.strftime("%Y-%m-%d %H:%M:%S UTC"))

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
        # add the flag and feature type attributes:
        dfa, dfr = create_flag_feature()
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'valid_max', int(dfa['byte_value'].sum()))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_masks', dfa['byte_value'].astype(np.uint64))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_meanings', ' '.join(dfa['label']))
        setattr(output_netcdf_obj.variables['XBT_accept_code'], 'flag_codes', ' '.join(dfa['code']))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'valid_max', int(dfr['byte_value'].sum()))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_masks', dfr['byte_value'].astype(np.uint64))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_meanings', ' '.join(dfr['label']))
        setattr(output_netcdf_obj.variables['XBT_reject_code'], 'flag_codes', ' '.join(dfr['code']))

        # append the data to the file
        for v in varslist.keys():
            # get the matching output variable name
            vname = varslist[v]
            if (v not in list(nco.variables.keys())) and (not hasattr(nco, v)):
                print("Variable not found in original file: \"%s\"." % v)
                continue
            if vname not in output_netcdf_obj.variables:
                print("Variable not written to output file: \"%s\"." % v)
                continue
            # read the data either from the variables or the globals
            if v in list(nco.variables.keys()):
                # data is in the variables section of the original file
                data = np.squeeze(nco.variables[v].values)
            else:
                # information is kept in the globals of the original file
                data = getattr(nco, v)
            # print(vname)
            if vname in ['TIME', 'TIME_RAW','XBT_manufacturer_date', 'SAMPLE_TIME']:
                if vname == 'SAMPLE_TIME':
                    # Convert numpy.datetime64 array to a list of datetime objects
                    datetime_list = [pd.to_datetime(d).to_pydatetime() for d in data]
                    # Convert the list of datetime objects to numeric values
                    time_val_dateobj = date2num(datetime_list, output_netcdf_obj[vname].units,
                                                output_netcdf_obj[vname].calendar)
                else:
                    if vname == 'XBT_manufacturer_date' and test:
                        # data is not applicable as it is a test canister, so set to fill value
                        time_val_dateobj = np.ma.array([output_netcdf_obj[vname]._FillValue],
                                                       mask=True, fill_value=output_netcdf_obj[vname]._FillValue)
                    else:
                        time_val_dateobj = date2num(pd.to_datetime(data), output_netcdf_obj[vname].units,
                                                    output_netcdf_obj[vname].calendar)
                        if vname == 'TIME':
                            # set the time_coverage_start and time_coverage_end
                            output_netcdf_obj.time_coverage_start = pd.to_datetime(data).strftime("%Y-%m-%dT%H:%M:%SZ")
                            output_netcdf_obj.time_coverage_end = pd.to_datetime(data).strftime("%Y-%m-%dT%H:%M:%SZ")
                output_netcdf_obj.variables[vname][:] = time_val_dateobj
            elif v == 'InterfaceCode':
                # get the recorder type information
                rct = get_recorder_type(nco)
                output_netcdf_obj.variables['XBT_recorder_type'][len(rct[0])] = str(rct[0])
                output_netcdf_obj.variables['XBT_recorder_type_name'][len(rct[1])] = str(rct[1])
                continue
            elif vname == 'XBT_recorder_software_version':
                # remove 'Version:' and any trailing spaces from the string
                output_netcdf_obj.variables[vname][len(data)] = str(data).split('Version:')[1].strip()
                continue
            elif vname == 'PROBE_TYPE':
                # do for both the PROBE_TYPE and the PROBE_TYPE_RAW
                for probe in ['PROBE_TYPE', 'PROBE_TYPE_RAW']:
                    output_netcdf_obj.variables[probe][len(data)] = str(data)
                    # get the probe type name
                    probe_type_name = read_section_from_xbt_config('PEQ$')[data].split(',')[0]
                    output_netcdf_obj.variables[probe + '_name'][len(probe_type_name)] = str(probe_type_name)
                    # get the probe type coefficients
                    probe_type_coef = read_section_from_xbt_config('FRE')[data].split(',')
                    output_netcdf_obj.variables[probe + '_coef_a'][:] = float(probe_type_coef[0])
                    output_netcdf_obj.variables[probe + '_coef_b'][:] = float(probe_type_coef[1]) * 1e-3
                # add quality control for the probe type
                output_netcdf_obj.variables['PROBE_TYPE_quality_control'][:] = 0
                continue
            else:
                # Check the shape of the NetCDF variable
                var_shape = output_netcdf_obj[vname].shape

                # Ensure the data from profile[v] matches the shape of the NetCDF variable
                if not isinstance(data, str) and data.shape == var_shape:
                    output_netcdf_obj.variables[vname][:] = data
                else:
                    if isinstance(output_netcdf_obj[vname][:], str):
                        output_netcdf_obj.variables[vname][len(data)] = str(data)
                    else:
                        output_netcdf_obj.variables[vname][:] = data
            # if this vname also has a variable with _RAW, add the data to that variable
            if vname + '_RAW' in output_netcdf_obj.variables:
                if isinstance(data, str):
                    output_netcdf_obj.variables[vname + '_RAW'] = data
                else:
                    output_netcdf_obj.variables[vname + '_RAW'][:] = data
            # if this vname has a *_quality_control variable, add 0 to indicate no QC
            if vname + '_quality_control' in output_netcdf_obj.variables:
                output_netcdf_obj.variables[vname + '_quality_control'][:] = 0

        # add the uncertainties
        temp_uncertainty, depth_uncertainty = add_uncertainties(nco)
        output_netcdf_obj.variables['TEMP_uncertainty'][:] = temp_uncertainty
        output_netcdf_obj.variables['DEPTH_uncertainty'][:] = depth_uncertainty

        # add the extra variables
        output_netcdf_obj.variables['XBT_input_filename'][0] = raw_netCDF_file
        output_netcdf_obj.variables['XBT_cruise_ID'][0] = crid
        # Profile Id
        output_netcdf_obj.variables['XBT_uniqueid'][0] = unique_id

        # read from the controlled list of global attributes in the config file
        globals_list = read_section_from_xbt_config('Turo_globals')

        # read a list of code defined in the Turo_codes conf file. Create a
        # dictionary of matching values
        for att_name, att_name_out in globals_list.items():
            try:
                # Get the attribute value from the output_netcdf_obj
                att_val = getattr(nco, att_name, None)
                setattr(output_netcdf_obj, att_name_out, att_val.strip())
            except:
                LOGGER.warning('Attribute %s not found in the input file' % att_name)

        # add institute information, should be in here from the previous section
        institute_code = nco.Agency
        # get the list from the config file
        institute_list = read_section_from_xbt_config('INSTITUTE')
        # match the institute code to the second value in the list and derive the agency code
        for institute in institute_list:
            if institute_list[institute].split(',')[1] == institute_code:
                output_netcdf_obj.variables['Institute_name'][0] = institute_list[institute].split(',')[0]
            else:
                continue
        if not isinstance(output_netcdf_obj.variables['Institute_name'][0], str):
            LOGGER.warning('Institute code %s is not defined in xbt_config file. Please edit xbt_config' % institute)
            output_netcdf_obj.variables['Institute_name'][0] = 'Unknown'

        # ship name, IMO and callsign
        output_netcdf_obj.variables['ship_name'] = ship_name
        output_netcdf_obj.variables['ship_IMO'] = ship_IMO
        output_netcdf_obj.variables['Platform_code'] = callsign

        # add some final global attributes
        output_netcdf_obj.qc_completed = 'no'
        output_netcdf_obj.geospatial_lat_min = nco.latitude
        output_netcdf_obj.geospatial_lat_max = nco.latitude
        output_netcdf_obj.geospatial_lon_min = nco.longitude
        output_netcdf_obj.geospatial_lon_max = nco.longitude
        output_netcdf_obj.geospatial_vertical_min = nco.depth[0]
        output_netcdf_obj.geospatial_vertical_max = nco.depth[-1]

        # Convert time to a string
        utctime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        output_netcdf_obj.date_created = utctime

        # add the line information
        output_netcdf_obj.variables['XBT_line_description'][0] = line_info[1]

        # if this is a test canister, add the TP code and associated information to the HISTORIES and update the QC and XBT_reject_code
        if test:
            # create a dataframe with the codes and their integer representation
            df = read_qc_config()
            # get the test probe code
            tp_code = df[df['code'] == 'TPR']['byte_value'].values[0]
            # add the test probe code to the XBT_reject_code
            output_netcdf_obj.variables['XBT_reject_code'][0] = tp_code
            # change the TEMP_quality_control to 4
            output_netcdf_obj.variables['TEMP_quality_control'][:] = df[df['code'] == 'TPR']['tempqc'].values[0]
            # update the HISTORIES
            output_netcdf_obj.variables['HISTORY_INSTITUTION'][0] = 'CSIRO'
            output_netcdf_obj.variables['HISTORY_SOFTWARE'][0] = 'TuroXBT2IMOSnc.py'
            output_netcdf_obj.variables['HISTORY_SOFTWARE_RELEASE'][0] = 'V1.0'
            output_netcdf_obj.variables['HISTORY_DATE'][0] = date2num(datetime.datetime.now(), output_netcdf_obj['HISTORY_DATE'].units,
                                                                        output_netcdf_obj['HISTORY_DATE'].calendar)
            output_netcdf_obj.variables['HISTORY_PARAMETER'][0] = df[df['code'] == 'TPR']['parameter'].values[0]
            output_netcdf_obj.variables['HISTORY_START_DEPTH'][0] = nco.depth[0]
            output_netcdf_obj.variables['HISTORY_STOP_DEPTH'][0] = nco.depth[-1]
            output_netcdf_obj.variables['HISTORY_QC_CODE'][0] = 'TPR'
            output_netcdf_obj.variables['HISTORY_QC_CODE_VALUE'][0] = df[df['code'] == 'TPR']['tempqc'].values[0]
            output_netcdf_obj.variables['HISTORY_QC_CODE_DESCRIPTION'][0] = df[df['code'] == 'TPR']['label'].values[0]



if __name__ == '__main__':
    # parse the input arguments
    vargs = args()
    global_vars(vargs)

    # set up the input and output directories
    files_pattern = os.path.join(vargs.input_xbt_path, "*.nc")
    files = sorted(glob.glob(files_pattern))
    # Filter out files that match the '*.n.nc' format using regular expression
    pattern = re.compile(r'.*\.\d+\.nc$')
    files = [file for file in files if not pattern.search(file)]
    first = True # to handle the test* files which also start at 1

    for file in files:  # read/write loop
        nco = xr.open_dataset(file)
        raw_netCDF_file = os.path.join(os.path.basename(vargs.input_xbt_path),os.path.basename(file))
        print(raw_netCDF_file)

        # read the drop number from filename of raw file
        # e.g. drop001.nc
        name, _ = os.path.splitext(os.path.basename(file))
        # make sure the name isn't a *.*.nc file
        name = name.split(".")
        n = int(name[0][4:])
        # check the cruise id and ship name
        crid = nco.Voyage
        callsign = nco.CallSign
        xbtline = nco.LineNo
        # for the first file only, ask the user to confirm the cruise id and ship name
        if n == 1 and first:
            first = False
            # ask the user to confirm the cruise id and ship name
            user_input = input("Is %s the correct cruise id [Y/N]: " % crid)
            if user_input == 'N':
                cid = input("Enter the correct cruise id: ")
            else:
                cid = crid
            user_input = input("Is %s the correct call sign [Y/N]: " % callsign).upper()
            if user_input == 'N':
                calls = input("Enter the correct call sign: ")
            else:
                calls = callsign
            user_input = input("Is %s the correct line number [Y/N]: " % xbtline).upper()
            if user_input == 'N':
                line = input("Enter the correct line number: ")
            else:
                line = xbtline
            # handle ship name, IMO and callsign
            if calls in SHIPS:
                ship_name = SHIPS[calls][0]
                ship_IMO = SHIPS[calls][1]
            elif difflib.get_close_matches(calls, SHIPS, n=1, cutoff=0.8) != []:
                callsmatch = difflib.get_close_matches(calls, SHIPS, n=1, cutoff=0.8)[0]
                ship_name = SHIPS[callsmatch][0]
                ship_IMO = SHIPS[callsmatch][1]
                LOGGER.warning(
                    'Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s' % (
                        SHIPS[callsmatch], calls))
                calls = callsmatch
            else:
                LOGGER.warning('Vessel call sign %s, name %s, is unknown in AODN vocabulary. Please contact '
                               'info@aodn.org.au' % callsign, nco.Ship)
                ship_name = 'Unknown'
                ship_IMO = 'Unknown'
            # get the line information from AODN vocabularies
            xbt_line_codes = [s for s in list(xbt_line_info().keys())]  # IMOS codes taken from vocabulary
            if line in xbt_line_codes:
                line_info = xbt_line_info()[xbtline]
            else:
                # warning if the line is not in the vocab
                LOGGER.warning('XBT line %s not found in the AODN vocabulary, assigning NOLINE line' % xbtline)
                # create a tuple with 'Unknown' values
                line_info = ('NOLINE', 'NO LINE')


        # if crid is not the same as cid, use cid
        if 'drop' in name[0]:
            if crid != cid:
                crid = cid
            if callsign != calls:
                callsign = calls

        # Write function
        netCDFout(nco, n, crid, callsign, ship_IMO, ship_name, line_info, raw_netCDF_file)

