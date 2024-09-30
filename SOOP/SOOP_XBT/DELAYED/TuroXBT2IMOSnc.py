# Converts XBT profile recorded by Turo XBT to standardised netCDF format ready for QC with PYQUEST
# A. Walsh V2 4/10/22
import argparse
import difflib
# V3 - adjustments to format for compatibility with MQUEST netCDF and changes to PyQUEST-XBT.py.
# For compatibility with MQUEST to IMOS netCDF converter change all LAT,LONG,DEPTH,DEPTH_RAW,TEMP,TEMP_RAW from type 'd' (double) to type 'f' (float)

####Usage####
# Before running this script:
# 1) create 2 folders: xbtdata_raw,xbtdata_standardised
# 2) Copy the raw data files (monthly folder contents from Turo XBT) dropXXX.nc to xbtdata_raw
# Ensure there are no duplicated raw files; include edited version of drop if any, not original unedited version
# General command:
# python TuroXBT2StdNetCDF-V3.py -i xbtdata_raw -o xbtdata_standardised -c YYNNNSS -s "HMAS XXXX"

# Where YYNNNSS = Cruise ID
# For Turo XBT a 'cruise' is normally a 1 month of data from 1 ship, as Turo auto organises data into monthly folders
# YY =last 2 digits of the observation year
# NNN= consecutive number of dataset recived in that year
# XX = 2 character abbreviation of the shipname e.g. SY=HMAS Sydney
# xbtdata_raw = input folder holding raw files from Turo XBT - dropXXX.nc
# xbtdata_standardised = output folder to hold standardised files produced by this script

# Example:
# python TuroXBT2StdNetCDF-V3.py -i xbtdata_raw_3 -o xbtdata_standardised_3 -c 21018AN -s "HMAS ANZAC"


####Output File names####
# General format - YYNNNSS-YYYYMMDDThhmmssZ-CCC.nc
# YYNNNSS = Cruise ID
# CCC = Consecutive No 000, 001, ...
# YYYYMMDDThhmmssZ = XBT drop UTC Date-time
# e.g. 20001SY-20210303T212721Z-000.nc

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
from optparse import OptionParser
import glob
import pandas as pd

from xbt_line_vocab import xbt_line_info
from xbt_parse import read_section_from_xbt_config
from generate_netcdf_att import generate_netcdf_att, get_imos_parameter_info
from ship_callsign import ship_callsign_list
from imos_logging import IMOSLogging
from xbt_utils import _error


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
    dfa = pd.read_csv(a_file_path)
    dfr = pd.read_csv(r_file_path)

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
    temp_uncertainty = np.ma.empty_like(nco.temperature)
    temp_uncertainty[:] = tunc
    # depth uncertainties:
    unc = np.ma.MaskedArray(nco.depth * dunc[0], mask=False)
    if len(dunc) > 1:
        unc[nco.depth <= 230] = dunc[1]
    depth_uncertainty = np.round(unc, 2)

    return temp_uncertainty, depth_uncertainty


def get_recorder_type(profile):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')

    att_name = 'XBT_recorder_type'

    item_val = str(int(nco.InterfaceCode))
    if item_val in list(rct_list.keys()):
        return item_val, rct_list[item_val].split(',')[0]
    else:
        LOGGER.warning(
            '{item_val} missing from recorder type part in xbt_config file, using unknown for recorder'.format(
                item_val=item_val))
        item_val = '99'
        return item_val, rct_list[item_val].split(',')[0]


def netCDFout(nco, n, crid, callsign, xbtline):
    # get the config file information:
    # get xbt line information from config file
    line_info = read_section_from_xbt_config(xbtline)

    # create the output file name
    test = False
    if nco.TestCanister == 'yes':
        test = True
    outfile, unique_id = create_out_filename(nco, line_info['XBT_line'], crid, n, test)
    outfile = os.path.join(vargs.output_folder, outfile)

    # create a ncobject to write out to new format
    with Dataset(outfile, "w", format="NETCDF4") as output_netcdf_obj:
        # Create the dimensions
        output_netcdf_obj.createDimension('DEPTH', len(nco.depth))
        output_netcdf_obj.createDimension('N_HISTORY', 0)  # make this unlimited

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
        varslist = ["DEPTH", "TEMP"]
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

        # add the recording system TEMP:
        output_netcdf_obj.createVariable("TEMP_RECORDING_SYSTEM", "f", dimensions=('DEPTH',),
                                         fill_value=999999.0)
        # and associated QC variable:
        output_netcdf_obj.createVariable("TEMP_RECORDING_SYSTEM_quality_control", "b", dimensions=('DEPTH',),
                                         fill_value=-51)

        # Create the last variables that are non-standard:
        output_netcdf_obj.createVariable("PROBE_TYPE", 'S3')
        output_netcdf_obj.createVariable("PROBE_TYPE_quality_control", "b", fill_value=99)
        output_netcdf_obj.createVariable("PROBE_TYPE_RAW", 'S3')

        accept_codes = output_netcdf_obj.createVariable("XBT_accept_code", "u8", dimensions=('DEPTH',),
                                                  fill_value=0)
        reject_codes = output_netcdf_obj.createVariable("XBT_reject_code", "u8", dimensions=('DEPTH',),
                                                    fill_value=0)

        # We have turo files, so lets make the resistance and sample_time variables
        output_netcdf_obj.createVariable("RESISTANCE", "f", dimensions=('DEPTH',), fill_value=float("nan"))
        output_netcdf_obj.createVariable("SAMPLE_TIME", "f", dimensions=('DEPTH',), fill_value=float("nan"))
        # set the sample time units
        year_value = nco.time.dt.year.astype(int).values[0]
        dt = datetime.datetime(year_value, 1, 1, 0, 0, 0)
        setattr(output_netcdf_obj.variables['SAMPLE_TIME'], 'units', 'milliseconds since ' +
                dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

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
        # add the flag and feature type attributes:
        dfa, dfr = create_flag_feature()
        setattr(accept_codes, 'valid_max', int(dfa['byte_value'].sum()))
        setattr(accept_codes, 'flag_masks', dfa['byte_value'].astype(np.uint64))
        setattr(accept_codes, 'flag_meanings', ' '.join(dfa['label']))
        setattr(accept_codes, 'flag_codes', ' '.join(dfa['code']))
        setattr(reject_codes, 'valid_max', int(dfr['byte_value'].sum()))
        setattr(reject_codes, 'flag_masks', dfr['byte_value'].astype(np.uint64))
        setattr(reject_codes, 'flag_meanings', ' '.join(dfr['label']))
        setattr(reject_codes, 'flag_codes', ' '.join(dfr['code']))
        # write coefficients out to the attributes. In the PROBE_TYPE, PROBE_TYPE_RAW, DEPTH, DEPTH_RAW
        # find the matching probe type in the config file
        probe_type = nco.Code
        if probe_type in fre_list:
            fre = fre_list[probe_type]
            prt = peq_list[probe_type]
        varnames = ['PROBE_TYPE', 'DEPTH', 'PROBE_TYPE_RAW', 'DEPTH_RAW']
        for v in varnames:
            setattr(output_netcdf_obj.variables[v], 'fallrate_coefficients', fre)
            setattr(output_netcdf_obj.variables[v], 'probe_type_name', prt.split(',')[0])

        # append the data to the file
        time_to_output = date2num(nco.time.values.astype('M8[s]').tolist(), output_netcdf_obj['TIME'].units)
        output_netcdf_obj.variables['TIME'][:] = time_to_output
        output_netcdf_obj.variables['TIME_RAW'][:] = time_to_output
        output_netcdf_obj.variables['TIME_quality_control'][:] = 0
        output_netcdf_obj.variables['LATITUDE'][:] = nco.latitude
        output_netcdf_obj.variables['LATITUDE_quality_control'][:] = 0
        output_netcdf_obj.variables['LONGITUDE'][:] = nco.longitude
        output_netcdf_obj.variables['LONGITUDE_quality_control'][:] = 0
        output_netcdf_obj.variables['LATITUDE_RAW'][:] = nco.latitude
        output_netcdf_obj.variables['LONGITUDE_RAW'][:] = nco.longitude
        output_netcdf_obj.variables['DEPTH'][:] = nco.depth
        output_netcdf_obj.variables['DEPTH_quality_control'][:] = np.zeros(len(nco.depth))
        output_netcdf_obj.variables['DEPTH_RAW'][:] = nco.depth
        output_netcdf_obj.variables['TEMP'][:] = nco.temperature
        output_netcdf_obj.variables['TEMP_quality_control'][:] = np.zeros(len(nco.depth))
        output_netcdf_obj.variables['TEMP_RAW'][:] = nco.temperature
        # add probe type
        output_netcdf_obj.variables['PROBE_TYPE'][len(probe_type)] = probe_type
        output_netcdf_obj.variables['PROBE_TYPE_RAW'][len(probe_type)] = probe_type
        output_netcdf_obj.variables['PROBE_TYPE_quality_control'][:] = 0
        # add the resistance and sample time
        output_netcdf_obj.variables['RESISTANCE'][:] = nco.resistance
        output_netcdf_obj.variables['SAMPLE_TIME'][:] = nco.sampleTime
        # add the recorder_temp and auto QC from Turo
        output_netcdf_obj.variables['TEMP_RECORDING_SYSTEM'][:] = nco.procTemperature
        output_netcdf_obj.variables['TEMP_RECORDING_SYSTEM_quality_control'][:] = nco.sampleQC

        # add the uncertainties
        temp_uncertainty, depth_uncertainty = add_uncertainties(nco)
        output_netcdf_obj.variables['TEMP_uncertainty'][:] = temp_uncertainty
        output_netcdf_obj.variables['DEPTH_uncertainty'][:] = depth_uncertainty

        # add the global attributes
        output_netcdf_obj.XBT_input_filename = raw_netCDF_file
        output_netcdf_obj.XBT_cruise_ID = cid
        # get the time as a string
        dt = nco.time.values[0]
        # Convert numpy.datetime64 to datetime.datetime
        dt = pd.to_datetime(str(dt)).to_pydatetime()
        # Profile Id
        pid = "%s_%s_%03d" % (cid, dt.strftime("%Y%m%d%H%M%S"), n)
        output_netcdf_obj.XBT_uniqueid = pid
        output_netcdf_obj.TestCanister = nco.TestCanister

        if callsign in SHIPS:
            output_netcdf_obj.ship_name = SHIPS[callsign][0]
            output_netcdf_obj.Callsign = callsign
            output_netcdf_obj.SOTID = nco.SOTID
            output_netcdf_obj.ship_IMO = SHIPS[callsign][1]
            output_netcdf_obj.Platform_code = callsign
        elif difflib.get_close_matches(callsign, SHIPS, n=1, cutoff=0.8) != []:
            output_netcdf_obj.Callsign = \
                difflib.get_close_matches(callsign, SHIPS, n=1, cutoff=0.8)[0]
            output_netcdf_obj.Platform_code = output_netcdf_obj.Callsign
            output_netcdf_obj.ship_name = SHIPS[output_netcdf_obj.Callsign]
            output_netcdf_obj.SOTID = nco.SOTID
            output_netcdf_obj.ship_IMO = SHIPS[output_netcdf_obj.Callsign][1]
            LOGGER.warning(
                'Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s' % (
                    SHIPS[output_netcdf_obj.Callsign], output_netcdf_obj.Callsign))
        else:
            output_netcdf_obj.ship_name = nco.Ship
            output_netcdf_obj.Callsign = callsign
            output_netcdf_obj.Platform_code = callsign
            output_netcdf_obj.SOTID = nco.SOTID
            output_netcdf_obj.ship_IMO = nco.IMO
            LOGGER.warning('Vessel call sign %s, name %s, is unknown in AODN vocabulary. Please contact '
                           'info@aodn.org.au' % callsign, nco.Ship)

        output_netcdf_obj.Recorder_hardware_serial_no = nco.HardwareSerialNo
        output_netcdf_obj.Recorder_HardwareCalibration = nco.HardwareCalibration
        output_netcdf_obj.Recorder_Graphical_User_Interface_version = nco.UIVersion
        output_netcdf_obj.Recorder_software_version = nco.ReleaseVersion
        output_netcdf_obj.Recorder_firmware_version = nco.FirmwareVersion
        output_netcdf_obj.TemperatureCoefficients = nco.TemperatureCoefficients

        # crc might not exist, skip if not
        if 'CRC' in nco:
            output_netcdf_obj.cyclic_redundancy_code = nco.CRC
        # get the recorder type information
        rct = get_recorder_type(nco)
        output_netcdf_obj.XBT_recorder_type = "WMO Code table 4770 code %s, %s" % rct
        output_netcdf_obj.XBT_probe_serial_number = nco.SerialNo
        output_netcdf_obj.XBT_calibration_SCALE = nco.Scale
        output_netcdf_obj.XBT_calibration_OFFSET = nco.Offset
        # reformat batch date from mm/dd/yyyy to yyyymmdd
        if not test:
            date_object = datetime.datetime.strptime(nco.BatchDate, "%m/%d/%y")
            output_netcdf_obj.XBT_manufacturer_date_yyyymmdd = date_object.strftime("%Y%m%d")
            output_netcdf_obj.XBT_box_number = nco.CaseNo
            output_netcdf_obj.XBT_height_launch_above_water_in_meters = float(nco.DropHeight)

        output_netcdf_obj.XBT_predrop_comments = nco.PreDropComments
        output_netcdf_obj.XBT_postdrop_comments = nco.PostDropComments
        output_netcdf_obj.qc_completed = 'no'

        output_netcdf_obj.geospatial_lat_min = nco.latitude
        output_netcdf_obj.geospatial_lat_max = nco.latitude
        output_netcdf_obj.geospatial_lon_min = nco.longitude
        output_netcdf_obj.geospatial_lon_max = nco.longitude

        output_netcdf_obj.geospatial_vertical_min = nco.depth[0]
        output_netcdf_obj.geospatial_vertical_max = nco.depth[-1]

        # Convert time to a string
        formatted_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        output_netcdf_obj.time_coverage_start = formatted_date
        output_netcdf_obj.time_coverage_end = formatted_date

        utctime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        output_netcdf_obj.date_created = utctime

        # add the line information
        for key, item in line_info.items():
             setattr(output_netcdf_obj, key, item)


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
        # for the first file only and 'drop' in the name, ask the user to confirm the cruise id and ship name
        if n == 1 and 'drop' in name[0]:
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

        # if crid is not the same as cid, use cid
        if crid != cid:
            crid = cid
        if callsign != calls:
            callsign = calls
        if line != xbtline:
            xbtline = line

        # Write function
        netCDFout(nco, n, crid, callsign, xbtline)
