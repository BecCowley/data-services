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

import sys
import tempfile
import xarray as xr
import glob

from xbt_line_vocab import xbt_line_info
from ship_callsign import ship_callsign_list
from imos_logging import IMOSLogging
from xbt_utils import *
from write2netcdf import write_output_nc


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
    # if the profile has 'HardwareSerialNo' is in the profile global attributes, include that in the filename
    if 'HardwareSerialNo' in profile.attrs:
        serial_number = profile.attrs['HardwareSerialNo']
        if serial_number is not None:
            crid = str(serial_number).strip() + '_' + str(crid).strip()
    # create the unique ID from the crid, time and drop number formatted to three digits
    uniqueid = crid + '_' + profile.time.dt.strftime('%Y%m%d%H%M%S').values[0] + '_' + str(n).zfill(3)

    return uniqueid


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
    ''' create three dataframes from the nco object and write them to a netCDF file using write_output_nc function.
    nco: xarray dataset object
    n: drop number
    crid: cruise id
    callsign: ship call sign
    ship_IMO: ship IMO number
    ship_name: ship name
    line_info: list of line information
    raw_netCDF_file: full path to the Turo netCDF file
    '''

    # create a unique identifier
    test = False
    if nco.TestCanister == 'yes':
        test = True
    unique_id = create_out_filename(nco, line_info[0], crid, n, test)

    # build the profile dataframe
    # First, get a list of variables mapped between nco and output_netcdf_obj
    varslist = read_variables_config()
    # remove the HISTORY* variables from the varslist
    varslist = varslist[~varslist['variable_name'].str.startswith('HISTORY_')]

    # create an empty dataframe from the variable_name column of varslist and the same legth as np.squeeze(nco.depth.data)
    dfprofile = pd.DataFrame(index=np.arange(len(np.squeeze(nco.depth.data))), columns=varslist['variable_name'].tolist())

    #loop through each row of the varslist dataframe
    for index, row in varslist.iterrows():
        vname = row['variable_name']
        # print(vname)
        # get the turo variable name
        turo_name = row['Turo']
        # if the Turo column is empty, skip it
        if turo_name is pd.NA:
            continue

        # read the data either from the variables or the globals
        if turo_name in list(nco.variables.keys()):
            # data is in the variables section of the original file
            data = np.squeeze(nco.variables[turo_name].values)
        else:
            if turo_name in list(nco.attrs.keys()):
                # information is kept in the globals of the original file
                data = getattr(nco, turo_name)
            else:
                # data not in variables or globals, skip this variable as it will have a fill value
                print("Variable not found in original file: \"%s\"." % vname)
                continue

        if vname in ['TIME','PROBE_manufacture_date_YYYY-MM-DD', 'SAMPLE_TIME']:
            if vname == 'SAMPLE_TIME':
                # Convert numpy.datetime64 array to a list of datetime objects
                datetime_list = [pd.to_datetime(d).to_pydatetime() for d in data]
                # save the datetime list to the profile dataframe
                dfprofile['SAMPLE_TIME'] = datetime_list
            else:
                if vname == 'PROBE_manufacture_date_YYYY-MM-DD':
                    # convert the string to a datetime object, assuming correct format entry of MM/DD/YY
                    data = convert_time_string(data, format='%m/%d/%y', output='datetime')

                    if data is None or test:
                        # data is not applicable as it is a test canister, fill the profile['PROBE_manufacture_date_YYYY-MM-DD'] with None
                        dfprofile['PROBE_manufacture_date_YYYY-MM-DD'] = None
                    else:
                        if type(data) == str or data is None:
                            # put None in the profile dataframe
                            dfprofile['PROBE_manufacture_date_YYYY-MM-DD'] = None
                        else:
                            # put the datetime object in the profile dataframe
                            dfprofile['PROBE_manufacture_date_YYYY-MM-DD'] = pd.to_datetime(data)
                else:
                    dfprofile[vname] = pd.to_datetime(data)

            # if vname is TIME, output the TIME_RAW variable as it is the same as TIME
            if vname == 'TIME':
                dfprofile['TIME_RAW'] = data
        elif vname == 'RECORDER_TYPE':
            # get the recorder type information
            rct = get_recorder_type(nco)
            dfprofile['RECORDER_TYPE'] = str(rct[0])
            dfprofile['RECORDER_TYPE_name'] = str(rct[1])
            continue
        elif vname == 'RECORDER_software_version':
            # remove 'Version:' and any trailing spaces from the string
            dfprofile[vname] = str(data).split('Version:')[1].strip()
            continue
        elif vname == 'PROBE_TYPE':
            # do for both the PROBE_TYPE and the PROBE_TYPE_RAW
            for probe in ['', '_RAW']:
                dfprofile['PROBE_TYPE' + probe] = data
                # get the probe type name, return 'Unknown' if not found
                if str(data) in list(peq_list.keys()):
                    probe_type_name = peq_list[str(data)].split(',')[0]
                else:
                    LOGGER.warning(
                        'Probe type %s missing from probe type part in xbt_config file, using unknown for probe type' % str(
                            data))
                    probe_type_name = ''
                dfprofile['PROBE_TYPE_name' + probe] = str(probe_type_name)
                # get the probe type coefficients
                if str(data) not in list(fre_list.keys()):
                    LOGGER.warning(
                        'Probe type %s missing from frequency part in xbt_config file, using default coefficients' % str(
                            data))
                    probe_type_coef = ''.split(',')
                else:
                    probe_type_coef = fre_list[str(data)].split(',')
                dfprofile['PROBE_TYPE_coefficient_a' + probe] = float(probe_type_coef[0])
                dfprofile['PROBE_TYPE_coefficient_b' + probe] = float(probe_type_coef[1]) * 1e-3
            # add quality control for the probe type
            dfprofile['PROBE_TYPE_quality_control'] = 0
            continue
        else:
            dfprofile[vname] = data
        # if this vname also has a variable with _RAW, and isn't TIME, add the data to that variable
        if (vname != 'TIME') and (vname + '_RAW' in dfprofile.columns):
            # if the variable is a string, convert it to a string
            if isinstance(data, str):
                dfprofile[vname + '_RAW'] = str(data)
            else:
                # otherwise just copy the data
                dfprofile[vname + '_RAW'] = data
        # if this vname has a *_quality_control variable, add 0 to indicate no QC
        if vname + '_quality_control' in dfprofile.columns:
            # fill the quality control variable with 0
            dfprofile[vname + '_quality_control'] = 0

    # add the uncertainties
    dfprofile = add_uncertainties(dfprofile)

    # add the extra variables
    dfprofile['Input_filename'] = raw_netCDF_file
    # Profile Id
    dfprofile['Institution_unique_identifier'] = unique_id

    # get the list from the config file
    institute_list = read_section_from_xbt_config('INSTITUTE')
    # match the institute code to the second value in the list and derive the agency code
    for institute in institute_list:
        if institute_list[institute].split(',')[1] == dfprofile['Institution_code'][0]:
            dfprofile['Institution'] = institute_list[institute].split(',')[0]
        else:
            continue
    if dfprofile['Institution'][0] is None:
        LOGGER.warning('Institute code %s is not defined. Please review' % institute)
        dfprofile['Institution'] = 'Unknown'

    # add Launcher_type
    dfprofile = add_launcher_variable(dfprofile)

    # add some final global attributes
    dfprofile['qc_completed'] = 'no'

    # add the line information
    dfprofile['SOOP_line_description'] = line_info[1]

    # add 0 to the QC_accept_code and QC_reject_code columns
    dfprofile['QC_accept_code'] = 0
    dfprofile['QC_reject_code'] = 0

    # add automatic CSR QC flag to the profile if it is not a test canister
    code = {'CSR': 0}

    # also add a WBR test here
    # run the WBR test
    wbr_point, wbr_result = wire_break(dfprofile)
    # if the WBR test failed write the WBR code to the QC_reject_code and add the WBR history
    if wbr_result:
        # append the WBR code to the code dictionary and the dep of 0
        code['WBR'] = wbr_point

    # if this is a test canister, add the TP code and associated information to the HISTORIES and update the QC and QC_reject_code
    if test:
        # CSR not applicable for test canisters, so add TP code
        code = {'TPR': 0}

    # create a dataframe for the history information
    dfhist = pd.DataFrame(columns=['HISTORY_INSTITUTION',
                           'HISTORY_SOFTWARE', 'HISTORY_SOFTWARE_RELEASE', 'HISTORY_DATE', 'HISTORY_PARAMETER', 'HISTORY_START_DEPTH',
                            'HISTORY_QC_CODE', 'HISTORY_QC_CODE_VALUE', 'HISTORY_QC_CODE_DESCRIPTION'])
    # add the history information
    for c, dep in code.items():
        # add the history information to the dataframe
        dfhist, dfprofile = update_histories(dfprofile, c, 'TuroXBT2IMOSnc.py', 'v1.0', dfhist, dep)

   # return the profile dataframe, the global attributes and history information
    return dfprofile, dfhist


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
            user_input = input("Is %s the correct cruise id [Y/N]: " % crid).upper()
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
                               'info@aodn.org.au' % (callsign, nco.Ship))
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
        profile, history = netCDFout(nco, n, crid, callsign, ship_IMO, ship_name, line_info, raw_netCDF_file)
        # write the output to a netCDF file
        write_output_nc(vargs.output_folder, profile, history, profile_raw=False)
        # write the output to a netCDF file with the raw profile
        output_folder = os.path.join(vargs.output_folder, 'non_qc')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        write_output_nc(output_folder, profile, history, profile_raw=True,historic_flags=False)
