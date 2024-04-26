#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ANMN NRS data from AIMS Web Service for Darwin, Yongala and Beagle
The script reads an XML file provided by AIMS and looks for channels with
new data to download. It compares this list with a pickle file (pythonic
way to store python variables) containing what has already been downloaded
in the previous run of this script.
Some modifications on the files have to be done so they comply with CF and
IMOS conventions.
The IOOS compliance checker is used to check if the first downloaded file of
a channel complies once modified. If not, the download of the rest of the
channel is aborted until some modification on the source code is done so
the channel can pass the checker.
Files which don't pass the checker will land in os.path.join(wip_path, 'errors')
for investigation. No need to reprocess them as they will be redownloaded on
next run until they end up passing the checker. Files in the 'errors' dir can be
removed at anytime

IMPORTANT:
is it essential to look at the logging os.path.join(wip_path, 'aims.log')
to know which channels have problems and why as most of the time, AIMS will
have to be contacted to sort out issues.


author Laurent Besnard, laurent.besnard@utas.edu.au
"""

import argparse
import datetime
import logging
import os
import sys
import re
import shutil
import traceback
import unittest as data_validation_test

from netCDF4 import Dataset
from itertools import groupby
from tendo import singleton

from aims_realtime_util import (convert_time_cf_to_imos,
                                create_list_of_dates_to_download, download_channel,
                                fix_data_code_from_filename,
                                fix_provider_code_from_filename,
                                has_var_only_fill_value,
                                is_no_data_found, is_time_monotonic,
                                is_time_var_empty, logging_aims, md5,
                                modify_aims_netcdf, parse_aims_xml,
                                remove_dimension_from_netcdf,
                                remove_end_date_from_filename, save_channel_info,
                                set_up, rm_tmp_dir, get_main_netcdf_var,
                                list_recursively_files_abs_path)
from dest_path import get_anmn_nrs_site_name
from util import pass_netcdf_checker

DATA_WIP_PATH = os.path.join(os.environ.get('WIP_DIR'), 'ANMN', 'NRS_AIMS_Darwin_Yongala_data_rss_download_temporary')
ANMN_NRS_INCOMING_DIR = os.path.join(os.environ.get('INCOMING_DIR'), 'AODN', 'ANMN_NRS_DAR_YON')
ANMN_NRS_ERROR_DIR = os.path.join(os.environ['ERROR_DIR'], 'ANMN_NRS_DAR_YON')


def modify_anmn_nrs_netcdf(netcdf_file_path, channel_id_info):
    """ Modify the downloaded netCDF file so it passes both CF and IMOS checker
    input:
       netcdf_file_path(str)    : path of netcdf file to modify
       channel_id_index(tupple) : information from xml for the channel
    """
    modify_aims_netcdf(netcdf_file_path, channel_id_info)

    netcdf_file_obj                 = Dataset(netcdf_file_path, 'a', format='NETCDF4')
    netcdf_file_obj.aims_channel_id =  int(channel_id_info['channel_id'])

    if 'Yongala' in channel_id_info['site_name']:
        netcdf_file_obj.site_code     = 'NRSYON'
        netcdf_file_obj.platform_code = 'Yongala NRS Buoy'
    elif 'Darwin' in channel_id_info['site_name']:
        netcdf_file_obj.site_code     = 'NRSDAR'
        netcdf_file_obj.platform_code = 'Darwin NRS Buoy'
    elif 'Beagle' in channel_id_info['site_name']:
        netcdf_file_obj.site_code     = 'DARBGF'
        netcdf_file_obj.platform_code = 'Beagle Gulf Mooring'
    else:
        return False

    if not (channel_id_info['metadata_uuid'] == 'Not Available'):
        netcdf_file_obj.metadata_uuid = channel_id_info['metadata_uuid']

    # some weather stations channels don't have a depth variable if sensor above water
    if 'depth' in netcdf_file_obj.variables.keys():
        var                 = netcdf_file_obj.variables['depth']
        var.long_name       = 'nominal depth'
        var.positive        = 'down'
        var.axis            = 'Z'
        var.reference_datum = 'sea surface'
        var.valid_min       = -10.0
        var.valid_max       = 30.0
        var.units           = 'm'  # some channels put degrees celcius instead ...
        netcdf_file_obj.renameVariable('depth', 'NOMINAL_DEPTH')

    if 'DEPTH' in netcdf_file_obj.variables.keys():
        var                 = netcdf_file_obj.variables['DEPTH']
        var.coordinates     = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"
        var.long_name       = 'actual depth'
        var.reference_datum = 'sea surface'
        var.positive        = 'down'
        var.valid_min       = -10.0
        var.valid_max       = 30.0
        var.units           = 'm'  # some channels put degrees celcius instead ...

    netcdf_file_obj.close()
    netcdf_file_obj = Dataset(netcdf_file_path, 'a', format='NETCDF4')  # need to close to save to file. as we call get_main_var just after
    main_var        = get_main_netcdf_var(netcdf_file_path)
    # DEPTH, LATITUDE and LONGITUDE are not dimensions, so we make them into auxiliary cooordinate variables by adding this attribute
    if 'NOMINAL_DEPTH' in netcdf_file_obj.variables.keys():
        netcdf_file_obj.variables[main_var].coordinates = "TIME LATITUDE LONGITUDE NOMINAL_DEPTH"
    else:
        netcdf_file_obj.variables[main_var].coordinates = "TIME LATITUDE LONGITUDE"

    netcdf_file_obj.close()

    if not convert_time_cf_to_imos(netcdf_file_path):
        return False

    remove_dimension_from_netcdf(netcdf_file_path)  # last modification to do in this order!
    return True


def move_to_tmp_incoming(netcdf_path):
    # [org_filename withouth creation date].[md5].nc to have unique filename in
    new_filename = '%s.%s.nc' % (os.path.splitext(os.path.basename(remove_end_date_from_filename(netcdf_path)))[0], md5(netcdf_path))

    os.chmod(netcdf_path, 0o0664)  # change to 664 for pipeline v2
    shutil.move(netcdf_path, os.path.join(TMP_MANIFEST_DIR, new_filename))


def process_monthly_channel(channel_id, aims_xml_info, level_qc):
    """ Downloads all the data available for one channel_id and moves the file to a wip_path dir
    channel_id(str)
    aims_xml_info(tuple)
    level_qc(int)

    aims_service : 1   -> FAIMMS data
                   100 -> SOOP TRV data
                   300 -> NRS DATA
    for monthly data download, only 1 and 300 should be use
    """
    logger.info('QC{level_qc} - Processing channel {channel_id}'.format(channel_id=str(channel_id),
                                                                        level_qc=str(level_qc)))
    channel_id_info = aims_xml_info[channel_id]
    from_date = channel_id_info['from_date']
    thru_date = channel_id_info['thru_date']
    [start_dates, end_dates] = create_list_of_dates_to_download(channel_id, level_qc, from_date, thru_date)

    if len(start_dates) != 0:
        # download monthly file
        for start_date, end_date in zip(start_dates, end_dates):
            start_date           = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date             = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            netcdf_tmp_file_path = download_channel(channel_id, start_date, end_date, level_qc)
            contact_aims_msg     = "Process of channel aborted - CONTACT AIMS"

            if netcdf_tmp_file_path is None:
                logger.error('   Channel %s - not valid zip file - %s' % (str(channel_id), contact_aims_msg))
                break

            # NO_DATA_FOUND file only means there is no data for the selected time period. Could be some data afterwards
            if is_no_data_found(netcdf_tmp_file_path):
                logger.info('Channel {channel_id}: No data for the time period:[{start_date} - {end_date}]'.format(
                    channel_id=str(channel_id),
                    start_date=start_date,
                    end_date=end_date))
                shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
            else:
                if is_time_var_empty(netcdf_tmp_file_path):
                    logger.error('Channel {channel_id}: No values in TIME variable - {message}'.format(
                        channel_id=str(channel_id),
                        message=contact_aims_msg))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                if not modify_anmn_nrs_netcdf(netcdf_tmp_file_path, channel_id_info):
                    logger.error('Channel{channel_id}: Could not modify the NetCDF file - Process of channel aborted'.
                                 format(channel_id=str(channel_id)))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                main_var = get_main_netcdf_var(netcdf_tmp_file_path)
                if has_var_only_fill_value(netcdf_tmp_file_path, main_var):
                    logger.error('Channel {channel_id}: _Fillvalues only in main variable - {message}'.format(
                        channel_id=str(channel_id),
                        message=contact_aims_msg))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                if get_anmn_nrs_site_name(netcdf_tmp_file_path) == []:
                    logger.error('Channel {channel_id}: Unknown site_code gatt value - {message}'.format(
                        channel_id=str(channel_id),
                        message=contact_aims_msg))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                if not is_time_monotonic(netcdf_tmp_file_path):
                    logger.error('Channel {channel_id}: TIME value is not strictly monotonic \
                                 - {message}'.format(channel_id=str(channel_id),
                                                     message=contact_aims_msg))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                # check every single file of the list. We don't assume that if one passes, all pass ... past proved this
                wip_path = os.environ.get('data_wip_path')
                checker_retval = pass_netcdf_checker(netcdf_tmp_file_path, tests=['cf:1.6', 'imos:1.3'])
                if not checker_retval:
                    logger.error('Channel {channel_id}: File does not pass CF/IMOS compliance checker - Process of channel aborted'
                                 .format(channel_id=str(channel_id)))
                    shutil.copy(netcdf_tmp_file_path, os.path.join(wip_path, 'errors'))

                    logger.error('File copied to {path} for debugging'.format(
                        path=os.path.join(wip_path, 'errors', os.path.basename(netcdf_tmp_file_path))))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                netcdf_tmp_file_path = fix_data_code_from_filename(netcdf_tmp_file_path)
                netcdf_tmp_file_path = fix_provider_code_from_filename(netcdf_tmp_file_path, 'IMOS_ANMN')

                if re.search('IMOS_ANMN_[A-Z]{1}_', netcdf_tmp_file_path) is None:
                    logger.error('   Channel %s - File name Data code does not pass REGEX - Process of channel aborted' % str(channel_id))
                    shutil.copy(netcdf_tmp_file_path, os.path.join(wip_path, 'errors'))
                    logger.error('   File copied to %s for debugging' % (os.path.join(wip_path, 'errors', os.path.basename(netcdf_tmp_file_path))))
                    shutil.rmtree(os.path.dirname(netcdf_tmp_file_path))
                    break

                move_to_tmp_incoming(netcdf_tmp_file_path)

                if TESTING:
                    # The 2 next lines download the first month only for every single channel. This is only used for testing
                    save_channel_info(channel_id, aims_xml_info, level_qc, end_date)
                    break

            save_channel_info(channel_id, aims_xml_info, level_qc, end_date)

    else:
        logger.info('QC{level_qc} - Channel {channel_id}: already up to date'.format(channel_id=str(channel_id),
                                                                                     level_qc=str(level_qc)))


def process_qc_level(level_qc):
    """ Downloads all channels for a QC level
    level_qc(int) : 0 or 1
    """

    logger.info('Process ANMN NRS download from AIMS web service - QC level {level_qc}'.format(level_qc=level_qc))
    xml_url = 'http://data.aims.gov.au/gbroosdata/services/rss/netcdf/level{level_qc}/300'.format(level_qc=level_qc)
    try:
        aims_xml_info = parse_aims_xml(xml_url)
    except Exception as err:
        logger.critical('RSS feed not available')
        exit(1)

    for channel_id in aims_xml_info.keys():
        try:
            process_monthly_channel(channel_id, aims_xml_info, level_qc)
        except Exception as err:
            logger.error('QC{qc_level} - Channel {channel_id}: Failed, unknown reason - manual debug required'.format(
                channel_id=str(channel_id),
                qc_level=str(level_qc)))
            logger.error(traceback.print_exc())


class AimsDataValidationTest(data_validation_test.TestCase):

    def setUp(self):
        """ Check that a the AIMS system or this script hasn't been modified.
        This function checks that a downloaded file still has the same md5.
        """
        channel_id                   = '84329'
        from_date                    = '2016-01-01T00:00:00Z'
        thru_date                    = '2016-01-02T00:00:00Z'
        level_qc                     = 1
        aims_rss_val                 = 300
        xml_url                      = 'http://data.aims.gov.au/gbroosdata/services/rss/netcdf/level%s/%s' % (str(level_qc), str(aims_rss_val))

        logger.info('Data validation unittests...')
        aims_xml_info                = parse_aims_xml(xml_url)
        channel_id_info = aims_xml_info[channel_id]
        self.netcdf_tmp_file_path    = download_channel(channel_id, from_date, thru_date, level_qc)
        modify_anmn_nrs_netcdf(self.netcdf_tmp_file_path, channel_id_info)

        # force values of attributes which change all the time
        netcdf_file_obj              = Dataset(self.netcdf_tmp_file_path, 'a', format='NETCDF4')
        netcdf_file_obj.date_created = "1970-01-01T00:00:00Z"  # epoch
        netcdf_file_obj.history      = 'data validation test only'
        netcdf_file_obj.NCO          = 'NCO_VERSION'

        netcdf_file_obj.close()

    def tearDown(self):
        shutil.copy(self.netcdf_tmp_file_path, os.path.join(os.environ['data_wip_path'], 'nc_unittest_%s.nc' % self.md5_netcdf_value))
        shutil.rmtree(os.path.dirname(self.netcdf_tmp_file_path))

    def test_aims_validation(self):
        if sys.version_info[0] < 3:
            self.md5_expected_value = '76c9a595264a8173545b6dc0c518a280'
        else:
            self.md5_expected_value = '78c6386529faf9dc2272e9bed5ed7fa2'

        self.md5_netcdf_value = md5(self.netcdf_tmp_file_path)

        self.assertEqual(self.md5_netcdf_value, self.md5_expected_value)


def args():
    """
    define the script arguments
    :return: vargs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testing",
                        action='store_true',
                        help="testing only - downloads the first month of each channel")

    return parser.parse_args()


if __name__ == '__main__':
    vargs = args()
    me = singleton.SingleInstance()
    os.environ['data_wip_path'] = os.path.join(os.environ.get('WIP_DIR'),
                                               'ANMN',
                                               'NRS_AIMS_Darwin_Yongala_data_rss_download_temporary')
    global TMP_MANIFEST_DIR
    global TESTING

    set_up()

    # initialise logging
    logging_aims()
    global logger
    logger = logging.getLogger(__name__)

    # data validation test
    runner = data_validation_test.TextTestRunner()
    itersuite = data_validation_test.TestLoader().loadTestsFromTestCase(AimsDataValidationTest)
    res = runner.run(itersuite)

    if not DATA_WIP_PATH:
        logger.critical('environment variable data_wip_path is not defined.')
        exit(1)

    # script optional argument for testing only. used in process_monthly_channel
    TESTING = vargs.testing

    rm_tmp_dir(DATA_WIP_PATH)

    if len(os.listdir(ANMN_NRS_INCOMING_DIR)) >= 2:
        logger.critical('Operation aborted, too many files in INCOMING_DIR')
        exit(1)

    if len(os.listdir(ANMN_NRS_ERROR_DIR)) >= 2:
        logger.critical('Operation aborted, too many files in ERROR_DIR')
        exit(1)

    if not res.failures:
        for level in [0, 1]:
            date_str_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            TMP_MANIFEST_DIR = os.path.join(DATA_WIP_PATH, 'manifest_dir_tmp_{date}'.format(
                date=date_str_now))
            os.makedirs(TMP_MANIFEST_DIR)

            process_qc_level(level)

            lines_per_file = 2**12
            file_list = list_recursively_files_abs_path(TMP_MANIFEST_DIR)
            if len(file_list) > 0:
                for file_number, lines in groupby(enumerate(file_list), key=lambda x: x[0] // lines_per_file):
                    incoming_file = os.path.join(DATA_WIP_PATH, 'anmn_nrs_aims_FV0{level}_{date}_{file_number}.manifest'.format(
                        level=str(level),
                        date=date_str_now,
                        file_number=file_number))
                    with open(incoming_file, 'w') as outfile:
                        for item in lines:
                            outfile.write("%s\n" % item[1])

                    os.chmod(incoming_file, 0o0664)  # change to 664 for pipeline v2
                    shutil.move(incoming_file, os.path.join(ANMN_NRS_INCOMING_DIR, os.path.basename(incoming_file)))

    else:
        logger.error('Data validation unittests failed')
