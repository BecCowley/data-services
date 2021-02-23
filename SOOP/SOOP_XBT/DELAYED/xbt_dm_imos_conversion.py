#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import difflib
import os
import sys
import tempfile
import re
from collections import OrderedDict
from configparser import ConfigParser
from datetime import datetime

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, date2num

from generate_netcdf_att import generate_netcdf_att, get_imos_parameter_info
from imos_logging import IMOSLogging
from ship_callsign import ship_callsign_list
from xbt_line_vocab import xbt_line_info


class XbtException(Exception):
    pass


def _error(message):
    """ Raise an exception with the given message."""
    raise XbtException('{message}'.format(message=message))


def _call_parser(conf_file):
    """ parse a config file """
    parser = ConfigParser()
    parser.optionxform = str  # to preserve case
    conf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), conf_file)
    parser.read(conf_file_path)
    return parser


def invalid_to_ma_array(invalid_array, fillvalue=0):
    """
    returns a masked array from an invalid XBT variable
    """
    masked = []
    array  = []
    for val in invalid_array:
        val = [''.join(chr(x)) for x in bytearray(val)][0]
        val = val.replace(' ', '')
        if val == '' or  val == '\x00':
            masked.append(True)
            array.append(np.inf)
        else:
            masked.append(False)
            array.append(int(val))

    array = ma.array(array, mask=masked, fill_value=fillvalue)
    array = ma.fix_invalid(array)
    array = ma.array(array).astype(int)
    return array


def temp_prof_info(netcdf_file_path):
    """
    retrieve profile info from input NetCDF: TODO: improve comments
    """
    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        no_prof = netcdf_file_obj['No_Prof'][0]
        for i in range(0, no_prof):
            prof_type = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Prof_Type'][:][i].data)).strip()
            if prof_type == 'TEMP':
                temp_prof = i
                break
        return no_prof, prof_type, temp_prof


def read_section_from_xbt_config(section_name):
    "return all the elements in the section called section_name from the xbt_config file"
    xbt_config = _call_parser('xbt_config')
    if section_name in xbt_config.sections():
        return dict(xbt_config.items(section_name))
    else:
        _error('xbt_config file not valid. missing section: {section}'.format(section=section_name))


def get_fallrate_eq_coef(netcdf_file_path):
    """return probe type name, coef_a, coef_b as defined in WMO1770"""
    fre_list = read_section_from_xbt_config('FRE')
    peq_list =read_section_from_xbt_config('PEQ$')
    ptyp_list = read_section_from_xbt_config('PTYP')

    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        gatts = parse_srfc_codes(netcdf_file_path)

        att_name = 'XBT_probetype_fallrate_equation'
        if att_name in list(gatts.keys()):
            item_val = gatts[att_name]
            item_val = ''.join(item_val.split())
            if item_val in list(ptyp_list.keys()):
                #old PTYP surface code, need to match up PEQ$code
                item_val = ptyp_list[item_val].split(',')[0]

            if item_val in list(fre_list.keys()):
                probetype = peq_list[item_val].split(',')[0]
                coef_a = fre_list[item_val].split(',')[0]
                coef_b = fre_list[item_val].split(',')[1]

                return probetype, item_val, float(coef_a), float(coef_b)
            else:
                coef_a = []
                coef_b = []
                probetype = []
                LOGGER.warning('{item_val} missing from FRE part in xbt_config file'.format(item_val=item_val))
                return probetype, item_val, coef_a, coef_b
        else:
            _error('XBT_probetype_fallrate_equation missing from {input_nc_path}'.format(input_nc_path=netcdf_file_path))


def get_recorder_type(netcdf_file_path):
    """
    return Recorder as defined in WMO4770
    """
    rct_list = read_section_from_xbt_config('RCT$')
    syst_list = read_section_from_xbt_config('SYST')

    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        gatts = parse_srfc_codes(netcdf_file_path)

        att_name = 'XBT_recorder_type'
        if att_name in list(gatts.keys()):
            item_val = str(int(gatts[att_name]))
            if item_val in list(syst_list.keys()):
                item_val=syst_list[item_val].split(',')[0]

            if item_val in list(rct_list.keys()):
                return item_val, rct_list[item_val].split(',')[0]
            else:
                _error('{item_val} missing from recorder type part in xbt_config file'.format(item_val=item_val))
        else:
            _error('XBT_recorder_type missing from {input_nc_path}'.format(input_nc_path=netcdf_file_path))


def parse_srfc_codes(netcdf_file_path):
    """
    Parse the surface codes in the mquest files
    """
    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        srfc_code_nc = netcdf_file_obj['SRFC_Code'][:]
        srfc_parm    = netcdf_file_obj['SRFC_Parm'][:]
        nsrfcodes    = int(netcdf_file_obj['Nsurfc'][:])

        srfc_code_list = read_section_from_xbt_config('SRFC_CODES')

        # read a list of srfc code defined in the srfc_code conf file. Create a
        # dictionary of matching values
        gatts = OrderedDict()
        for i in range(0,nsrfcodes):
            srfc_code_iter = ''.join([chr(x) for x in bytearray(srfc_code_nc[i].data)]).rstrip('\x00')
            if srfc_code_iter in list(srfc_code_list.keys()):
                att_name = srfc_code_list[srfc_code_iter].split(',')[0]
                att_type = srfc_code_list[srfc_code_iter].split(',')[1]
                att_val = ''.join([chr(x) for x in bytearray(srfc_parm[i].data)]).strip()
                if att_val.replace(' ', '') != '':
                    gatts[att_name] = att_val
                    try:
                        if att_type == 'float':
                            gatts[att_name] = float(gatts[att_name].replace(' ', ''))
                        elif att_type == 'int':
                                gatts[att_name] = int(gatts[att_name].replace(' ', ''))
                    except ValueError:
                        LOGGER.warning('"%s = %s" could not be converted to %s()' % (att_name, gatts[att_name], att_type.upper()))

            else:
                if srfc_code_iter != '':
                    LOGGER.warning('%s code is not defined in srfc_code conf file. Please edit conf' % srfc_code_iter)

        return gatts


def parse_gatts_nc(netcdf_file_path):
    """
    retrieve global attributes only for input NetCDF file
    """
    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:

        no_prof, prof_type, temp_prof = temp_prof_info(netcdf_file_path)

        cruise_id = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Cruise_ID'][:].data)).strip()
        deep_depth = netcdf_file_obj['Deep_Depth'][temp_prof]

        source_id = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Source_ID'][:].data)).replace('\x00', '').strip()
        source_id = 'AMMC' if source_id == '' else source_id
        digitisation_code = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Digit_Code'][:].data)).replace('\x00', '').strip()
        precision = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Standard'][:].data)).replace('\x00', '').strip()
        try:
            predrop_comments = ''.join(chr(x) for x in bytearray(netcdf_file_obj['PreDropComments'][:].data)).replace('\x00', '').strip()
            postdrop_comments = ''.join(chr(x) for x in bytearray(netcdf_file_obj['PostDropComments'][:].data)).replace('\x00', '').strip()
        except:
            #print('No postdrop or predrop comments in file, continuing')
            predrop_comments = ''
            postdrop_comments = ''

        gatts = parse_srfc_codes(netcdf_file_path)

        # cleaning
        att_name = 'XBT_probetype_fallrate_equation'
        if att_name in list(gatts.keys()):
            del(gatts[att_name])

        att_name = 'XBT_recorder_type'
        if att_name in list(gatts.keys()):
            recorder_val, recorder_type = get_recorder_type(netcdf_file_path)
            gatts[att_name] = recorder_val + ', ' + recorder_type

        att_name = 'XBT_height_launch_above_water_in_meters'
        if att_name in list(gatts.keys()):
            if gatts[att_name] > 50:
                LOGGER.warning('HTL$, xbt launch height attribute seems to be very high: %s meters' % gatts[att_name])

        gatts['geospatial_vertical_max'] = deep_depth.item(0)
        gatts['XBT_cruise_ID'] = cruise_id
        gatts['gts_insertion_node'] = source_id
        gatts['gtspp_digitisation_method_code'] = digitisation_code
        gatts['gtspp_precision_code'] = precision
        gatts['predrop_comments'] = predrop_comments
        gatts['postdrop_comments'] = postdrop_comments

        if INPUT_DIRNAME is None:
            gatts['XBT_input_filename'] = os.path.basename(netcdf_file_path)  # case when input is a file
        else:
            gatts['XBT_input_filename'] = netcdf_file_path.replace(os.path.dirname(INPUT_DIRNAME), '').strip('/')  # we keep the last folder name of the input as the 'database' folder

        # get xbt line information from config file
        xbt_config = _call_parser('xbt_config')
        # some files don't have line information
        isline = gatts.get('XBT_line')
        if not isline:
            gatts['XBT_line'] = 'NOLINE'
            
        xbt_line_conf_section = [s for s in xbt_config.sections() if gatts['XBT_line'] in s]
        xbt_alt_codes = [s for s in list(XBT_LINE_INFO.keys()) if XBT_LINE_INFO[s] is not None]  # alternative IMOS codes taken from vocabulary
        if xbt_line_conf_section != []:
            xbt_line_att = dict(xbt_config.items(xbt_line_conf_section[0]))
            gatts.update(xbt_line_att)
        elif gatts['XBT_line'] in xbt_alt_codes:
            xbt_line_conf_section = [s for s in xbt_config.sections() if XBT_LINE_INFO[gatts['XBT_line']] == s]
            xbt_line_att = dict(xbt_config.items(xbt_line_conf_section[0]))
            gatts.update(xbt_line_att)
        else:
            LOGGER.error('XBT line : "%s" is not defined in conf file(Please edit), or an alternative code has to be set up by AODN in vocabs.ands.org.au(contact AODN)' % gatts['XBT_line'])
            exit(1)

        return gatts


def parse_annex_nc(netcdf_file_path):
    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        data_avail = netcdf_file_obj['Data_Avail'][0]
        dup_flag = netcdf_file_obj['Dup_Flag'][0]
        nhist = netcdf_file_obj['Num_Hists'][0]
        ident_code = netcdf_file_obj['Ident_Code'][0:nhist]
        
        no_prof, prof_type, temp_prof = temp_prof_info(netcdf_file_path)
        # previous values history. same indexes and dimensions of all following vars
        act_code = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['Act_Code'][0:nhist].data if
                    bytearray(xx).strip()]
        act_code = [x.replace('\x00', '') for x in act_code]
        act_code = list(filter(None, act_code))

        act_parm = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['Act_Parm'][0:nhist].data if
                    bytearray(xx).strip()]
        act_parm = [x.replace('\x00', '') for x in act_parm]
        act_parm = list(filter(None, act_parm))

        prc_code = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['PRC_Code'][0:nhist].data if
                    bytearray(xx).strip()]
        prc_code = [x.replace('\x00', '') for x in prc_code]
        prc_code = list(filter(None, prc_code))

        prc_date = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['PRC_Date'][0:nhist].data if
                    bytearray(xx).strip()]
        prc_date = [x.replace('\x00', '') for x in prc_date]
        prc_date = list(filter(None, prc_date))

        prc_date = [date.replace(' ','0') for date in prc_date]

        prc_date = [datetime.strptime(date, '%Y%m%d') for date in prc_date]
        aux_id = netcdf_file_obj['Aux_ID'][0:nhist]  # depth value of modified act_parm var modified
        version_soft = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['Version'][0:nhist].data if
                        bytearray(xx).strip()]
        #AW Bug - problem here if the previous value string contains a non-numeric char like : which occurs when time is changed with TEA flag
        #get error converting e,g, 22:12 to a float - Python error invalid literal for float() - simple fix - strip the ':' character
        #previous_val = [float(x.replace(':','')) for x in [''.join(chr(x) for x in bytearray(xx).strip()).rstrip('\x00') for xx in
        #netcdf_file_obj.variables['Previous_Val'][0:nhist]] if x]
        
        #TODO: check this bug. Leave in place for now.
        previous_val = [float(x) for x in [''.join(chr(x) for x in bytearray(xx).strip()).rstrip('\x00') for xx in
                                           netcdf_file_obj['Previous_Val'][0:nhist]] if x]
        ident_code = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in ident_code if bytearray(xx).strip()]
        data_type = ''.join(chr(x) for x in bytearray(netcdf_file_obj['Data_Type'][:].data)).strip()
        
        #tidy up the aux_id, previous_val, etc. Remove duplicated values of CS (where there are 99.99 in previous_val)
        i99 = [i for i,x in enumerate(previous_val) if x==99.99]
        ics = [i for i,x in enumerate(act_code) if x=='CS']
        idup = list(set(i99) & set(ics))

        if any(idup):
            for index in sorted(idup, reverse=True):
                del previous_val[index]
                aux_id = np.delete(aux_id,index)
                del act_code[index]
                del act_parm[index]
                del version_soft[index]
                del ident_code[index]
                del prc_code[index]
                del prc_date[index]
        print(act_code)
        annex = {}
        annex['data_type'] = data_type
        annex['dup_flag'] = dup_flag
        annex['ident_code'] = ident_code
        annex['data_avail'] = data_avail
        annex['act_code'] = act_code
        annex['act_parm'] = act_parm
        annex['aux_id'] = aux_id
        annex['prc_code'] = prc_code
        annex['prc_date'] = prc_date
        annex['version_soft'] = version_soft
        annex['no_prof'] = no_prof
        annex['prof_type'] = prof_type
        annex['previous_val'] = previous_val

        probetype, fre_val, coef_a, coef_b = get_fallrate_eq_coef(netcdf_file_path)
        annex['fre_val'] = fre_val
        annex['probetype'] = probetype
        annex['fallrate_equation_coefficient_a'] = coef_a
        annex['fallrate_equation_coefficient_b'] = coef_b

        return annex


def parse_data_nc(netcdf_file_path):
#    LOGGER.info('Parsing data from %s' % netcdf_file_path)
    with Dataset(netcdf_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        woce_date = netcdf_file_obj['woce_date'][0]
        woce_time = netcdf_file_obj['woce_time'][0]
        q_date_time = int(netcdf_file_obj['Q_Date_Time'][0])
        latitude = netcdf_file_obj['latitude'][0]
        longitude = netcdf_file_obj['longitude'][0]
        q_pos = netcdf_file_obj['Q_Pos'][0]

        no_prof, prof_type, temp_prof = temp_prof_info(netcdf_file_path)

        # position QC - check this. Probably put a '1' in if the value is not set and the data has been QCd
        if q_pos == '1':
            q_pos = 1
        else:
            q_pos = 1  # We should have flags of '1' on the lat/long, as these have been QC'd. Although not explicit in the original netcdf files (Bec Cowley 03/2020)
	#insert zeros into dates with spaces
        xbt_date = '%sT%s' % (woce_date, str(woce_time).zfill(6))  # add leading 0
        str1 = [x.replace(' ','0') for x in xbt_date]
        xbt_date = ''.join(str1)
        xbt_date = datetime.strptime(xbt_date, '%Y%m%dT%H%M%S')

        depth_press = netcdf_file_obj['Depthpress'][temp_prof, :]
        depth_press_flag = netcdf_file_obj['DepresQ'][temp_prof, :, 0].flatten()
        depth_press_flag = np.ma.masked_array(invalid_to_ma_array(depth_press_flag, fillvalue=0))
        if isinstance(netcdf_file_obj['Profparm'][temp_prof, 0, :, 0, 0], np.ma.MaskedArray):
            prof = np.ma.masked_where(netcdf_file_obj['Profparm'][temp_prof, 0, :, 0, 0].data > 50,
                                      netcdf_file_obj['Profparm'][temp_prof, 0, :, 0, 0])
        else:
            prof = np.ma.masked_where(netcdf_file_obj['Profparm'][temp_prof, 0, :, 0, 0] > 50,
                                      netcdf_file_obj['Profparm'][temp_prof, 0, :, 0, 0])
            prof.set_fill_value(-99.99)

        prof_flag = netcdf_file_obj['ProfQP'][temp_prof, 0, :, 0, 0].flatten()
        prof_flag = np.ma.masked_array(
            invalid_to_ma_array(prof_flag, fillvalue=99))  # replace masked values for IMOS IODE flags

        data = {}
        data['LATITUDE'] = latitude
        data['LATITUDE_quality_control'] = q_pos
        data['LONGITUDE'] = longitude
        data['LONGITUDE_quality_control'] = q_pos
        data['TIME'] = xbt_date
        data['TIME_quality_control'] = q_date_time

        if isinstance(depth_press, np.ma.MaskedArray):
            data['DEPTH'] = depth_press[
                ~ma.getmask(depth_press)].flatten()  # DEPTH is a dimension, so we remove mask values, ie FillValues
            data['DEPTH_quality_control'] = depth_press_flag[~ma.getmask(depth_press)].flatten()
            data['TEMP'] = prof[~ma.getmask(depth_press)].flatten()
            data['TEMP_quality_control'] = prof_flag[~ma.getmask(depth_press)].flatten()
        else:
            data['DEPTH'] = depth_press
            data['DEPTH_quality_control'] = depth_press_flag
            data['TEMP'] = prof
            data['TEMP_quality_control'] = prof_flag

        return data


def parse_nc(netcdf_file_path):
    """ Read an edited XBT file written in an un-friendly NetCDF format
    global attributes, data and annex information are returned

    gatts, data, annex = parse_nc(netcdf_file_path)
    """
    LOGGER.info('Parsing %s' % netcdf_file_path)

    netcdf_file_path = netcdf_file_path
    gatts = parse_gatts_nc(netcdf_file_path)
    annex = parse_annex_nc(netcdf_file_path)
    data = parse_data_nc(netcdf_file_path)

    return gatts, data, annex


def raw_for_ed_path(netcdf_file_path):
    """
    for an edited NetCDF file path, return the raw NetCDF file path if exists
    """
    raw_netcdf_path = netcdf_file_path.replace('ed.nc', 'raw.nc')
    if os.path.exists(raw_netcdf_path):
        return raw_netcdf_path


def create_filename_output(gatts, data):
    filename = 'XBT_T_%s_%s_FV01_ID-%s' % (data['TIME'].strftime('%Y%m%dT%H%M%SZ'), gatts['XBT_line'], gatts['XBT_uniqueid'])
    
    #decide what prefix is required
    names = read_section_from_xbt_config('VARIOUS')
    str = names['FILENAME']
    if str == 'Cruise_ID':
        str = gatts['XBT_cruise_ID']
        filename = '{}-{}'.format(str,filename)
    else:
        if data['TIME'] > datetime(2008, 0o1, 0o1):
            filename = 'IMOS_SOOP-{}'.format(filename)

    if '/' in filename:
        LOGGER.error('The sign \'/\' is contained inside the NetCDF filename "%s". Likely '
                     'due to a slash in the XTB_line attribute. Please ammend '
                     'the XBT_line attribute in the config file for the XBT line "%s"'
                     % (filename, gatts['XBT_line']))
        exit(1)

    return filename


def check_nc_to_be_created(annex):
    """ different checks to make sure we want to create a netcdf for this profile
    """
    #sometimes we have non-XBT data in the files, skip this
    #will probably need to think about XCTD data!!

    if annex['data_type'] != 'XB':
        LOGGER.error('Profile not processed as it is not an XBT')
        return False
    
    if annex['dup_flag'] == 'D':
        LOGGER.error('Profile not processed. Tagged as duplicate in original netcdf file')
        return False

    if 'TP' in annex['act_code'] or 'DU' in annex['act_code']:
        LOGGER.error('Profile not processed. Tagged as test probe in original netcdf file')
        return False

#    if annex['no_prof'] > 1:
#        LOGGER.error('Profile not processed. No_Prof variable is greater than 0')
#        return False

    if annex['prof_type'] != 'TEMP':
        LOGGER.error('Profile not processed. Main variable is not TEMP')
        return False

    return True


def adjust_position_qc_flags(annex, data):
    """ When a 'PE' flag is present in the Act_Code, the latitude and longitude qc flags need to be adjusted"""
    #AW change distinguish between PE+LALO - flag =4 (position fail) and PE+LATI|LONG - flag 2 (position corrected)
    #AW we also should also set the time QC flag to 4 for date-time failures see func adjust_time_qc_flags() below
    #print("Annex=",annex)
    if 'PE' in annex['act_code']:
        if ('LATI' in annex['act_parm']) or ('LONG' in annex['act_parm']):
            #print("annex['act_code']1=",annex['act_code'])
            #print("annex['act_parm']1=",annex['act_parm'])
            LOGGER.info('Position correction (PEA) in original file, changing position flags to level 2.')

            data['LATITUDE_quality_control'] = 2
            data['LONGITUDE_quality_control'] = 2
        if 'LALO' in annex['act_parm']:

            LOGGER.info('Position failure (PER) in original file, changing position flags to level 4.')
            data['LATITUDE_quality_control'] = 4
            data['LONGITUDE_quality_control'] = 4
    
    return data

def adjust_time_qc_flags(annex, data):
    #AW Add function  we also should also set the time QC flag to 4 for date-time failures TE in annex['act_code'] + DATI in annex['act_parm']
    #or set time QC to flag 2 if date/time has been corrected
    #print("Annex=",annex)
    if 'TE' in annex['act_code'] and 'DATI' in annex['act_parm']:
        LOGGER.info('Date-Time failure (TER) in original file, setting time qc flag to level 4.')
        data['TIME_quality_control'] = 4
        
    if 'TE' in annex['act_code'] and ('TIME' in annex['act_parm'] or 'DATE' in annex['act_parm']):
        LOGGER.info('Date and/or Time has been corrected (TEA) in original file, setting time qc flag to level 2.')
        data['TIME_quality_control'] = 2
    return data
    
def generate_xbt_gatts_nc(gatts, data, annex, output_folder):
    """
    generate the global attributes of a NetCDF file
    returns path of NetCDF
    """
    #AW changes - we want to organise output by folders named by cruiseid
    #e.g. <CRUISEID>/<CRUISEID>_<Date_time>-<uniqueid>.nc
    #Make a folder with name from gatts['XBT_cruise_ID'] if it does not exist and make that the output_folder
    cid=gatts['XBT_cruise_ID']
    outpath="%s%s" % (output_folder,cid)
    #print("outpath",outpath)
    
    if os.path.isdir(outpath):
        #print("folder already exists",outpath)
        output_folder=outpath
    else: #make new folder
        os.makedirs(outpath)
        output_folder=outpath
    #AW end changes
    
    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(gatts, data))

    with Dataset(netcdf_filepath, "w", format="NETCDF4") as output_netcdf_obj:
        # set global attributes
        for gatt_name in list(gatts.keys()):
            setattr(output_netcdf_obj, gatt_name, gatts[gatt_name])

        # this will overwrite the value found in the original NetCDF file
        ships = SHIP_CALL_SIGN_LIST
        if gatts['Platform_code'] in ships:
            output_netcdf_obj.ship_name = ships[gatts['Platform_code']]
            output_netcdf_obj.Callsign = gatts['Platform_code']
        elif difflib.get_close_matches(gatts['Platform_code'], ships, n=1, cutoff=0.8) != []:
            output_netcdf_obj.Callsign = difflib.get_close_matches(gatts['Platform_code'], ships, n=1, cutoff=0.8)[0]
            output_netcdf_obj.Platform_code = output_netcdf_obj.Callsign
            output_netcdf_obj.ship_name = ships[output_netcdf_obj.Callsign]
            LOGGER.warning('Vessel call sign %s seems to be wrong. Using the closest match to the AODN vocabulary: %s' % (
                gatts['Platform_code'], output_netcdf_obj.Callsign))
        else:
            LOGGER.warning('Vessel call sign %s is unknown in AODN vocabulary, Please contact info@aodn.org.au' % gatts[
                'Platform_code'])

        output_netcdf_obj.date_created = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(data['DEPTH'], np.ma.MaskedArray):
            output_netcdf_obj.geospatial_vertical_min = np.ma.MaskedArray.min(data['DEPTH']).item(0)
            output_netcdf_obj.geospatial_vertical_max = np.ma.MaskedArray.max(data['DEPTH']).item(0)
        else:
            output_netcdf_obj.geospatial_vertical_min = min(data['DEPTH'])
            output_netcdf_obj.geospatial_vertical_max = max(data['DEPTH'])

        output_netcdf_obj.geospatial_lat_min = data['LATITUDE']
        output_netcdf_obj.geospatial_lat_max = data['LATITUDE']
        output_netcdf_obj.geospatial_lon_min = data['LONGITUDE']
        output_netcdf_obj.geospatial_lon_max = data['LONGITUDE']
        output_netcdf_obj.time_coverage_start = data['TIME'].strftime('%Y-%m-%dT%H:%M:%SZ')
        output_netcdf_obj.time_coverage_end = data['TIME'].strftime('%Y-%m-%dT%H:%M:%SZ')

        setattr(output_netcdf_obj, 'XBT_recorder_type',
                "WMO Code table 4770 code \"{xbt_recorder_type}\"".
                format(xbt_recorder_type=gatts['XBT_recorder_type']))

    return netcdf_filepath


def generate_xbt_nc(gatts_ed, data_ed, annex_ed, output_folder, *argv):
    """create an xbt profile"""

    is_raw_parsed = False
    if len(argv) > 0:
        for arg in argv:
            data_raw = arg[1]
            annex_raw = arg[2]
            is_raw_parsed = True

    netcdf_filepath = os.path.join(output_folder, "%s.nc" % create_filename_output(gatts_ed, data_ed))

    netcdf_filepath = generate_xbt_gatts_nc(gatts_ed, data_ed, annex_ed, output_folder)
    LOGGER.info('Creating output %s' % netcdf_filepath)

    # adjust lat lon qc flags if required
    data_ed = adjust_position_qc_flags(annex_ed, data_ed)
    #adjust date and time QC flags if required
    data_ed= adjust_time_qc_flags(annex_ed, data_ed)

    with Dataset(netcdf_filepath, "a", format="NETCDF4") as output_netcdf_obj:
        var_time = output_netcdf_obj.createVariable("TIME", "d", fill_value=get_imos_parameter_info('TIME', '_FillValue'))
        output_netcdf_obj.createVariable("TIME_quality_control", "b", fill_value=99)
        
        
        output_netcdf_obj.createVariable("LATITUDE", "f", fill_value=get_imos_parameter_info('LATITUDE', '_FillValue'))
        output_netcdf_obj.createVariable("LATITUDE_quality_control", "b", fill_value=99)

        output_netcdf_obj.createVariable("LONGITUDE", "f", fill_value=get_imos_parameter_info('LONGITUDE', '_FillValue'))
        output_netcdf_obj.createVariable("LONGITUDE_quality_control", "b", fill_value=99)

        # append the raw data to the file
        if is_raw_parsed:
            output_netcdf_obj.createDimension("DEPTH_RAW", data_raw["DEPTH"].size)
            output_netcdf_obj.createVariable("DEPTH_RAW", "f", "DEPTH_RAW")
            output_netcdf_obj.createVariable("DEPTH_RAW_quality_control", "b", "DEPTH_RAW")

            # set DEPTH fallrate equation coef as attributes
            setattr(output_netcdf_obj['DEPTH_RAW'],
                    'fallrate_equation_coefficient_a', annex_raw['fallrate_equation_coefficient_a'])
            setattr(output_netcdf_obj['DEPTH_RAW'],
                    'fallrate_equation_coefficient_b', annex_raw['fallrate_equation_coefficient_b'])

            XBT_probetype_fallrate_equation_DEPTH_RAW_msg = "WMO Code Table 1770 \"probe={probetype},code={fre_val},a={coef_a},b={coef_b}\"".\
                format(probetype=annex_raw['probetype'],fre_val=annex_raw['fre_val'],
                       coef_a=annex_raw['fallrate_equation_coefficient_a'],
                       coef_b=annex_raw['fallrate_equation_coefficient_b'])

        output_netcdf_obj.createDimension("DEPTH", data_ed["DEPTH"].size)
        output_netcdf_obj.createVariable("DEPTH", "f", "DEPTH")
        output_netcdf_obj.createVariable("DEPTH_quality_control", "b", "DEPTH")

        # set DEPTH fallrate equation coef as attributes
        setattr(output_netcdf_obj['DEPTH'],
                'fallrate_equation_coefficient_a', annex_ed['fallrate_equation_coefficient_a'])
        setattr(output_netcdf_obj['DEPTH'],
                'fallrate_equation_coefficient_b', annex_ed['fallrate_equation_coefficient_b'])

        XBT_probetype_fallrate_equation_DEPTH_msg = "WMO Code Table 1770 \"probe={probetype},code={fre_val},a={coef_a},b={coef_b}\"".\
            format(probetype=annex_ed['probetype'],fre_val=annex_ed['fre_val'],
            coef_a=annex_ed['fallrate_equation_coefficient_a'],
            coef_b=annex_ed['fallrate_equation_coefficient_b'])

        # append the raw TEMP to the file
        if is_raw_parsed:
            output_netcdf_obj.createVariable("TEMP_RAW", "f", ["DEPTH_RAW"],
                                             fill_value=get_imos_parameter_info('TEMP', '_FillValue'))
            output_netcdf_obj.createVariable("TEMP_RAW_quality_control", "b", ["DEPTH_RAW"],
                                             fill_value=data_raw['TEMP_quality_control'].fill_value)

            conf_file_generic = os.path.join(os.path.dirname(__file__), 'generate_nc_raw_file_att')
            generate_netcdf_att(output_netcdf_obj, conf_file_generic, conf_file_point_of_truth=True)

            # rename keys in raw data
            data_raw['TEMP_RAW'] = data_raw.pop('TEMP')
            data_raw['TEMP_RAW_quality_control'] = data_raw.pop('TEMP_quality_control')
            data_raw['DEPTH_RAW'] = data_raw.pop('DEPTH')
            data_raw['DEPTH_RAW_quality_control'] = data_raw.pop('DEPTH_quality_control')

            for var in list(data_raw.keys()):
                if var in ['DEPTH_RAW', 'TEMP_RAW', 'DEPTH_RAW_quality_control', 'TEMP_RAW_quality_control']:
                    output_netcdf_obj[var][:] = data_raw[var]

            #now TEMP
            output_netcdf_obj.createVariable("TEMP", "f", ["DEPTH"], fill_value=99)
            output_netcdf_obj.createVariable("TEMP_quality_control", "b", ["DEPTH"], fill_value=data_ed['TEMP_quality_control'].fill_value)

        # this is done at the end to have those gatts next to each others (once raw data is potentially handled)
        setattr(output_netcdf_obj, 'XBT_probetype_fallrate_equation_DEPTH',
                XBT_probetype_fallrate_equation_DEPTH_msg)
        if 'XBT_probetype_fallrate_equation_DEPTH_RAW_msg' in locals():
            setattr(output_netcdf_obj, 'XBT_probetype_fallrate_equation_DEPTH_RAW', XBT_probetype_fallrate_equation_DEPTH_RAW_msg)

        # Create the unlimited time dimension:
        output_netcdf_obj.createDimension('N_HISTORY', None)
        # create HISTORY variable set associated
        output_netcdf_obj.createVariable("HISTORY_INSTITUTION", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_STEP", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_SOFTWARE", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_SOFTWARE_RELEASE", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_DATE", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_PARAMETER", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_START_DEPTH", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_STOP_DEPTH", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_PREVIOUS_VALUE", "f", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_QC_FLAG", "str", 'N_HISTORY')
        output_netcdf_obj.createVariable("HISTORY_QC_FLAG_DESCRIPTION", "str", 'N_HISTORY')

        conf_file_generic = os.path.join(os.path.dirname(__file__), 'generate_nc_file_att')
        generate_netcdf_att(output_netcdf_obj, conf_file_generic, conf_file_point_of_truth=True)

        ############# HISTORY vars
        # For both edited and raw. Could probably do all this better, but here it is for now.
        act_code_full_profile = read_section_from_xbt_config('ACT_CODES_FULL_PROFILE')
        act_code_single_point = read_section_from_xbt_config('ACT_CODES_SINGLE_POINT')
        act_code_next_flag = read_section_from_xbt_config('ACT_CODES_TO_NEXT_FLAG')
        act_code_both = read_section_from_xbt_config('ACT_CODES_BOTH')
        act_code_list = {**act_code_full_profile, **act_code_single_point, **act_code_next_flag, **act_code_both}

        # edited file
        if annex_ed['prc_date']: #only do this if there are history records in the file
            for idx, date in enumerate(annex_ed['prc_date']):
                if annex_ed['act_code'][idx] in act_code_list:
                    act_code_def = act_code_list[annex_ed['act_code'][idx]]
                else:
                    act_code_def = annex_ed['act_code'][idx]
                    LOGGER.warning("ACT CODE \"%s\" is not defined. Please edit config file" % annex_ed['act_code'][idx])
                
                output_netcdf_obj["HISTORY_QC_FLAG_DESCRIPTION"][idx] = act_code_def
                #update variable names to match what is in the file
                if 'TEMP' in annex_ed['act_parm'][idx]:
                    annex_ed['act_parm'][idx] = 'TEMP'
                if 'DEPH' in annex_ed['act_parm'][idx]:
                    annex_ed['act_parm'][idx] = 'DEPTH'
                if 'LATI' in annex_ed['act_parm'][idx]:
                    annex_ed['act_parm'][idx] = 'LATITUDE'
                if 'LONG' in annex_ed['act_parm'][idx]:
                    annex_ed['act_parm'][idx] = 'LONGITUDE'
                    
                #update institute names to be more descriptive: set up for BOM and CSIRO only at the moment
                if 'CS' in annex_ed['ident_code'][idx]:
                    annex_ed['ident_code'][idx] = 'CSIRO'
                if 'BO' in annex_ed['ident_code'][idx]:
                    annex_ed['ident_code'][idx] = 'Australian Bureau of Meteorology'
                    
                #set the software value to 2.0 for CS flag as we are keeping them in place and giving a flag of 3
                if 'CS' in annex_ed['act_code'][idx]:
                    annex_ed['version_soft'][idx] = '2.0'
            

            history_date_obj = date2num(annex_ed['prc_date'],
                output_netcdf_obj['HISTORY_DATE'].units,
                output_netcdf_obj['HISTORY_DATE'].calendar)
                                        
            # sort the flags by depth order to help with histories
            idx_sort = sorted(range(len(annex_ed['aux_id'])), key=lambda k: annex_ed['aux_id'][k])
            vals = data_ed['DEPTH'].data
            qcvals_temp = data_ed['TEMP_quality_control'].data
            qcvals_depth = data_ed['DEPTH_quality_control'].data
            for idx in idx_sort:
                # slicing over VLEN variable -> need a for loop
                output_netcdf_obj["HISTORY_INSTITUTION"][idx] = annex_ed['ident_code'][idx]
                output_netcdf_obj["HISTORY_STEP"][idx] = annex_ed['prc_code'][idx]
                names = read_section_from_xbt_config('VARIOUS')
                output_netcdf_obj["HISTORY_SOFTWARE"][idx] = names['HISTORY_SOFTWARE']
                output_netcdf_obj["HISTORY_SOFTWARE_RELEASE"][idx] = annex_ed['version_soft'][idx]
                output_netcdf_obj["HISTORY_DATE"][idx] = history_date_obj[idx]
                output_netcdf_obj["HISTORY_PARAMETER"][idx] = annex_ed['act_parm'][idx]
                output_netcdf_obj["HISTORY_PREVIOUS_VALUE"][idx] = annex_ed['previous_val'][idx]
                output_netcdf_obj["HISTORY_QC_FLAG"][idx] = annex_ed['act_code'][idx]

                #QC,RE, TE, PE and EF flag applies to entire profile
                res = annex_ed['act_code'][idx] in act_code_full_profile
                if res:
                    output_netcdf_obj["HISTORY_START_DEPTH"][idx] = vals[0]
                    output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                    continue
                    
                # Find stop depth depending on which flags are in place
                start_idx =  np.int_(np.where(np.logical_and(vals <= annex_ed['aux_id'][idx]+0.1, vals >= annex_ed['aux_id'][idx]-0.1)))
                # make the start depth equal to actual depth in depth array
                output_netcdf_obj["HISTORY_START_DEPTH"][idx] = vals[start_idx]

                #find next deepest flag depth
                stop_depth = [i for i in annex_ed['aux_id'] if i > annex_ed['aux_id'][idx]]
                # if the flag is in act_code_single_point list, then stop depth is same as start
                res = annex_ed['act_code'][idx] in act_code_single_point
                if res:
                    output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = annex_ed['aux_id'][idx]
                    continue
                    
                # if the flag is in act_code_next_flag, then stop depth is the next depth or bottom
                res = annex_ed['act_code'][idx] in act_code_next_flag
                if res:
                    if stop_depth:  # if not the last flag, next greatest depth
                        stop_idx =  np.int_(np.where(np.logical_and(vals <= stop_depth[0]+0.1, vals >= stop_depth[0]-0.1)))
                        stopdepth = vals[stop_idx-1]
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = stopdepth
                    else:
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                    continue

                # if the flag is in act_code_both, then stop depth depends on flag_severity
                res = annex_ed['act_code'][idx] in act_code_both
                if res:
                    # get the right set of flags to suit the QC flag
                    if 'TEMP' in annex_ed['act_parm'][idx]:
                        flags = qcvals_temp
                    else:
                        flags = qcvals_depth
                    flag = flags[start_idx]
                    if flag in [1,2,5]: #single point, same stop depth
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = annex_ed['aux_id'][idx]
                    elif stop_depth:  # if not the last flag, next greatest depth
                        stop_idx =  np.int_(np.where(np.logical_and(vals <= stop_depth[0]+0.1, vals >= stop_depth[0]-0.1)))
                        stopdepth = vals[stop_idx-1]
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = stopdepth
                    else:
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                    continue

            # raw file, only do this if there are flags to add from the raw file
            if np.logical_and(is_raw_parsed, annex_raw['aux_id'][:]):
                for idx, date in enumerate(annex_raw['prc_date']):
                    if annex_raw['act_code'][idx] in act_code_list:
                        act_code_def = act_code_list[annex_raw['act_code'][idx]]
                    else:
                        act_code_def = annex_raw['act_code'][idx]
                        LOGGER.warning("ACT CODE \"%s\" is not defined. Please edit config file" % annex_raw['act_code'][idx])
                        
                    output_netcdf_obj["HISTORY_QC_FLAG_DESCRIPTION"][idx] = act_code_def
                    #update variable names to match what is in the file
                    if 'TEMP' in annex_raw['act_parm'][idx]:
                        annex_raw['act_parm'][idx] = 'TEMP_RAW'
                    if 'DEPH' in annex_raw['act_parm'][idx]:
                        annex_raw['act_parm'][idx] = 'DEPTH_RAW'
                    if 'LATI' in annex_raw['act_parm'][idx]:
                        annex_raw['act_parm'][idx] = 'LATITUDE'
                    if 'LONG' in annex_raw['act_parm'][idx]:
                        annex_raw['act_parm'][idx] = 'LONGITUDE'
                        
                    #update institute names to be more descriptive: set up for BOM and CSIRO only at the moment
                    if 'CS' in annex_raw['ident_code'][idx]:
                        annex_raw['ident_code'][idx] = 'CSIRO'
                    if 'BO' in annex_raw['ident_code'][idx]:
                        annex_raw['ident_code'][idx] = 'Australian Bureau of Meteorology'
                        
                    #set the software value to 2.0 for CS flag as we are keeping them in place and giving a flag of 3
                    if 'CS' in annex_raw['act_code'][idx]:
                        annex_raw['version_soft'][idx] = '2.0'
                        
                history_date_obj = date2num(annex_raw['prc_date'],
                        output_netcdf_obj['HISTORY_DATE'].units,
                        output_netcdf_obj['HISTORY_DATE'].calendar)
                                            
                # sort the flags by depth order to help with histories
                idx_sort = sorted(range(len(annex_raw['aux_id'])), key=lambda k: annex_raw['aux_id'][k])
                vals = data_raw['DEPTH'].data
                qcvals_temp = data_raw['TEMP_quality_control'].data
                qcvals_depth = data_raw['DEPTH_quality_control'].data
                for idx in idx_sort:
                    # slicing over VLEN variable -> need a for loop
                    output_netcdf_obj["HISTORY_INSTITUTION"][idx] = annex_raw['ident_code'][idx]
                    output_netcdf_obj["HISTORY_STEP"][idx] = annex_raw['prc_code'][idx]
                    output_netcdf_obj["HISTORY_SOFTWARE"][idx] = get_history_val()
                    output_netcdf_obj["HISTORY_SOFTWARE_RELEASE"][idx] = annex_raw['version_soft'][idx]
                    output_netcdf_obj["HISTORY_DATE"][idx] = history_date_obj[idx]
                    output_netcdf_obj["HISTORY_PARAMETER"][idx] = annex_raw['act_parm'][idx]
                    output_netcdf_obj["HISTORY_PREVIOUS_VALUE"][idx] = annex_raw['previous_val'][idx]
                    output_netcdf_obj["HISTORY_QC_FLAG"][idx] = annex_raw['act_code'][idx]

                    #QC,RE, PE, TE and EF flag applies to entire profile
                    res = annex_raw['act_code'][idx] in act_code_full_profile
                    if res:
                        output_netcdf_obj["HISTORY_START_DEPTH"][idx] = vals[0]
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                        continue
                        
                    # Find stop depth depending on which flags are in place
                    start_idx =  np.int_(np.where(np.logical_and(vals <= annex_ed['aux_id'][idx]+0.1, vals >= annex_ed['aux_id'][idx]-0.1)))
                    # make the start depth equal to actual depth in depth array
                    output_netcdf_obj["HISTORY_START_DEPTH"][idx] = vals[start_idx]

                    #find next deepest flag depth
                    stop_depth = [i for i in annex_raw['aux_id'] if i > annex_raw['aux_id'][idx]]
                    # if the flag is in act_code_single_point list, then stop depth is same as start
                    res = annex_raw['act_code'][idx] in act_code_single_point
                    if res:
                        output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = annex_raw['aux_id'][idx]
                    continue
                    
                    # if the flag is in act_code_next_flag, then stop depth is the next depth or bottom
                    res = annex_raw['act_code'][idx] in act_code_next_flag
                    if res:
                        if stop_depth:  # if not the last flag, next greatest depth
                            stop_idx =  np.int_(np.where(np.logical_and(vals <= stop_depth[0]+0.1, vals >= stop_depth[0]-0.1)))
                            stopdepth = vals[stop_idx-1]
                            output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = stopdepth
                        else:
                            output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                        continue

                    # if the flag is in act_code_both, then stop depth depends on flag_severity
                    res = annex_raw['act_code'][idx] in act_code_both
                    if res:
                        # get the right set of flags to suit the QC flag
                        if 'TEMP' in annex_raw['act_parm'][idx]:
                            flags = qcvals_temp
                        else:
                            flags = qcvals_depth
                        flag = flags[start_idx]
                        if flag in [1,2,5]: #single point, same stop depth
                            output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = annex_raw['aux_id'][idx]
                        elif stop_depth:  # if not the last flag, next greatest depth
                            stop_idx =  np.int_(np.where(np.logical_and(vals <= stop_depth[0]+0.1, vals >= stop_depth[0]-0.1)))
                            stopdepth = vals[stop_idx-1]
                            output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = stopdepth
                        else:
                            output_netcdf_obj["HISTORY_STOP_DEPTH"][idx] = output_netcdf_obj.geospatial_vertical_max
                        continue

        for var in list(data_ed.keys()):
            if var == 'TIME':
                time_val_dateobj = date2num(data_ed['TIME'], output_netcdf_obj['TIME'].units, output_netcdf_obj['TIME'].calendar)
                var_time[:]      = time_val_dateobj
            else:
                #if isinstance(data_ed[var], np.ma.MaskedArray):
                output_netcdf_obj[var][:] = data_ed[var]#.data

        # default value for abstract
        if not hasattr(output_netcdf_obj, 'abstract'):
            setattr(output_netcdf_obj, 'abstract', output_netcdf_obj.title)

    # cleaning TEMPERATURE data
    if is_raw_parsed:
        netcdf_filepath = clean_temp_val(netcdf_filepath, annex_ed, annex_raw)
    else:
        netcdf_filepath = clean_temp_val(netcdf_filepath, annex_ed)

    return netcdf_filepath


def clean_temp_val(netcdf_filepath, annex_ed, *argv):
    """
    From Bec:
    HISTORY_PREVIOUS_VALUE: I would like to restore the temperature values that are associated with
    the 'CS' (surface spike removed) flag. That means identifying them, putting them back into the
    TEMP_ADJUSTED field, then putting a flag of 3 (probably bad) on them. The values can also stay
    in the HISTORY_PREVIOUS_VALUE field. This process would need to apply to both the TEMP_ADJUSTED
    and TEMP (from the *raw.nc file).
    """
    is_raw_parsed = False
    if len(argv) > 0:
        annex_raw = argv[0]
        is_raw_parsed = True

    with Dataset(netcdf_filepath, "a", format="NETCDF4") as output_netcdf_obj:

        ## first part, editing ADJUSTED TEMP values
        # index of Surface Spike removed and TEMP parameter
        idx_ed_cs_flag = [aa and bb for aa, bb in zip(['CS' == a for a in annex_ed['act_code']],
                                                      ['TEMP' == a for a in annex_ed['act_parm']])]

        depth_ed_flags_val = annex_ed['aux_id'][:]
        param_ed_flags_val = annex_ed['previous_val'][:]

        for idx, ii_logic in enumerate(idx_ed_cs_flag):
            if ii_logic:
                idx_val_to_modify = np.logical_and(output_netcdf_obj["DEPTH"][:] <= depth_ed_flags_val[idx]+0.1,output_netcdf_obj["DEPTH"][:] >= depth_ed_flags_val[idx]-0.1)
                if sum(idx_val_to_modify) > 1:
                    _error("Cleaning TEMP: more than one depth value matching") #TODO improve msg
                elif sum(idx_val_to_modify) == 0:
                        _error("no CS flags in file, check QC!!")
                else:
                    output_netcdf_obj['TEMP'][idx_val_to_modify] = param_ed_flags_val[idx]
                    output_netcdf_obj['TEMP_quality_control'][idx_val_to_modify] = '3'
                    

        ## second part, editing TEMP values
        if is_raw_parsed:
            ## editing TEMP_RAW values if required
            # index of Surface Spike removed and TEMP parameter
            idx_raw_cs_flag = [aa and bb for aa, bb in zip(['CS' == a for a in annex_raw['act_code']],
                                                          ['TEMP_RAW' == a for a in annex_raw['act_parm']])]
            depth_raw_flags_val = annex_raw['aux_id'][:]
            param_raw_flags_val = annex_raw['previous_val'][:]

            for idx, ii_logic in enumerate(idx_raw_cs_flag):
                if ii_logic:
                    idx_val_to_modify = np.logical_and(output_netcdf_obj["DEPTH_RAW"][:] <= depth_ed_flags_val[idx]+0.1,output_netcdf_obj["DEPTH_RAW"][:] >= depth_ed_flags_val[idx]-0.1)
                    if sum(idx_val_to_modify) > 1:
                        _error("Cleaning TEMP_RAW: more than one depth value matching") #TODO improve msg
                    elif sum(idx_val_to_modify) == 0:
                            _error("no depth value matching") #TODO improve msg
                    else:
                        output_netcdf_obj['TEMP_RAW'][idx_val_to_modify] = param_raw_flags_val[idx]
                        output_netcdf_obj['TEMP_RAW_quality_control'][idx_val_to_modify] = '3'
    return netcdf_filepath


def parse_keys_nc(keys_file_path):
    """
    input: path to *.keys.nc
    output: data dictionary containing unique values of station_number (to be used in edited and raw NetCDF to match
            with the SRFC_Parm value
    """
    with Dataset(keys_file_path, 'r', format='NETCDF4') as netcdf_file_obj:
        station_number = [''.join(chr(x) for x in bytearray(xx)).strip() for xx in netcdf_file_obj['stn_num'][:].data if
                          bytearray(xx).strip()]
        station_number = list(set(station_number))  # make sure we have a unique list of IDs. Sometimes they are repeated in the keys file (a fault in some of them)

        data = {}
        data['station_number'] = [int(x) for x in station_number]  # station_number values are integers
        return data


def args():
    """ define input argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-xbt-campaign-path', type=str,
                        help="path to *_keys.nc or campaign folder below the keys.nc file")
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

    if not os.path.exists(vargs.input_xbt_campaign_path):
        msg = '%s not a valid path' % vargs.input_xbt_campaign_path
        print(msg, file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(vargs.output_folder):
        os.makedirs(vargs.output_folder)

    return vargs


def process_xbt_file(xbt_file_path, output_folder):
    ed_nc_path = xbt_file_path
    gatts_ed, data_ed, annex_ed = parse_nc(ed_nc_path)

    if check_nc_to_be_created(annex_ed):

        # parse raw file if exists and append the raw data to the new xbt file
        raw_nc_path = raw_for_ed_path(ed_nc_path)
        vargs = None
        if raw_nc_path:
            gatts_raw, data_raw, annex_raw = parse_nc(raw_nc_path)
            vargs = (gatts_raw, data_raw, annex_raw)

        return generate_xbt_nc(gatts_ed, data_ed, annex_ed,output_folder,
                               vargs)

    return


def retrieve_keys_campaign_path(vargs):
    """
    find the keys.nc file above the input folder (root folder)
    since vargs.input_xbt_campaign_path can either be a _keys.nc or the campaign folder
    """
    if vargs.input_xbt_campaign_path.endswith('_keys.nc'):
        keys_file_path = vargs.input_xbt_campaign_path
        input_xbt_campaign_path = keys_file_path.replace('_keys.nc', '')
    else:
        keys_file_path = '{campaign_path}_keys.nc'.format(campaign_path=vargs.input_xbt_campaign_path.rstrip(os.path.sep))
        input_xbt_campaign_path = vargs.input_xbt_campaign_path

    if not os.path.exists(keys_file_path):
        msg = '{keys_file_path} does not exist%s\nProcess aborted'.format(keys_file_path=keys_file_path)
        print(msg, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(input_xbt_campaign_path):
        msg = '{input_xbt_campaign_path} does not exist%s\nProcess aborted'.format(keys_file_path=input_xbt_campaign_path)
        print(msg, file=sys.stderr)
        sys.exit(1)

    return keys_file_path, input_xbt_campaign_path


def global_vars(vargs):
    global LOGGER
    logging = IMOSLogging()
    LOGGER  = logging.logging_start(vargs.log_file)

    global NETCDF_FILE_PATH  # defined as glob to be used in exception

    global SHIP_CALL_SIGN_LIST
    SHIP_CALL_SIGN_LIST = ship_callsign_list()  # AODN CALLSIGN vocabulary

    global XBT_LINE_INFO
    XBT_LINE_INFO = xbt_line_info()

    global INPUT_DIRNAME  # in the case we're processing a directory full of NetCDF's and not ONE NetCDF only
    INPUT_DIRNAME = None


if __name__ == '__main__':
    """
    Example:
    ./xbt_dm_imos_conversion.py -i XBT/GTSPPmer2017/GTSPPmer2017MQNC_keys.nc -o /tmp/xb
    ./xbt_dm_imos_conversion.py -i XBT/GTSPPmer2017/GTSPPmer2017MQNC -o /tmp/xb
    """
    os.umask(0o002)
    vargs = args()
    global_vars(vargs)

    keys_file_path, input_xbt_campaign_path = retrieve_keys_campaign_path(vargs)
    keys_info = parse_keys_nc(keys_file_path)
    
    for f in keys_info['station_number']:
        INPUT_DIRNAME = input_xbt_campaign_path
        fpath = '/'.join(re.findall('..',str(f)))+'ed.nc'
        fname = os.path.join(input_xbt_campaign_path, fpath)

        #edited_nc = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_xbt_campaign_path)
        #    for f in filenames if f.endswith('ed.nc')]
        NETCDF_FILE_PATH = fname

        if os.path.isfile(fname):
            path = process_xbt_file(fname, vargs.output_folder)
        else:
            LOGGER.warning('file %s is in keys file, but does not exist' % f)
