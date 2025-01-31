import numpy as np
import numpy.ma as ma
import re, os
import pandas as pd
from datetime import datetime
from configparser import ConfigParser

class XbtException(Exception):
    pass


def _error(message):
    """ Raise an exception with the given message."""
    raise XbtException('{message}'.format(message=message))


def convert_time_string(time_string, format='%Y%m%dT%H%M%S', output='datetime'):
    """
    convert a time string to a datetime object
    """
    try:
        if isinstance(time_string, pd.Series):
            dt = time_string.apply(lambda x: x.replace(' ', '0') if isinstance(x, str) else x)
        else:
            dt = time_string.replace(' ', '0')
        dt = pd.to_datetime(dt, errors='coerce', format=format)
        if output == 'datetime':
            return dt
        elif output == 'string':
            return dt.strftime(format)
        else:
            return dt
    except:
        _error('Time string not in a valid format')


def read_qc_config():
    # set up a dataframe of the codes and their values
    # codes from the new cookbook, read from csv file
    # Specify the file path
    a_file_path = os.path.join(os.path.dirname(__file__), 'xbt_accept_code.csv')
    r_file_path = os.path.join(os.path.dirname(__file__), 'xbt_reject_code.csv')

    # Read the CSV file and convert it to a DataFrame
    dfa = pd.read_csv(a_file_path)
    dfr = pd.read_csv(r_file_path)
    # merge the two dataframes
    df = pd.concat([dfa, dfr])

    return df


def invalid_to_ma_array(invalid_array, fillvalue=0):
    """
    returns a masked array from an invalid XBT variable
    """
    masked = []
    array = []
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


# Define a function to remove control characters
def remove_control_chars(s):
    return re.sub(r'[\x00-\x1F\x7F]', '', s)


def decode_bytearray(byte_array):
    '''
    decode a numpy masked array of bytes into a regular string
    '''
    if byte_array.mask.size != byte_array.data.size or len(byte_array) == 1:
        return ''.join(chr(x) for x in bytearray(byte_array[:]).strip())
    else:
        return ''.join([a.decode('UTF-8') for i, a in enumerate(byte_array) if not byte_array.mask[i]])


def temp_prof_info(netcdf_file_obj):
    """
    retrieve profile info from input NetCDF, location of TEMP and (if there) PSAL/COND profile information in the file
    """
    #TODO: if there is more than one profile (eg, from XCTD) need to handle this somewhere,
    # if the converter is to be used for other data types
    no_prof = netcdf_file_obj['No_Prof'][:]
    prof_type = dict()
    for i in range(no_prof.item()):
        prof_type[i] = decode_bytearray(netcdf_file_obj['Prof_Type'][i])

    return prof_type

def _call_parser(conf_file):
    """ parse a config file """
    parser = ConfigParser()
    parser.optionxform = str  # to preserve case
    conf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), conf_file)
    parser.read(conf_file_path)
    return parser

def read_section_from_xbt_config(section_name):
    "return all the elements in the section called section_name from the xbt_config file"
    xbt_config = _call_parser('xbt_config')
    if section_name in xbt_config.sections():
        return dict(xbt_config.items(section_name))
    elif [index for index, item in enumerate(xbt_config.sections()) if section_name in item]:
        index = [index for index, item in enumerate(xbt_config.sections()) if section_name in item][0]
        return dict(xbt_config.items(xbt_config.sections()[index]))
    else:
        _error('xbt_config file not valid. missing section: {section}'.format(section=section_name))

def _find_var_conf(parser):
    """
    list NETCDF variable names from conf file
    """

    variable_list = parser.sections()
    if 'global_attributes' in variable_list:
        variable_list.remove('global_attributes')

    return variable_list


def generate_table_att(conf_file, conf_file_point_of_truth=False):
    """
    main function to generate the attributes of a table for parquet file
    """
    parser = _call_parser(conf_file)

    variable_list = _find_var_conf(parser)
    table_att = dict()
    for var in variable_list:
        var_att = dict(parser.items(var))
        table_att[var] = var_att


    # return the dictionary of attributes
    return table_att, variable_list
