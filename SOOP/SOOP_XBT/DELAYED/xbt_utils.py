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

def read_globals_config():
    """
    read the global attributes from the xbt_config file
    """
    # Specify the file path
    file_path = 'netcdfGlobalAtts.csv'
    # Read the CSV file and convert it to a DataFrame
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), file_path))
    # fill any empty cells and strings with None
    df = df.fillna(value=pd.NA)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    return df

def read_variables_config():
    """
    read the variable attributes from the xbt_config file
    """
    # Specify the file path
    file_path = 'netcdfVars.csv'
    # Read the CSV file and convert it to a DataFrame
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), file_path))
    # fill any empty cells and strings with NaN
    df = df.fillna(value=pd.NA)
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    return df

def read_flag_quality_table(all=False):
    # Specify the file path
    # Read the CSV file and convert it to a DataFrame
    file_path = 'flag_quality_table.csv'
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),file_path))
    # drop the rows with 'selected' in the rule_direction column
    df = df[df['rule_direction'] != 'selected']
    if ~all:
        # limit to only the codes that have a 0 in the historic_extra_code column
        df = (df[df['historic_extra_code'] == 0])
    df = df.reset_index(drop=True)
    # remove the historic_extra_code column
    df = df.drop(columns=['historic_extra_code'])
    # replace the NaN values with 0 in depth column
    df['depth'] = df['depth'].fillna(1)
    # convert the depth and rule_direction columns to match categories in the xbt_config file
    df['depth'] = df['depth'].map({0: 'ACT_CODES_FULL_PROFILE', 1: 'ACT_CODES_TO_NEXT_FLAG', 3.6: 'ACT_CODES_SINGLE_POINT'})
    # drop the rows with NaN values in the XBT_accept_code column
    dfa = df.dropna(subset=['QC_accept_code'])
    # drop the rows with NaN values in the XBT_reject_code column
    dfr = df.dropna(subset=['QC_reject_code'])

    return dfa, dfr

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
            # if the result is NaT, return None
            if isinstance(dt, pd.Series):
                dt = dt.apply(lambda x: None if pd.isna(x) else x)
                return dt
            elif isinstance(dt, pd.Timestamp):
                if pd.isna(dt):
                    return None
                else:
                    return dt
        elif output == 'string':
            return dt.strftime(format)
        else:
            return dt
    except:
        _error('Time string not in a valid format')

def add_launcher_variable(df):
    # add Launcher variable and assign 'LM-3A Hand-Held' if the vessel is not l'Astrolabe and date is less than 2020-11-01
    # else assign 'LM-4A Thru-Hull'

    # if profile_qc.data['Ship_name'].unique().item() contains 'Astrolabe' and date is > 2020-11-01, assign 'LM-4A Thru-Hull'
    if 'Astrolabe' in df['Ship_name'].unique().item() and \
            df['TIME'].unique().item() > datetime(2020, 11, 1):
        df['Launcher_type'] = 'LM-4A Thru-Hull'
    else:
        df['Launcher_type'] = 'LM-3A Hand-Held'

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


def wire_break(dat):
    """
    Check for wire break in the XBT data.
    Parameters
    ----------
    dat : DataFrame
        DataFrame containing the XBT data with 'TEMP' and 'DEPTH' columns.
    -------

    """
    dat.reset_index(inplace=True)

    # calc diff between each pair of points
    d = np.diff(dat['TEMP'])
    # add one more so num. of rows are the same as temp df
    # d  = np.insert(d, 0, 0)
    d = np.append(d, 0)
    # points within acceptable temp bounds
    valid_data = np.where((dat['TEMP'] > -2.4) & (dat['TEMP'] < 32.) & np.abs(d <= 0.1))[0]
    last_valid = min(valid_data[-1] + 1, len(dat) - 1)  # Use min with length to prevent going out of index bounds
    # find the first point that is not in jj
    #TODO: refine this to select the first time it happens in depth order
    first_wb = next((i for i in range(last_valid + 1, len(dat)) if i not in valid_data), None)
    flag = False
    if first_wb is not None:
        # set flag to True
        flag = True

    return first_wb, flag