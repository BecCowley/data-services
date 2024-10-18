import numpy as np
import numpy.ma as ma
import re, os
import pandas as pd


class XbtException(Exception):
    pass


def _error(message):
    """ Raise an exception with the given message."""
    raise XbtException('{message}'.format(message=message))

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