import numpy as np
import numpy.ma as ma


class XbtException(Exception):
    pass


def _error(message):
    """ Raise an exception with the given message."""
    raise XbtException('{message}'.format(message=message))


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


def decode_bytearray(byte_array):
    '''
    decode a numpy masked array of bytes into a regular string
    '''
    return ''.join(chr(x) for x in bytearray(byte_array[:]).strip())
    #return ''.join([a.decode('UTF-8') for i, a in enumerate(byte_array) if not byte_array.mask[i]])


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
