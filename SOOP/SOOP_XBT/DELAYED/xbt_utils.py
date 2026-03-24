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


def make_transect_id(soop_line, date_like, existing_ids):
    """
    Return a unique transect id like: soop_line-YYYYMM-I
    where I starts at 1 and increments until the id is not in existing_ids.
    """
    yyyymm = pd.to_datetime(date_like).strftime('%Y%m')
    i = 1
    while True:
        candidate = f"{soop_line}-{yyyymm}-{i}"
        if candidate not in existing_ids:
            return candidate
        i += 1


def read_globals_config(file_path):
    """
    read the global attributes from the xbt_config file
    """
    # Read the CSV file into a dictionary
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), file_path))
    # fill any empty cells and strings with NaN
    df = df.fillna(value=pd.NA)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    # convert the DataFrame to a dictionary of 'Attribute Name': 'Attribute Value' pairs
    global_att = {}
    for index, row in df.iterrows():
        # remove any leading or trailing whitespace from the attribute name
        att_name = row['Attribute Name'].strip()
        att_value = row['Attribute Value']
        if pd.isna(att_value):
            att_value = None
        global_att[att_name] = att_value
    return global_att

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
    if not all:
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

def convert_time_string(time_string, format='%Y%m%dT%H%M%S', output='datetime', outformat='%Y%m%d%H%M%S'):
    """
    convert a time string to a datetime object
    """
    try:
        if isinstance(time_string, pd.Series):
            dt = time_string.apply(lambda x: x.replace(' ', '0') if isinstance(x, str) else x)
        else:
            dt = time_string.replace(' ', '0')
            # in case it has a '.' in the string, assume it is a decimal point so convert to an integer then back to string
            if isinstance(dt, str) and '.' in dt:
                dt = str(int(float(dt)))
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
            if isinstance(dt, pd.Series):
                dt = dt.apply(lambda x: x.strftime(outformat) if not pd.isna(x) else None)
                return dt
            elif pd.isna(dt):
                return None
            elif isinstance(dt, pd.Timestamp):
                dt = dt.strftime(outformat)
                return dt
        else:
            return dt
    except:
        _error('Time string not in a valid format')

def add_launcher_variable(df):
    # add Launcher variable and assign 'LM-3A Hand-Held' if the vessel is not l'Astrolabe and date is less than 2020-11-01
    # else assign 'LM-4A Thru-Hull'

    # if profile_qc.data['Ship_name'].unique().item() contains 'Astrolabe' and date is > 2020-11-01, assign 'LM-4A Thru-Hull'
    if 'Astrolabe' in df['Ship_name'] and \
            df['TIME'][0] > datetime(2020, 11, 1):
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
    # calc diff between each pair of points
    d = np.diff(dat['TEMP'])
    # add one more so num. of rows are the same as temp df
    # d  = np.insert(d, 0, 0)
    d = np.append(d, 0)
    # points within acceptable temp bounds
    valid_data = np.where((dat['TEMP'] > -2.4) & (dat['TEMP'] < 32.) & np.abs(d <= 0.1))[0]
    # if there is no valid data, return None
    if len(valid_data) == 0:
        return None, False
    last_valid = min(valid_data[-1] + 1, len(dat) - 1)  # Use min with length to prevent going out of index bounds
    # find the first point that is not in jj
    #TODO: refine this to select the first time it happens in depth order
    first_wb = next((i for i in range(last_valid + 1, len(dat)) if i not in valid_data), None)
    flag = False
    if first_wb is not None:
        # set flag to True
        flag = True

    return first_wb, flag

def add_uncertainties(df):
    """ return the profile with added uncertainties"""

    # use standard uncertainties assigned by IQuOD procedure:
    # XBT manufacturers other than Sippican and TSK and unknown manufacturer / type:  0.2;  <= 230m: 4.6m; > 230 m: 2%
    # XBT deployed from submarines or Tsurumi - Seiki Co(TSK) manufacturer 0.15;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT Sippican manufacturer 0.1;  <= 230 m: 4.6 m; > 230 m: 2%
    # XBT deployed from aircraft 0.056
    # XCTD(pre - 1998) 0.06; 4 %
    # XCTD(post - 1998) 0.02; 2 %

    pt = int(df['PROBE_TYPE'].unique().item())
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
        year_value = df['TIME'].dt.year.astype(int).values[0]
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
    temp_uncertainty = np.ma.empty_like(df['TEMP'])
    temp_uncertainty[:] = tunc
    # depth uncertainties:
    unc = np.ma.MaskedArray(df['DEPTH'] * dunc[0], mask=False)
    if len(dunc) > 1:
        unc[df['DEPTH'] <= 230] = dunc[1]
    df['DEPTH_uncertainty'] = np.round(unc, 2)
    df['TEMP_uncertainty'] = np.round(temp_uncertainty, 2)

    # if the DEPTH or TEMP columns contain NaN values, fill the corresponding TEMP_quality_control,
    # DEPTH_quality control, TEMP_uncertainty and DEPTH_uncertainty rows with NaN
    idx = df['DEPTH'].isna()
    if idx.any():
        df.loc[idx, 'DEPTH_quality_control'] = np.nan
        df.loc[idx, 'DEPTH_uncertainty'] = np.nan
    idx = df['TEMP'].isna()
    if idx.any():
        df.loc[idx, 'TEMP_quality_control'] = np.nan
        df.loc[idx, 'TEMP_uncertainty'] = np.nan

    return df

def update_histories(dfprofile, code, software, release, dfhist, dep=0):
    """
    update the histories of the XBT data with the given code
    """
    # first read the flag quality table
    dfa, dfr = read_flag_quality_table(all=True)
    # check if the code is in one of the dfa or dfr dataframes
    if code not in dfa['full_code'].values and code not in dfr['full_code'].values:
        _error('Code {code} not found in the flag quality table'.format(code=code))

    # if code ends in 'R', it is a reject code, so we need to use the dfr dataframe
    if code.endswith('R'):
        hist_info = dfr[dfr['full_code'] == code]
    else:
        hist_info = dfa[dfa['full_code'] == code]

    # add the code to the QC_accept_code or QC_reject_code column in the dfprofile dataframe
    if code.endswith('R'):
        # if this is a 'CSR', apply the reject code to all depths above the dep index
        if hist_info['rule_direction'].values[0] == 'up':
            # get the index from the deepest point where DEPTH < 4
            dep_range = dfprofile['DEPTH'] < 4
            dep = dep_range[dep_range].index[-1]
            dfprofile.loc[:dep, 'QC_reject_code'] = hist_info['QC_reject_code'].values[0]
            dep_range = dep_range[dep_range].index
        else:
            # reject code just applied to the depth index
            dfprofile.loc[dep, 'QC_reject_code'] = hist_info['QC_reject_code'].values[0]
            dep_range = [dep]
    else:
        # accept code
        dfprofile.loc[dep, 'QC_accept_code'] = hist_info['QC_accept_code'].values[0]
        dep_range = [dep]

    # change the TEMP_quality_control to the value from the dataframe, apply to all depths indicated by the hist_info['rule_direction'] column
    # where 'up' indicates from the dep index to all the depths above it, 'down' indicates from the dep index to all the depths below it
    if hist_info['rule_direction'].values[0] == 'up':
        dfprofile.loc[:dep, 'TEMP_quality_control'] = hist_info['TEMP_quality_control'].values[0]
    elif hist_info['rule_direction'].values[0] == 'down':
        dfprofile.loc[dep:, 'TEMP_quality_control'] = hist_info['TEMP_quality_control'].values[0]
    else:
        _error('Unknown rule direction: {direction}'.format(direction=hist_info['rule_direction'].values[0]))

    # update the HISTORIES for each dep in dep_range
    for deps in dep_range:
        row_data = {'HISTORY_INSTITUTION': dfprofile['Institution'][0] if 'Institution' in dfprofile.columns else 'Unknown',
                    'HISTORY_SOFTWARE': software,
                    'HISTORY_SOFTWARE_RELEASE': release,
                    'HISTORY_DATE': datetime.now().replace(microsecond=0),
                    'HISTORY_PARAMETER': hist_info['Parameter'].values[0],
                    'HISTORY_START_DEPTH': dfprofile['DEPTH'][deps],
                    'HISTORY_QC_CODE': code,
                    'HISTORY_QC_CODE_VALUE': hist_info['TEMP_quality_control'].values[0],
                    'HISTORY_QC_CODE_DESCRIPTION': hist_info['name'].values[0]}
        # append the row to the dfhist dataframe
        dfhist.loc[len(dfhist)] = row_data

    return dfhist, dfprofile


def is_string_or_list_of_strings(obj):
    if isinstance(obj, str):
        return True
    if isinstance(obj, (list, tuple)):
        return all(isinstance(x, str) for x in obj)
    return False