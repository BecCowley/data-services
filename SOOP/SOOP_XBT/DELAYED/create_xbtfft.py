# create a text file with the XBT_fault_and_feature_type binary values and their corresponding labels
import pandas as pd
from decode_flags import convert


def create_xbtfft():
    # Read the CSV file and convert it to a DataFrame
    file_path = 'flag_quality_table.csv'
    df = pd.read_csv(file_path)
    # drop the rows with 'selected' in the rule_direction column
    df = df[df['rule_direction'] != 'selected']
    # keep some of the columns
    df = df[['name', 'full_code', 'XBT_accept_code', 'XBT_reject_code', 'TEMP_quality_control','depth', 'Parameter']]
    # replace the NaN values with 0 in depth column
    df['depth'] = df['depth'].fillna(1)
    # convert the depth and rule_direction columns to match categories in the xbt_config file
    df['depth'] = df['depth'].map({0: 'ACT_CODES_FULL_PROFILE', 1: 'ACT_CODES_TO_NEXT_FLAG', 3.6: 'ACT_CODES_SINGLE_POINT'})
    # drop the rows with NaN values in the XBT_accept_code column
    dfa = df.dropna(subset=['XBT_accept_code'])
    # drop the rows with NaN values in the XBT_reject_code column
    dfr = df.dropna(subset=['XBT_reject_code'])

    # convert the name and full_code
    labels = dfa['name'].tolist()
    full_code = dfa['full_code'].tolist()
    tempqc = dfa['TEMP_quality_control'].tolist()
    depth = dfa['depth'].tolist()
    parameter = dfa['Parameter'].tolist()
    with open('xbt_accept_code.csv', 'w') as f:
        # write the header
        f.write('label,code,byte_value,tempqc,group_label,parameter\n')
        for i in range(len(dfa)):
            # write the labels, code and the binary value with a comma in between each
            f.write(f'{convert(2**i, labels)},{full_code[i]},{2**i},{tempqc[i]},{depth[i]},{parameter[i]}\n')
    labels = dfr['name'].tolist()
    full_code = dfr['full_code'].tolist()
    tempqc = dfr['TEMP_quality_control'].tolist()
    depth = dfr['depth'].tolist()
    parameter = dfr['Parameter'].tolist()
    with open('xbt_reject_code.csv', 'w') as f:
        # write the header
        f.write('label,code,byte_value,tempqc,group_label,parameter\n')
        for i in range(len(dfr)):
            # write the labels, code and the binary value with a comma in between each
            f.write(f'{convert(2**i, labels)},{full_code[i]},{2**i},{tempqc[i]},{depth[i]},{parameter[i]}\n')
    return


create_xbtfft()