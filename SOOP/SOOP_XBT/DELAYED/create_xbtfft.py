# create a text file with the XBT_fault_and_feature_type binary values and their corresponding labels
import pandas as pd
from decode_flags import convert


def create_xbtfft():
    # Read the CSV file and convert it to a DataFrame
    file_path = 'flag_quality_table.csv'
    df = pd.read_csv(file_path)
    # keep only the name, full_code
    df = df[['name', 'full_code', 'XBT_accept_code', 'XBT_reject_code']]
    # drop the rows with NaN values in the XBT_accept_code column
    dfa = df.dropna(subset=['XBT_accept_code'])
    # drop the rows with NaN values in the XBT_reject_code column
    dfr = df.dropna(subset=['XBT_reject_code'])

    # convert the name and full_code
    labels = dfa['name'].tolist()
    full_code = dfa['full_code'].tolist()
    with open('xbt_accept_code.csv', 'w') as f:
        # write the header
        f.write('label,code,byte_value\n')
        for i in range(len(dfa)):
            # write the labels, code and the binary value with a comma in between each
            f.write(f'{convert(2**i, labels)},{full_code[i]},{2**i}\n')
    labels = dfr['name'].tolist()
    full_code = dfr['full_code'].tolist()
    with open('xbt_reject_code.csv', 'w') as f:
        # write the header
        f.write('label,code,byte_value\n')
        for i in range(len(dfr)):
            # write the labels, code and the binary value with a comma in between each
            f.write(f'{convert(2**i, labels)},{full_code[i]},{2**i}\n')
    return


create_xbtfft()