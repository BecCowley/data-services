# create a text file with the XBT_fault_and_feature_type binary values and their corresponding labels
import pandas as pd
from decode_flags import convert


def create_xbtfft():
    # Read the CSV file and convert it to a DataFrame
    file_path = 'flag_quality_table.csv'
    df = pd.read_csv(file_path)
    # keep only the name, full_code
    df = df[['name', 'full_code']]
    # convert the name and full_code
    labels = df['name'].tolist()
    code = df['full_code'].tolist()
    with open('xbt_fault_and_feature_type.csv', 'w') as f:
        # write the header
        f.write('label,code,byte_value\n')
        for i in range(len(df)):
            # write the labels, code and the binary value with a comma in between each
            f.write(f'{convert(2**i, labels)},{code[i]},{2**i}\n')
    return


create_xbtfft()