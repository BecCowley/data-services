import pandas as pd
from netCDF4 import Dataset

def read_netcdf_and_output_table(netcdf_file_path):
    # Open the NetCDF file
    dataset = Dataset(netcdf_file_path, 'r')

    # Initialize a list to store the variable information
    variable_info = []

    # Iterate through the dimensions:
    for dim in dataset.dimensions:
        variable_info.append({
            'Dimension Name': dim,
            'Dimension Size': dataset.dimensions[dim].size
        })

    # Iterate through the variables in the NetCDF file
    # if for each variable, list the variable name, attributes and data type in the first iteration of the attribute loop
    # and the attributes in subsequent iterations
    for var in dataset.variables:
        variable = dataset.variables[var]
        variable_info.append({
            'Variable Name': var,
            'Variable Dimensions': variable.dimensions,
            'Data Type': variable.dtype
        })

        for att in variable.ncattrs():
            variable_info.append({
                'Attribute Name': att,
                'Attribute Value': variable.getncattr(att)
            })

    # Convert the list to a DataFrame for better readability
    df = pd.DataFrame(variable_info)

    # Initialize a list to store the global attributes
    global_attributes = []

    # Iterate through the global attributes in the NetCDF file
    for att in dataset.ncattrs():
        global_attributes.append({
            'Attribute Name': att,
            'Attribute Value': dataset.getncattr(att)
        })

    # Convert the list to a DataFrame for better readability
    global_attributes_df = pd.DataFrame(global_attributes)

    # Close the NetCDF file
    dataset.close()

    return df, global_attributes_df

# Example usage
netcdf_file_path = '/Users/cow074/code/SOOP/newIMOSformatnc/IMOS_SOOP-XBT_T_PX32_20250213180301Z_FV01_ID-RD3203_20250213180301_054.nc'
df, globaldf = read_netcdf_and_output_table(netcdf_file_path)
# output the df to a csv file
df.to_csv('/Users/cow074/code/SOOP/PyQuest/netCDF_format/netcdf_variables_info.csv', index=False)
globaldf.to_csv('netcdf_global_attributes.csv', index=False)