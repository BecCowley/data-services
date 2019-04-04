import datetime
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import xlrd
from dateutil import parser as dt_parser
from netCDF4 import Dataset, date2num, stringtochar

from common import param_mapping_parser, set_var_attr, set_glob_attr, read_metadata_file
from generate_netcdf_att import generate_netcdf_att

logger = logging.getLogger(__name__)

BOM_WAVE_PARAMETER_MAPPING = os.path.join(os.path.dirname(__file__), 'bom_wave_dm_parameters_mapping.csv')
NC_ATT_CONFIG = os.path.join(os.path.dirname(__file__), 'generate_nc_file_att')


def metadata_info(station_path):
    """
    generate metadata dictionary from station_path folder name
    :param station_path:
    :return: dictionary of metadata
    """
    df = read_metadata_file()

    if 'CapeDuCouedic' in station_path:
        site_code = 'COUEDIC'
    elif 'CapeSorell' in station_path:
        site_code = 'SORELL'

    timezone = df.loc[site_code]['timezone']
    timezone = dt_parser.parse(timezone[:]).time()

    return {'site_name': df.loc[site_code]['site_name'],
            'site_code': site_code,
            'latitude': df.loc[site_code]['latitude'],
            'longitude': df.loc[site_code]['longitude'],
            'timezone': timezone.hour + timezone.minute/60.0,
            'title': "Waverider Buoy observations at {site_name}".format(site_name=df.loc[site_code]['site_name']),
            'instrument_maker': df.loc[site_code]['instrument_maker'],
            'instrument_model': df.loc[site_code]['instrument_model'],
            'waverider_type': df.loc[site_code]['waverider_type'],
            'water_depth': df.loc[site_code]['water_depth'],
            'water_depth_units': 'meters',
            'wmo_id': df.loc[site_code]['wmo_id']
            }


def parse_xls_xlsx_bom_wave(filepath):
    """
    parser for xls and xlsx bom wave files
    :param filepath:
    :return: dataframe of data
    """
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        workbook = xlrd.open_workbook(filepath)

        for row in range(20):  # looking how many rows to skip before start of data
            if "Time (UTC" in workbook.sheets()[0].cell(row, 0).value:
                skip_row = row
                break

        if skip_row == None:
            logger.error("{file} has no date column in the form of \"Date (UTC\"".format(file=filepath))
            exit(1)

        df = pd.read_excel(filepath,
                           skiprows=skip_row
                           )
        time_var_name = [s for s in df.columns.values if "Time" in s][0]  # find time variable name

        df.rename(columns={time_var_name: 'datetime'}, inplace=True)
        df.rename(columns={"Hs (m)": "Hs"}, inplace=True)
        df.rename(columns=lambda x: x.strip())  # strip leading trailing spaces from header

        """ by default, xls xlsx formatting of date string is a date understood by pandas. But in some files, the date
        is badly written and and some leading and trailing spaces in datetime variable (for example cpso2004.xlsx, cdec2007)
        Also some original files had a date written as  03/09/2003 05:00:001 <- ":001" !! so they were manually edited
        In this case we do the following
        """

        if isinstance(df['datetime'].values[0], unicode) or isinstance(df['datetime'].values[0], str):
            date_format = '%d/%m/%Y %H:%M:%S'
            df['datetime'] = pd.to_datetime(df['datetime'].map(lambda x: x.strip()), format=date_format)
            logger.warning('Date column in spreadsheet is not of type date; Converting from string using "{format}"'.
                           format(format=date_format))

        return df


def parse_csv_bom_wave(filepath):
    """
    parser for csv bom wave files
    :param filepath:
    :return: dataframe of data
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, header=None,
                         engine='python')

        if any(df[0] == 'Time (UTC+10)'):
            time_var_name = 'Time (UTC+10)'
        elif any(df[0] == 'Time (UTC+9.5)'):
            time_var_name = 'Time (UTC+9.5)'

        df2 = df.iloc[(df.loc[df[0] == time_var_name].index[0]):, :].reset_index(drop=True)  # skip metadata lines
        df2.columns = df2.loc[0]  # set column header as first row
        df2.drop(df2.index[0], inplace=True)  # remove first row which was the header
        df2.rename(columns={time_var_name: "datetime"}, inplace=True)
        df2.rename(columns=lambda x: x.strip())  # strip leading trailing spaces from header
        date_format = '%d/%m/%Y %H:%M'
        df2['datetime'] = pd.to_datetime(df2['datetime'], format=date_format)
        logger.warning('date format; {format}'.format(format=date_format))
        df2.rename(columns={"Hs (m)": "Hs"}, inplace=True)

        return df2


def parse_bom_wave(filepath):
    """
    call either parse_csv_bom_wave or parse_xls_xlsx_bom_wave depending of file extension
    :param filepath:
    :return: dataframe
    """
    if filepath.endswith('.csv'):
        return parse_csv_bom_wave(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return parse_xls_xlsx_bom_wave(filepath)


def gen_nc_bom_wave_dm_deployment(filepath, metadata, output_path):
    """
    generate a FV01 NetCDF file of current data.
    :param filepath_path: the path to a wave file to parse
    :param metadata: metadata output from metadata_info function
    :param output_path: NetCDF file output path
    :return: output file path
    """

    wave_df = parse_bom_wave(filepath)  # only one file

    # substract timezone to be in UTC
    wave_df['datetime'] = wave_df['datetime'].dt.tz_localize(None).astype('O').values - \
                          datetime.timedelta(hours=metadata['timezone'])

    var_mapping = param_mapping_parser(BOM_WAVE_PARAMETER_MAPPING)
    site_code = metadata['site_code']
    nc_file_name = 'BOM_W_{date_start}_{site_code}_WAVERIDER_FV01_END-{date_end}.nc'.format(
        date_start=wave_df.datetime.dt.strftime('%Y%m%dT%H%M%SZ').values.min(),
        site_code=site_code,
        date_end=wave_df.datetime.dt.strftime('%Y%m%dT%H%M%SZ').values.max()
    )

    temp_dir = tempfile.mkdtemp()
    nc_file_path = os.path.join(temp_dir, nc_file_name)

    try:
        with Dataset(nc_file_path, 'w', format='NETCDF4') as nc_file_obj:
            nc_file_obj.createDimension("TIME", wave_df.datetime.shape[0])
            nc_file_obj.createDimension("station_id_strlen", 30)

            nc_file_obj.createVariable("LATITUDE", "d", fill_value=99999.)
            nc_file_obj.createVariable("LONGITUDE", "d", fill_value=99999.)
            nc_file_obj.createVariable("STATION_ID", "S1", ("TIME", "station_id_strlen"))

            nc_file_obj["LATITUDE"][:] = metadata['latitude']
            nc_file_obj["LONGITUDE"][:] = metadata['longitude']
            nc_file_obj["STATION_ID"][:] = [stringtochar(np.array(metadata['site_name'], 'S30'))] * wave_df.shape[0]

            var_time = nc_file_obj.createVariable("TIME", "d", "TIME")

            # add gatts and variable attributes as stored in config files
            generate_netcdf_att(nc_file_obj, NC_ATT_CONFIG, conf_file_point_of_truth=True)

            time_val_dateobj = date2num(wave_df.datetime.dt.to_pydatetime(), var_time.units, var_time.calendar)

            var_time[:] = time_val_dateobj

            df_varname_ls = list(wave_df[wave_df.keys()].columns.values)
            df_varname_ls.remove("datetime")

            for df_varname in df_varname_ls:
                df_varname_mapped_equivalent = df_varname
                mapped_varname = var_mapping.loc[df_varname_mapped_equivalent]['VARNAME']

                dtype = wave_df[df_varname].values.dtype
                if dtype == np.dtype('int64'):
                    dtype = np.dtype('int16')  # short
                else:
                    dtype = np.dtype('f')

                nc_file_obj.createVariable(mapped_varname, dtype, "TIME")
                set_var_attr(nc_file_obj, var_mapping, mapped_varname, df_varname_mapped_equivalent, dtype)
                setattr(nc_file_obj[mapped_varname], 'coordinates', "TIME LATITUDE LONGITUDE")

                nc_file_obj[mapped_varname][:] = wave_df[df_varname].values

            set_glob_attr(nc_file_obj, wave_df, metadata)

        # we do this for pipeline v2
        os.chmod(nc_file_path, 0664)
        shutil.move(nc_file_path, output_path)

    except Exception as err:
        logger.error(err)

    shutil.rmtree(temp_dir)

    return os.path.join(output_path, os.path.basename(nc_file_path))