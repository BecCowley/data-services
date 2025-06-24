import datetime
import json
import logging
import os
import tempfile
import re

import pandas
import numpy as np
from aodntools.ncwriter import ImosTemplate
from jsonmerge import merge
from netCDF4 import date2num
from netCDF4 import num2date, Dataset
from ardc_nrt.lib.bom import config as bomconfig
from . import config
from .lookup import lookup


LOGGER = logging.getLogger(__name__)

SOURCES_METADATA_FILENAME = config.sources_metadata_filename

class wave(object):
    def __init__(self, api_config_path, source_id, df, output_dir):
        self.logging_filepath = logging.getLogger(__name__)
        self.logger = logging.getLogger(__name__)

        self.source_id = source_id
        self.api_config_path = api_config_path

        self.ardc_lookup = lookup(self.api_config_path)
        self.institution_template_path = self.ardc_lookup.get_institution_netcdf_template(self.source_id)
        self.sources_id_metadata_template_path = self.ardc_lookup.sources_id_metadata_template_path

        self.merge_source_id_with_institution_template()

        self.df = df
        self.output_dir = output_dir

    def convert_wave_data_to_netcdf(self, true_dates=False):
        """
        convert a pandas dataframe into an IMOS compliant NetCDF files

            Parameters:
                true_dates (boolean): default (False) -> NetCDF filename date is monthly ..._{date_start}_monthly_FV00_END
                                                 True -> filename is ..._{date_start}_FV00_END-{date_end}.nc

            Returns:
                path (string): NetCDF file path
        """
        template = ImosTemplate.from_json(self.template_merged_json_path)


        # create list of variable present in dataset to determine if any is missing
        variable_list_present = list()
        for df_variable_name in self.df.columns.values:
            aodn_variable_name = self.ardc_lookup.get_matching_aodn_variable(df_variable_name)
            if aodn_variable_name:
                variable_list_present.append(aodn_variable_name)

            if aodn_variable_name == "TIME":
                time_val_dateobj = date2num(self.df.timestamp,
                                            template.variables['TIME']['units'],
                                            template.variables['TIME']['calendar'])

                template.variables['TIME']['_data'] = time_val_dateobj

            elif aodn_variable_name is not None:
                template.variables[aodn_variable_name]['_data'] = self.df[df_variable_name].values

        # log info if some variables have no matching aodn variable
        if len(self.df.columns) != len(variable_list_present):
            self.logger.warning(
                'Variable(s) not matched up with AODN variable. '
                'NetCDF file will be created without this variable(s)')

        data_shape = list(self.df.shape)
        nvar = data_shape[1]

        # generate quality control data: QC value set to 2 - not evaluated
        filldata = np.full(data_shape[0], 2)
        self.df.insert(nvar, 'wave_qc', filldata.astype(np.int8))
        template.variables['WAVE_quality_control']['_data'] = self.df['wave_qc'].values

        template.add_extent_attributes(time_var='TIME', vert_var=None, lat_var='LATITUDE', lon_var='LONGITUDE')
        template.add_date_created_attribute()
        template.global_attributes.update({
            'history': "this file was file created on: {date_created}".format(
                date_created=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
        })

        if template.global_attributes['institution_code'].lower() == 'bom':
            template.global_attributes.update({
                'institution_code': bomconfig.mapped_source_id[self.source_id]
            })
        elif template.global_attributes['institution_code'].lower() == 'omc':
            template.global_attributes.update({'institution_code': 'PPA'})
        elif template.global_attributes['institution_code'].lower() == 'vic' and \
                 not re.match('^IMOS', template.global_attributes['institution'].upper()):
            template.global_attributes.update({'institution_code': 'VIC-DEAKIN-UNI'})
        elif template.global_attributes['institution_code'].lower() == 'uwa-smart':
            template.global_attributes.update({'institution_code': 'UWA'})
        elif re.match('^IMOS/', template.global_attributes['institution'].upper()) and \
                not template.global_attributes['institution_code'].lower() == 'imos':
            # rename VIC and UWA operated buoys from IMOS-NTP facility
            template.global_attributes.update({'institution_code': 'IMOS_NTP-WAVE'})
        elif re.match('IMOS COASTAL WAVE BUOYS', template.global_attributes['institution'].upper()) and \
                not template.global_attributes['institution_code'].lower() == 'imos':
            # case when VIC and UWA operated buoys used for IMOS deployments
            template.global_attributes.update({'institution_code': 'IMOS_COASTAL-WAVE-BUOYS'})
        elif template.global_attributes['institution_code'].lower() == 'imos':
            template.global_attributes.update({'institution_code': 'IMOS_COASTAL-WAVE-BUOYS'})

        month_start = datetime.datetime(self.df.timestamp.min().year, self.df.timestamp.min().month, 1, 0, 0, 0)

        output_nc_filename = '{institution_code}_{date_start}_{site_name}_RT_WAVE-PARAMETERS_monthly.nc'.format(
            institution_code=template.global_attributes['institution_code'].upper(),
            site_name=template.global_attributes['site_name'].upper().replace(" ","-"),
            date_start=datetime.datetime.strftime(month_start, '%Y%m%d')
        )

        if true_dates:
            output_nc_filename = '{institution_code}_{date_start}_{site_name}_RT_WAVE-PARAMETERS_END-{date_end}.nc'.\
                format(
                institution_code=template.global_attributes['institution_code'].upper(),
                site_name=template.global_attributes['site_name'].upper().replace(" ","-"),
                date_start=datetime.datetime.strftime(self.df.timestamp.min(), '%Y%m%dT%H%M%SZ'),
                date_end=datetime.datetime.strftime(self.df.timestamp.max(), '%Y%m%dT%H%M%SZ'),
            )

        # Remove unnecessary attributes
        #deployment_start_date only present if spotter is deployed currently
        if 'deployment_start_date' in template.global_attributes.keys():
            template.global_attributes.pop('deployment_start_date')

        template.global_attributes.pop('institution_code')
        template.global_attributes.pop('geospatial_vertical_min')
        template.global_attributes.pop('geospatial_vertical_max')
        template.global_attributes.pop('latitude_nominal')
        template.global_attributes.pop('longitude_nominal')
        # add data for Timeseries variable
        template.variables['timeSeries']['_data'] = np.int16([1])
        template.variables['WAVE_quality_control']['valid_min'] = np.int8(
                            template.variables['WAVE_quality_control']['valid_min'])
        template.variables['WAVE_quality_control']['valid_max'] = np.int8(
                            template.variables['WAVE_quality_control']['valid_max'])
        template.variables['WAVE_quality_control']['flag_values'] = np.int8(
                            template.variables['WAVE_quality_control']['flag_values'])

        netcdf_path = os.path.join(self.output_dir, output_nc_filename)
        template.to_netcdf(netcdf_path)

        return netcdf_path

    def merge_source_id_with_institution_template(self):
        """
        Merging source_id specific NetCDF template with its affiliated institution generic NetCDF template.
        the source_id template will overwrite key values if similar keys are found in both templates

            Parameters:

            Returns:
                (string): path of merged json file
        """
        with open(self.institution_template_path) as f:
            institution_template_json_data = json.load(f)

        with open(self.sources_id_metadata_template_path) as f:
            sources_template_json_data = json.load(f)

        if type(self.source_id) == int:
            source_id_template_json_data = sources_template_json_data[str(self.source_id)]
        else:
            source_id_template_json_data = sources_template_json_data[self.source_id]

        self.template_merged_json_path = merge_json(institution_template_json_data, source_id_template_json_data)
        return self.template_merged_json_path


def merge_json(json_primary, json_secondary):
    """
    Merge json_secondary json file onto json_primary json file and overwritte json_primary keys simlar key found in
    json_secondary

        Parameters:
            json_primary (string): path of json file
            json_secondary (string): path of json file

        Returns:
            json_tmp_file (string): path of temporary json file

    """
    merge_json = merge(json_primary, json_secondary)
    json_tmp_file = tempfile.mktemp()
    with open(json_tmp_file, "w") as file:
        json.dump(merge_json, file, indent=2, sort_keys=True)

    return json_tmp_file


def nc_get_max_timestamp(nc_path):
    """
    Returns the max value of the TIME variable in a NetCDF file

        Parameters:
            nc_path (string): NetCDF absolute path

        Returns:
            Pandas timestamp: Max timestamp (date at UTC) of TIME variable
    """
    with Dataset(nc_path) as nc_obj:
        cf_time_obj = num2date(nc_obj["TIME"][:].max(), nc_obj["TIME"].units, nc_obj["TIME"].calendar)

        val = pandas.Timestamp(year=cf_time_obj.year,
                               month=cf_time_obj.month,
                               day=cf_time_obj.day,
                               hour=cf_time_obj.hour,
                               minute=cf_time_obj.minute,
                               second=cf_time_obj.second,
                               tz='UTC')

        return val



