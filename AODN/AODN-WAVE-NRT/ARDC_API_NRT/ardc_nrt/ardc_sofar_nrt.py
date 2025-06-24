#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SOFAR API Documentation available at https://docs.sofarocean.com/

Script to download wave data from SOFAR API.

Data is converted into IMOS compliant NetCDF files
author Laurent Besnard, laurent.besnard@utas.edu.au
"""

import datetime
import logging
import os

from ardc_nrt.lib.common.lookup import lookup
from ardc_nrt.lib.common.pickle_db import ardcPickle
from ardc_nrt.lib.common.processing import process_wave_dataframe, get_timestamp_start_end_to_download
from ardc_nrt.lib.common.utils import IMOSLogging, args
from ardc_nrt.lib.sofar import config
from ardc_nrt.lib.sofar.api import sofarApi
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY


def process_wave_source_id(source_id, incoming_path=None):
    """
    Core function which process all new data available for a source_id

        Parameters:
            source_id (string): spotter_id value
            incoming_path (string): AODN pipeline incoming path

        Returns:
    """
    sources_id_metadata = ardc_lookup.get_sources_id_metadata()
    site_name = sources_id_metadata[source_id]['site_name']
    LOGGER.info(f'processing source_id: {source_id}')
    LOGGER.info(f'site_name: {site_name}')

    api_sofar = sofarApi()
    latest_timestamp_available_source_id = api_sofar.get_source_id_latest_timestamp(source_id)

    ardc_pickle = ardcPickle(OUTPUT_PATH)
    latest_timestamp_processed_source_id = ardc_pickle.get_latest_processed_date(source_id)

    timestamp_start_end = get_timestamp_start_end_to_download(config.conf_dirpath, source_id,
                                                              latest_timestamp_available_source_id,
                                                              latest_timestamp_processed_source_id)
    if not timestamp_start_end:  # already up to date
        return

    timestamp_start, timestamp_end = timestamp_start_end

    # api call to download one month at a time
    start_date = timestamp_start.replace(tzinfo=datetime.timezone.utc)
    end_date = timestamp_end.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
    months_to_download = [dt for dt in rrule(MONTHLY, dtstart=start_date, until=end_date + relativedelta(months=1))][0:-1]

    for month in months_to_download:
        data = api_sofar.get_source_id_wave_data_time_range(source_id, month, month + relativedelta(months=1))

        if data is None:

            LOGGER.error(f"Processing {source_id} aborted. No data available BETWEEN {month} AND {month + relativedelta(months=1)}")
            return

        if data is not None:
            data = data.dropna(subset = ['latitude', 'longitude'], how = 'all') # Remove rows associated with NAs for lat/long columns
            template_dirpath = config.conf_dirpath
            process_wave_dataframe(data, source_id, template_dirpath, OUTPUT_PATH, incoming_path)


if __name__ == "__main__":

    vargs = args()

    # set up logging
    IMOSLogging().logging_start(os.path.join(vargs.output_path, 'process.log'))
    global LOGGER
    LOGGER = logging.getLogger(__name__)

    # set up output path of the NetCDF files and logging
    global OUTPUT_PATH
    OUTPUT_PATH = vargs.output_path

    # set up saved pickle file to store information of previous runs of the
    # script
    global PICKLE_FILE
    PICKLE_FILE = ardcPickle(OUTPUT_PATH).pickle_file_path()

    api_config = config.conf_dirpath
    ardc_lookup = lookup(api_config)
    sources_id_metadata = ardc_lookup.get_sources_id_metadata()

    for source_id in sources_id_metadata.keys():
        process_wave_source_id(source_id, incoming_path=vargs.incoming_path)


