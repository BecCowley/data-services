import re
import xml.etree.ElementTree as ET
from functools import lru_cache
import logging


import numpy as np
import pandas as pd
from owslib.wfs import WebFeatureService
from tenacity import *

from . import config


class bomWFS(object):
    def __init__(self):
        self.url_prefix = config.url_prefix
        self.typename = config.typename
        self.logger = logging.getLogger(__name__)


    @lru_cache(maxsize=None)
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def wfs_query(self):
        wfs = WebFeatureService(url=self.url_prefix, version='1.1.0', timeout=30)
        # for some reason the getfeature function defaults to http:// instead of https://, which bom doesn't support anymore
        get_feature_op = wfs.getOperationByName('GetFeature')
        for method in get_feature_op.methods:
            if method['type'].lower() == 'get':
                method['url'] = method['url'].replace('http://', 'https://')

        response = wfs.getfeature(typename=self.typename)
        res = response.read()

        return res

    @staticmethod
    def xml2dataframe(xml_data):
        root = ET.XML(xml_data)

        dict = {}
        for i in range(1, len(root)):
            n_elements = len(root[i][0])

            for j in range(2, n_elements):
                varname = re.sub('^.*}', '', root[i][0][j].tag)
                varval = root[i][0][j].text

                if varname == 'datetime':
                    varval = pd.Timestamp(varval)

                elif varname == 'statid':
                    varval = int(varval)

                else:
                    if varval is None:
                        varval = np.nan
                    else:
                        varval = float(varval)

                if varname in dict.keys():
                    dict[varname].append(varval)
                else:
                    dict[varname] = [varval]

        return pd.DataFrame(dict)

    @staticmethod
    def cleansing(data):
        data.rename(columns={'statid': 'source_id'}, inplace=True)
        data.rename(columns={'datetime': 'timestamp'}, inplace=True)

        return data

    def get_sources_id_data(self):
        res = self.wfs_query()
        df = self.xml2dataframe(res)
        df = self.cleansing(df)

        return df

    def get_sources_id_metadata(self):
        data = self.get_sources_id_data()
        return data.drop_duplicates(subset=['source_id'])[['source_id', 'lat', 'lon']]

    def get_source_id_metadata(self, source_id):
        data = self.get_sources_id_metadata()
        if data.empty:
            self.logger.error(f"{source_id}: No metadata available. Please check json template/contact BOM")
            return pd.DataFrame()
        return data.loc[data['source_id'] == source_id]

    def get_source_id_data(self, source_id):
        data = self.get_sources_id_data()

        data = data.loc[data['source_id'] == source_id]
        if data.empty:
            self.logger.error(f"{source_id}: No data available. Please check json template/contact BOM")
            return pd.DataFrame()

        # BOM WFS outputs data not always ordered with timestamp
        data = data.sort_values("timestamp")  # sort based on timestamp
        if not (data['timestamp'].is_monotonic_increasing and data['timestamp'].is_unique):
            duplicated_values = data[data['timestamp'].duplicated(keep=False)]
            self.logger.error(
                f"{source_id}: Duplicate values of timestamp \r\n {duplicated_values} \r\n Data not processed. Contact BOM"
            )
            return pd.DataFrame()

        data.sort_values(by=['timestamp'], inplace=True)
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)

        return data
