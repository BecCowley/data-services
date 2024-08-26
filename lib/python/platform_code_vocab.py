"""
Find the platform_code platform_name equivalence from poolparty platform vocab
xml file stored in content.aodn.org.au

How to use:
    from platform_code_vocab import *

    platform_altlabels_per_preflabel()
    platform_type_uris_by_category()
    platform_altlabels_per_preflabel('Fixed station')
    platform_altlabels_per_preflabel('Mooring and buoy')
    platform_altlabels_per_preflabel('Vessel')

author : Besnard, Laurent
"""

import warnings
import xml.etree.ElementTree as ET

from six.moves.urllib.request import urlopen


def platform_type_uris_by_category():
    """
    retrieves a list of platform category and their narrowMatch url type which
    defines their category
    """
    platform_cat_vocab_url = 'http://content.aodn.org.au/Vocabularies/platform-category/aodn_aodn-platform-category-vocabulary.rdf'
    response               = urlopen(platform_cat_vocab_url)
    html                   = response.read()
    root                   = ET.fromstring(html)
    platform_cat_list      = {}

    for item in root:
        if 'Description' in item.tag:
            platform_cat_url_list = []
            platform_cat          = None

            for val in item:
                platform_element_sublabels = val.tag

                # handle more than 1 url match per category of platform
                if platform_element_sublabels is not None:
                    if 'narrowMatch' in platform_element_sublabels:
                        val_cat_url = [item for item in val.attrib.values()][0]
                        platform_cat_url_list.append(val_cat_url)

                    elif 'prefLabel' in platform_element_sublabels:
                        platform_cat = val.text

            if platform_cat is not None and platform_cat_url_list:
                platform_cat_list[platform_cat] = platform_cat_url_list

    response.close()
    return platform_cat_list


def platform_altlabels_per_preflabel(category_name=None):
    """
    retrieves a list of platform code - platform name dictionnary.
    The function can either retrieves ALL platform codes, or only platform codes
    for a specific category, which can be found in platform_type_uris_by_category()

    Example:
        platform_altlabels_per_preflabel()
        platform_altlabels_per_preflabel('Vessel')
    """

    platform_vocab_url = 'http://content.aodn.org.au/Vocabularies/platform/aodn_aodn-platform-vocabulary.rdf'
    response           = urlopen(platform_vocab_url)
    html               = response.read()
    root               = ET.fromstring(html)
    platform           = {}
    filter_cat_type    = False

    if category_name:
        # a platform category is defined by a list of urls.
        # first we check the category exist in the category list, secondly we
        # get the list of url vocab attached to this category
        filter_cat_name = category_name
        filter_cat_list = platform_type_uris_by_category()
        if filter_cat_name in filter_cat_list.keys():
            filter_cat_url_list = filter_cat_list[filter_cat_name]
            filter_cat_type     = True
        else:
            warnings.warn("Platform category %s not in platform category vocabulary" % filter_cat_name)

    for item in root:
        # main element name we're interested in
        if 'Description' in item.tag:
            # for every element, iterate over the sub elements and look for
            # common platform label
            platform_code    = []
            platform_name    = None
            platform_imo     = None
            platform_url_cat = None

            for val in item:
                platform_element_sublabels = val.tag

                # handle more than 1 alternative label for same pref label
                if platform_element_sublabels is not None:
                    if 'altLabel' in platform_element_sublabels:
                        platform_code.append(val.text)

                    elif 'prefLabel' in platform_element_sublabels:
                        platform_name = val.text

                    elif 'hiddenLabel' in platform_element_sublabels:
                        platform_imo = val.text

                    elif 'broader' in platform_element_sublabels:
                        val_cat_url = [item for item in val.attrib.values()][0]
                        platform_url_cat = val_cat_url

            # use the optional argument
            if filter_cat_type:
                if platform_url_cat in filter_cat_url_list:
                    if platform_name is not None and platform_code:
                        for platform_code_item in platform_code:
                            platform[platform_code_item] = platform_name, platform_imo

            else:
                if platform_name is not None and platform_code:
                    for platform_code_item in platform_code:
                        platform[platform_code_item] = platform_name, platform_imo

    response.close()
    return platform
