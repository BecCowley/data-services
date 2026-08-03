"""
Read the XBT line vocabulary from vocabs.ands.org.au

How to use:
    from xbt_line_vocab import *

    xbt_info = xbt_line_info() # simple dictionnary of all the xbt line labels
    the dict keys refer to the prefLabel
    the dict values : None if no code, otherwise the code value

author : Besnard, Laurent
"""

import ssl
import os
try:
    from platform_code_vocab import is_url_accessible, _fetch_xml_root
except ImportError:
    from lib.python.platform_code_vocab import is_url_accessible, _fetch_xml_root
import xml.etree.ElementTree as ET

def xbt_line_info():
    """
    retrieves a dictionnary of xbt line code with their IMOS code equivalent if available
    """
    xbt_line_vocab_url = 'http://content.aodn.org.au/Vocabularies/XBT-line/aodn_aodn-xbt-line-vocabulary.rdf'

    if is_url_accessible(xbt_line_vocab_url):
        # certificate handling
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        root, response               = _fetch_xml_root(xbt_line_vocab_url)
        url = True
    else:
        # look for the file in the same directory
        try:
            rdf_file_path = os.path.join(os.path.dirname(__file__), 'aodn_aodn-xbt-line-vocabulary.rdf')
            tree = ET.parse(rdf_file_path)
            root = tree.getroot()
            url = False
        except Exception as e:
            print(f"XBT line vocab url not accessible and failed to load local XBT line vocab file. {e}")
            return None

    xbt_dict = {}

    for item in root:
        if 'Description' in item.tag:
            xbt_line_code = None
            xbt_line_pref_label = None
            xbt_line_alt_label = ""

            for val in item:
                platform_element_sublabels = val.tag

                if platform_element_sublabels is not None:
                    if 'prefLabel' in platform_element_sublabels:
                        xbt_line_pref_label = val.text
                    if 'code' in platform_element_sublabels:
                        xbt_line_code = val.text
                    else:
                        xbt_line_code = xbt_line_pref_label
                    if 'altLabel' in platform_element_sublabels:
                        xbt_line_alt_label += val.text if xbt_line_alt_label == "" else ", " + val.text

            if xbt_line_pref_label:
                xbt_dict[xbt_line_pref_label] = (xbt_line_code, xbt_line_alt_label)

    if url:
        response.close()
    return xbt_dict
