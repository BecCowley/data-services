"""
Read the XBT line vocabulary from vocabs.ands.org.au

How to use:
    from xbt_line_vocab import *

    xbt_info = xbt_line_info() # simple dictionnary of all the xbt line labels
    the dict keys refer to the prefLabel
    the dict values : None if no code, otherwise the code value

author : Besnard, Laurent
"""

import urllib.request, urllib.error, urllib.parse
import xml.etree.ElementTree as ET
import ssl


def xbt_line_info():
    """
    retrieves a dictionnary of xbt line code with their IMOS code equivalent if available
    """
    xbt_line_vocab_url = 'http://content.aodn.org.au/Vocabularies/XBT-line/aodn_aodn-xbt-line-vocabulary.rdf'

    try:
        # certificate handling
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        response               = urllib.request.urlopen(xbt_line_vocab_url, context=ctx)
        html                   = response.read()
        root                   = ET.fromstring(html)
    except Exception:
        print(('xbt line vocab url \"%{xbt_url}\" is not accessible. contact info@aodn.org.au'.format(xbt_url=xbt_line_vocab_url)))

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

    response.close()
    return xbt_dict
