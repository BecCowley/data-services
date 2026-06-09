"""
Reads the ship_callsign from platform vocab and returns a dictionnary of
callsign and vessel names.
Used for SOOP
How to use:
    from ship_callsign import ship_callsign_list

    ship_callsign_list()
    ship_callsign('VRDU8')

author : Besnard, Laurent
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from lib.python.platform_code_vocab import platform_altlabels_per_preflabel

from functools import lru_cache


@lru_cache(maxsize=32)
def ship_callsign_list():
    """
    renaming of platform_code_vocab function and create exceptions for
    Astrolabe vessel, and others if required
    Vessel names have a '-' instead of blank space for the folder structure
    """
    platform_codes = platform_altlabels_per_preflabel('Vessel')
    for key, value in platform_codes.items():
        # Convert the tuple to a list
        value_list = list(value)
        # Replace the string in the first element
        value_list[0] = value_list[0].replace(' ', '-')
        # remove any ' from the vessel name, targeting the L'Astrolabe vessel
        value_list[0] = value_list[0].replace("'", "")
        # Convert the list back to a tuple and update the dictionary
        platform_codes[key] = tuple(value_list)

    """ this section deals with vessels which have a different callsign, but a similar name. This is the case for new 
    vessels replacing their older 'version'. In the vocabulary, in order to deal with this special case, the vessel name
    is written as 'Vessel-Name-{callsign}'
    Example: "Highland-Chief-{VROJ8}"
    Also remove the "IMO:" string in the IMO field
    """
    for callsign in platform_codes:
        str_to_rm = '-{{{callsign}}}'.format(callsign=callsign)
        if str_to_rm in platform_codes[callsign][0]:
            value_list = list(platform_codes[callsign])
            value_list[0] = value_list[0].replace(str_to_rm, '')
            platform_codes[callsign] = tuple(value_list)
        # remove IMO:
        imo_field = platform_codes[callsign][1]
        if imo_field is not None:
            if ':' in imo_field:
                cleaned_imo = imo_field.split(':', 1)[1].strip()
            else:
                cleaned_imo = imo_field
            if cleaned_imo != imo_field:
                value_list = list(platform_codes[callsign])
                value_list[1] = cleaned_imo
                platform_codes[callsign] = tuple(value_list)
    return platform_codes


def ship_callsign(callsign):
    """
    returns the vessel name of a specific callsign
    returns none if the vessel name does not exist
    """
    callsigns = ship_callsign_list()
    if callsign in callsigns.keys():
        return callsigns[callsign]
    else:
        return None
