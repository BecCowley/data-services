#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
TODO
-improve reporting function
"""

import argparse
import csv
import glob
import operator
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta
from functools import partial
from itertools import groupby
try:
    from StringIO import StringIO  # for Python 2
except ImportError:
    from io import StringIO  # for Python 3

from multiprocessing import Pool, cpu_count

from geopy.distance import vincenty
from netCDF4 import Dataset, num2date
from osgeo import gdal, osr
from six.moves.urllib.request import urlopen
from wand.image import Image

from imos_logging import IMOSLogging

AUV_WIP_DIR = os.path.join(os.environ.get('WIP_DIR'), 'AUV', 'AUV_VIEWER_PROCESSING')
IS_NETCDF_EXISTS = True # see global variable explanation in _netcdf_dive_path function

def list_geotiff_dive(dive_path):
    """
    list the geotiffs images of one dive. Only looking for one geotiff for each pair
    """
    geotiff_dir_dive_path = _geotiff_dive_path(dive_path)
    geotiff_list          = []

    # different pattern to arbitrary look for first left images, otherwise fore images
    pattern_lc = re.compile("^PR_([0-9]{8})_([0-9]{6})_([0-9]{3})_LC16.tif$")  # left right images
    pattern_fc = re.compile("^PR_([0-9]{8})_([0-9]{6})_([0-9]{3})_FC16.tif$")  # fore images
    pattern_ac = re.compile("^PR_([0-9]{8})_([0-9]{6})_([0-9]{3})_AC16.tif$")  # aft images
    pattern_rm = re.compile("^PR_([0-9]{8})_([0-9]{6})_([0-9]{3})_RM16.tif$")  # rear(? only one dive under WA201403 has these...) images

    for file in os.listdir(geotiff_dir_dive_path):
        if pattern_lc.match(file) is not None:
            geotiff_list.append(os.path.join(geotiff_dir_dive_path, file))
        elif pattern_fc.match(file) is not None:
            geotiff_list.append(os.path.join(geotiff_dir_dive_path, file))
        elif pattern_ac.match(file) is not None:
            geotiff_list.append(os.path.join(geotiff_dir_dive_path, file))
        elif pattern_rm.match(file) is not None:
            geotiff_list.append(os.path.join(geotiff_dir_dive_path, file))

    geotiff_list.sort()

    return geotiff_list

def is_geotiff_valid(geotiff_path):
    """
    Check file geotiff validity
    Return boolean
    """
    try:
        # gdal cant be opened with with statement. see https://stackoverflow.com/questions/40513144/open-geotiff-with-gdal-produces-attributeerror-exit
        ds = gdal.Open(geotiff_path)
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjectionRef())
        del ds

        return True

    except Exception as err:
        logger.error('geotiff not valid {path}\n{err}'.format(path=geotiff_path,
                                                              err=err))
        return False


def geotiff_corner_coordinates(geotiff_path):
    """
    retrieves the latlon coordinate corners of a geotiff image file
    """
    ds     = gdal.Open(geotiff_path)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    gt        = ds.GetGeoTransform()

    lower_lft_xy = gdal.ApplyGeoTransform(gt, 0, ds.RasterYSize)
    lower_rgt_xy = gdal.ApplyGeoTransform(gt, ds.RasterXSize, ds.RasterYSize)
    upper_lft_xy = gdal.ApplyGeoTransform(gt, 0, 0)
    upper_rgt_xy = gdal.ApplyGeoTransform(gt, ds.RasterXSize, 0)
    center_xy    = gdal.ApplyGeoTransform(gt, ds.RasterXSize / 2, ds.RasterYSize / 2)

    lower_lft_latlon = transform.TransformPoint(lower_lft_xy[0], lower_lft_xy[1])
    lower_rgt_latlon = transform.TransformPoint(lower_rgt_xy[0], lower_rgt_xy[1])
    upper_lft_latlon = transform.TransformPoint(upper_lft_xy[0], upper_lft_xy[1])
    upper_rgt_latlon = transform.TransformPoint(upper_rgt_xy[0], upper_rgt_xy[1])
    center_latlon    = transform.TransformPoint(center_xy[0], center_xy[1])

    return lower_lft_latlon, lower_rgt_latlon, upper_lft_latlon, upper_rgt_latlon, center_latlon


def distance_btw_latlon(lft_latlon, rgt_latlon):
    """
    returns the distance in meters between 2 lat lon coordinates
    """
    return vincenty(lft_latlon, rgt_latlon).meters


def geotiff_list_metadata(geotiff_list):
    """
    generate a list of gdal metadata for all geotiff found for one dive
    """
    header = "campaign_code,dive_code,image_filename,longitude,latitude,"\
        "image_width,depth_sensor,altitude_sensor,depth,"\
        "sea_water_temperature,sea_water_salinity,"\
        "chlorophyll_concentration_in_sea_water,backscattering_ratio,"\
        "colored_dissolved_organic_matter,time,cluster_tag,up_left_lon,"\
        "up_left_lat,up_right_lon,up_right_lat,low_right_lon,low_right_lat,"\
        "low_left_lon,low_left_lat".split(",")

    geotiff_list_metadata = []
    for row, geotiff in enumerate(geotiff_list):

        if is_geotiff_valid(geotiff):
            geotiff_list_metadata.append({k: '' for k in header})  # initialise dict for every row
            geotiff_coordinate = geotiff_corner_coordinates(geotiff)
            geotiff_width      = distance_btw_latlon(geotiff_coordinate[0][0:2],
                                                     geotiff_coordinate[1][0:2])

            row_idx = len(geotiff_list_metadata) - 1  # this is incremented at each run. value can be different from row if geotiff is not valid

            geotiff_list_metadata[row_idx]["image_filename"] = os.path.splitext(os.path.basename(geotiff))[0]
            geotiff_list_metadata[row_idx]["up_left_lon"]    = geotiff_coordinate[2][0]
            geotiff_list_metadata[row_idx]["up_left_lat"]    = geotiff_coordinate[2][1]
            geotiff_list_metadata[row_idx]["up_right_lon"]   = geotiff_coordinate[3][0]
            geotiff_list_metadata[row_idx]["up_right_lat"]   = geotiff_coordinate[3][1]
            geotiff_list_metadata[row_idx]["low_left_lon"]   = geotiff_coordinate[0][0]
            geotiff_list_metadata[row_idx]["low_left_lat"]   = geotiff_coordinate[0][1]
            geotiff_list_metadata[row_idx]["low_right_lon"]  = geotiff_coordinate[1][0]
            geotiff_list_metadata[row_idx]["low_right_lat"]  = geotiff_coordinate[1][1]
            geotiff_list_metadata[row_idx]["image_width"]    = geotiff_width

    return geotiff_list_metadata


def _csv_track_dive_path(dive_path):
    """
    retrieves the csv track file path for one dive
    """
    for file in os.listdir(dive_path):
        if file.endswith("track_files"):
            track_folder_path = os.path.join(dive_path, file)
            break

    for file in os.listdir(track_folder_path):
        if file.endswith("_latlong.csv"):
            return os.path.join(track_folder_path, file)

    # if file not found
    logger.error('CSV Track file not found - Process of campaign aborted')
    raise Exception('CSV Track file not found')


def read_track_csv(dive_path):
    """
    import the data found in the csv track file for one dive. This version handles
    the cluster tag info. Some improvements could be done to generate automatically
    the converters variable
    """
    csv_path = _csv_track_dive_path(dive_path)
    f        = open(csv_path, 'rt')
    reader   = csv.reader(f)
    headers  = next(reader)

    while headers[0] != 'year':
        headers = next(reader)
        while headers == []:
            headers = next(reader)

    column = {}
    for h in headers:
        column[h] = []

    converters = [int] * 5 + [float] * 10 + [str] * 2 + [int]
    for row in reader:
        for h, v, conv in zip(headers, row, converters):
            column[h].append(conv(v))

    f.close()

    # remove png extension
    column['leftimage'] = [os.path.splitext(w)[0] for w in column['leftimage']]
    column['rightimage'] = [os.path.splitext(w)[0] for w in column['rightimage']]

    return column

# def _lookup_item_dict(item_name, item_value, dict):
    # return (item for item in dict if item[item_name] == item_value).next()

# def lookup_geotiff_info(geotiff_name, geotiff_list_metadata):
    # """
    # ex; lookup_geotiff_info('PR_20140329_044228_859_LC16.tif', dict)
    # """
    # return _lookup_item_dict("image_filename", geotiff_name, geotiff_list_metadata )


def _generate_geotiff_thumbnail(thumbnail_dir_path, geotiff_path):
    """
    generate the thumbnail of one image
    """
    thumbnail_path = os.path.join(thumbnail_dir_path, '%s.jpg' %
                                  os.path.splitext(os.path.basename(geotiff_path))[0])

    # replace last occurence of 'thumbnails' with 'full_res'
    full_res_path = 'full_res'.join(thumbnail_path.rsplit('thumbnails', 1))
    try:
        with Image(filename=geotiff_path) as img:
            img.save(filename=full_res_path)
            img.resize(453, 341)
            img.save(filename=thumbnail_path)
    except:
        pass


def generate_geotiff_thumbnails_dive(geotiff_dive_list, thumbnail_dir_path):
    """
    generate the thumbnails of geotiffs used by the auv viewer. This is done in
    a multithreading way by looking at the number of cores available on the machine
    files go to wip_dir
    """
    if not os.path.exists(thumbnail_dir_path):
        os.makedirs(thumbnail_dir_path)

    full_res_path = 'full_res'.join(thumbnail_dir_path.rsplit('thumbnails', 1))
    if not os.path.exists(full_res_path):
        os.makedirs(full_res_path)

    partial_job = partial(_generate_geotiff_thumbnail, thumbnail_dir_path)
    n_cores     = cpu_count()
    if n_cores > 1:
        pool = Pool(2)  # only use 2 cores to let other processes run smoothly
    else:
        pool = Pool(1)

    pool.map(partial_job, geotiff_dive_list)
    pool.close()
    pool.join()


def _geotiff_dive_path(dive_path):
    """
    retrieve the folder path containing geotiffs imagery of one dive
    """
    for file in os.listdir(dive_path):
        if file.endswith("_gtif"):
            return os.path.join(dive_path, file)

    logger.error('GEOTIFF folder not found - Process aborted')
    raise Exception('GEOTIFF folder not found')


def _netcdf_dive_path(dive_path):
    """
    get the path of the dir containing netcdf files for one dive
    """
    for file in os.listdir(dive_path):
        if file.endswith("hydro_netcdf"):
            return os.path.join(dive_path, file)

    logger.warning('NetCDF folder not found - Waiting for User input')
    # some dives dont have any NetCDF. If this is the case, it is better to contact the facility to know
    # if this is an issue or not. The user who uses this script will have to reply to this code with a
    # yes or no. If
    global IS_NETCDF_EXISTS
    IS_NETCDF_EXISTS = False

    if user_yes_no('Keep processing of dive without NetCDF'):
        logger.info('Process continues')
        return ''
    else:
        logger.error('Process aborted')
        raise Exception('NetCDF folder not found')


def user_yes_no(question):
    """
    Prompt user with a basic yes no question
    :param question: string
    :return: boolean
    """
    answer = input(question + "(y/n): ").lower().strip()
    print("")
    while not(answer == "y" or answer == "yes" or \
    answer == "n" or answer == "no"):
        print("Input yes or no")
        answer = input(question + "(y/n):").lower().strip()
        print("")
    if answer[0] == "y":
        return True
    else:
        return False


def read_netcdf_st(netcdf_path):
    """
    retrieve data from ST netcdf
    """
    try:
        netcdf_file_obj = Dataset(netcdf_path, mode='r')
        variables       = netcdf_file_obj.variables.keys()
        time            = netcdf_file_obj.variables['TIME']
        time            = num2date(time[:], time.units)
    except Exception as err:
        logger.warning('No ST data in NetCDF. Check with facility this is correct. err:{err}'.format(err=err))
        return []

    psal  = []
    temp  = []
    depth = []

    if 'PSAL' in variables:
        psal = netcdf_file_obj.variables['PSAL'][:]
    if 'TEMP' in variables:
        temp = netcdf_file_obj.variables['TEMP'][:]
    if 'DEPTH' in variables:
        depth = netcdf_file_obj.variables['DEPTH'][:]

    latitude  = netcdf_file_obj.variables['LATITUDE'][:]
    longitude = netcdf_file_obj.variables['LONGITUDE'][:]
    netcdf_file_obj.close()

    data_st = {'PSAL': psal,
               'TEMP': temp,
               'DEPTH': depth,
               'TIME': time,
               'LAT': latitude,
               'LON': longitude}

    return data_st


def read_netcdf_b(netcdf_path):
    """
    retrieve data from B netcdf
    """
    try:
        netcdf_file_obj = Dataset(netcdf_path, mode='r')
        variables       = netcdf_file_obj.variables.keys()
        time            = netcdf_file_obj.variables['TIME']
        time            = num2date(time[:], time.units)
    except Exception:
        logger.warning('No B data in NetCDF. Check with facility this is correct')
        return []

    cdom  = []
    cphl  = []
    opbs  = []
    depth = []

    if 'CDOM' in variables:
        cdom = netcdf_file_obj.variables['CDOM'][:]
    if 'CPHL' in variables:
        cphl = netcdf_file_obj.variables['CPHL'][:]
    if 'OPBS' in variables:
        opbs = netcdf_file_obj.variables['OPBS'][:]
    if 'DEPTH' in variables:
        depth = netcdf_file_obj.variables['DEPTH'][:]

    latitude  = netcdf_file_obj.variables['LATITUDE'][:]
    longitude = netcdf_file_obj.variables['LONGITUDE'][:]
    netcdf_file_obj.close()

    data_b = {'CDOM': cdom,
              'CPHL': cphl,
              'OPBS': opbs,
              'DEPTH': depth,
              'TIME': time,
              'LAT': latitude,
              'LON': longitude}

    return data_b


def read_netcdf(dive_path):
    """
    retrieve data from both hydro netcdf files of one dive only
    """
    netcdf_dir_dive_path = _netcdf_dive_path(dive_path)
    data_st              = []
    data_b               = []

    if not netcdf_dir_dive_path == '':
        for file in os.listdir(netcdf_dir_dive_path):
            nc_file = os.path.join(netcdf_dir_dive_path, file)
            if 'IMOS_AUV_ST_' in nc_file:
                data_st = read_netcdf_st(nc_file)
            elif 'IMOS_AUV_B_' in nc_file:
                data_b = read_netcdf_b(nc_file)

    return data_st, data_b


def match_csv_track_info_with_geotiff(csv_track_data, geotiff_metadata, campaign_name, dive_name):
    """
    match information found in the track_info csv file with the geotiff images
    found in the geotiff directory of one dive
    """
    for row, rest in enumerate(geotiff_metadata):
        try:
            if 'AC16' in geotiff_metadata[row]['image_filename'] or 'RC16' in geotiff_metadata[row]['image_filename'] or 'RM16' in geotiff_metadata[row]['image_filename']:
                idx = csv_track_data['rightimage'].index(geotiff_metadata[row]['image_filename'])
            elif 'FC16' in geotiff_metadata[row]['image_filename'] or 'LC16' in geotiff_metadata[row]['image_filename'] or 'RM16' in geotiff_metadata[row]['image_filename']:
                idx = csv_track_data['leftimage'].index(geotiff_metadata[row]['image_filename'])

            geotiff_metadata[row]['altitude_sensor'] = csv_track_data['altitude'][idx]
            geotiff_metadata[row]['depth_sensor']    = csv_track_data['depth'][idx]
            geotiff_metadata[row]['latitude']        = csv_track_data['latitude'][idx]
            geotiff_metadata[row]['longitude']       = csv_track_data['longitude'][idx]
            geotiff_metadata[row]['time']            = '%d%02d%02dT%02d%02d%02dZ' % (
                csv_track_data['year'][idx],
                csv_track_data['month'][idx],
                csv_track_data['day'][idx],
                csv_track_data['hour'][idx],
                csv_track_data['minute'][idx],
                csv_track_data['second'][idx])
            geotiff_metadata[row]['depth']           = csv_track_data['altitude'][idx] + csv_track_data['depth'][idx]
            geotiff_metadata[row]['campaign_code']   = campaign_name
            geotiff_metadata[row]['dive_code']       = dive_name

            if 'label' in csv_track_data.keys():
                geotiff_metadata[row]['cluster_tag'] = csv_track_data['label'][idx]
            else:
                geotiff_metadata[row]['cluster_tag'] = 9999

        except Exception:
            logger.warning('Warning %s not in CSV track file' % geotiff_metadata[row]['image_filename'])
            # if the image is not in the CSV track file, we can still have
            # access to some info about it
            geotiff_metadata[row]['altitude_sensor'] = geotiff_metadata[row - 1]['altitude_sensor']
            geotiff_metadata[row]['depth_sensor']    = geotiff_metadata[row - 1]['depth_sensor']
            geotiff_metadata[row]['latitude']        = geotiff_metadata[row]["up_left_lat"]
            geotiff_metadata[row]['longitude']       = geotiff_metadata[row]["up_left_lon"]
            geotiff_metadata[row]['depth']           = geotiff_metadata[row - 1]['depth']
            geotiff_metadata[row]['campaign_code']   = campaign_name
            geotiff_metadata[row]['dive_code']       = dive_name
            geotiff_metadata[row]['cluster_tag']     = 9999

            img_name                      = geotiff_metadata[row]['image_filename']
            img_name_time_digits          = [int(s) for s in img_name.split('_') if s.isdigit()]
            geotiff_metadata[row]['time'] = '%dT%06dZ' % (img_name_time_digits[0], img_name_time_digits[1])

    return geotiff_metadata


def match_netcdf_data_geotiff_metadata(netcdf_data, geotiff_metadata):
    """
    match data found in both netcdf files with images based on nearest time. The
    csv track file being the link between an image and its associated time
    Add information to geotiff_metadata
    """
    for row, rest in enumerate(geotiff_metadata):
        time_geotiff = datetime.strptime(geotiff_metadata[row]['time'], "%Y%m%dT%H%M%SZ")

        # first netcdf file is IMOS_AUV_ST*
        if netcdf_data[0] != []:
            dates = netcdf_data[0]['TIME']
            [idx, time_value] = min(enumerate(dates), key=lambda x: x[1] - time_geotiff if x[1] > time_geotiff else timedelta.max)
            if idx <= len(netcdf_data[0]['PSAL']):
                geotiff_metadata[row]['sea_water_salinity'] = netcdf_data[0]['PSAL'][idx]
            if idx <= len(netcdf_data[0]['TEMP']):
                geotiff_metadata[row]['sea_water_temperature'] = netcdf_data[0]['TEMP'][idx]

        # second netcdf file is IMOS_AUV_B*
        if netcdf_data[1] != []:
            dates = netcdf_data[1]['TIME']
            [idx, time_value] = min(enumerate(dates), key=lambda x: x[1] - time_geotiff if x[1] > time_geotiff else timedelta.max)
            if idx <= len(netcdf_data[1]['CDOM']):
                geotiff_metadata[row]['colored_dissolved_organic_matter'] = netcdf_data[1]['CDOM'][idx]
            if idx <= len(netcdf_data[1]['CPHL']):
                geotiff_metadata[row]['chlorophyll_concentration_in_sea_water'] = netcdf_data[1]['CPHL'][idx]
            if idx <= len(netcdf_data[1]['OPBS']):
                geotiff_metadata[row]['backscattering_ratio'] = netcdf_data[1]['OPBS'][idx]

    return geotiff_metadata


def table_data_csv(geotiff_metadata, csv_output_path):
    """
    Append date to the DATA_... csv file. This file is harvested by a talend harvester
    to populate pgsql tables used by the AUV viewer
    need to finish the function to add campaign, dive, and proper destination,
    also order fieldnames
    """
    writenames = "campaign_code,dive_code,image_filename,longitude,latitude,"\
        "image_width,depth_sensor,altitude_sensor,depth,"\
        "sea_water_temperature,sea_water_salinity,"\
        "chlorophyll_concentration_in_sea_water,backscattering_ratio,"\
        "colored_dissolved_organic_matter,time,cluster_tag,up_left_lon,"\
        "up_left_lat,up_right_lon,up_right_lat,low_right_lon,low_right_lat,"\
        "low_left_lon,low_left_lat".split(",")

    write_csv_dict_header_reorder(csv_output_path, writenames, geotiff_metadata, 'append')


def write_csv_dict_header_reorder(csv_output_path, header_order, dict_list, option='write'):
    """ write a csv file to csv_path, by changing the order of dict_list to the
    order specified by header_order, so that the order of columns in the csv file is forced.
    Example:
        header_order = "col1,col2,col3".split(",")
        dict_list[0] = {'col3' : 'test' , 'col2' : 'test', 'col3' : 'test'}
    """
    header     = dict_list[0].keys()
    if option == 'write':
        option = 'w'
    elif option == 'append':
        option = 'a'
    else:
        option = 'w'

    with open(csv_output_path, option) as outcsv:
        writer       = csv.writer(outcsv)
        name2index   = dict((name, index) for index, name in enumerate(header))
        writeindices = [name2index[name] for name in header_order]
        reorderfunc  = operator.itemgetter(*writeindices)
        writer.writerow(header_order)
        for row in dict_list:
            writer.writerow(reorderfunc(list(row.values())))


def compute_track_distance(geotiff_metadata):
    lon_seq = [x['up_right_lon'] for x in geotiff_metadata]
    lat_seq = [x['up_right_lat'] for x in geotiff_metadata]

    total_distance = 0
    for row, rest in enumerate(lat_seq):
        if row != len(lat_seq) - 1:
            total_distance += distance_btw_latlon([lat_seq[row], lon_seq[row]],
                                                  [lat_seq[row + 1], lon_seq[row + 1]])

    return total_distance


def get_dive_number(dive_name, campaign_path):
    """
    read the dive path to guess the dive number
    """
    # look for 2 digits in a row between 2 underscores
    a = re.search('_(\d{2})_', dive_name)
    if a is not None:
        return int(a.group(1))

    # or look for 2 digits and a letter a b
    a = re.search('_(\d{2})\w_', dive_name)
    if a is not None:
        return int(a.group(1)[0:2])

    # if still in the function, we have to invent a dive number
    list_dive   = list_dives(campaign_path)
    dive_number = [i for i, x in enumerate(list_dive) if x == dive_name]
    return dive_number[0] + 1


def get_abstract_dive(dive_path):

    # return an empty string for abstract if there aren't any NetCDF directory. using global variable so users doesn't
    # get asked twice about keeping on processing the dive or not
    global IS_NETCDF_EXISTS
    if IS_NETCDF_EXISTS == False:
        return ''

    netcdf_dir_dive_path = _netcdf_dive_path(dive_path)
    abstract = ''
    for file in os.listdir(netcdf_dir_dive_path):
        netcdf_file_path = os.path.join(netcdf_dir_dive_path, file)
        if 'IMOS_AUV_ST_' in netcdf_file_path:
            try:
                netcdf_file_obj = Dataset(netcdf_file_path, mode='r')
                abstract        = netcdf_file_obj.abstract
            except Exception:
                logger.warning('No ST data in NetCDF. Check with facility this is correct')
                return

    return abstract


def table_metadata_csv(geotiff_metadata, campaign_path, dive_name, csv_output_path):
    """
    Create the TABLE_METADATA_... csv file. This file is harvested by a talend harvester
    to populate pgsql tables used by the AUV viewer
    """
    dive_report_path = os.path.join(os.path.basename(campaign_path), 'all_reports', '%s_report.pdf' % dive_name)
    campaign_name    = os.path.basename(campaign_path)
    for file in os.listdir(os.path.join(campaign_path, dive_name, 'track_files')):
        if file.endswith('kml'):
            kml_path = os.path.join(campaign_name, dive_name, 'track_files', file)

    image_path = os.path.basename(_geotiff_dive_path(os.path.join(campaign_path, dive_name)))
    dive_number = get_dive_number(dive_name, campaign_path)

    dive_metadata_uuid = str(uuid.uuid4())
    platform_code      = 'SIRIUS'
    pattern            = 'Trajectory'
    abstract           = get_abstract_dive(os.path.join(campaign_path, dive_name))

    lon_seq   = [x['up_right_lon'] for x in geotiff_metadata]
    lat_seq   = [x['up_right_lat'] for x in geotiff_metadata]
    depth_seq = [x['depth'] for x in geotiff_metadata]

    geospatial_lat_min      = min(lat_seq)
    geospatial_lat_max      = max(lat_seq)
    geospatial_lon_min      = min(lon_seq)
    geospatial_lon_max      = max(lon_seq)
    geospatial_vertical_min = min(depth_seq)
    geospatial_vertical_max = max(depth_seq)
    distance_covered_in_m   = compute_track_distance(geotiff_metadata)

    time_seq            = [x['time'] for x in geotiff_metadata]
    time_coverage_start = min([datetime.strptime(time, '%Y%m%dT%H%M%SZ') for time in time_seq]).strftime('%Y%m%dT%H%M%SZ')
    time_coverage_end   = max([datetime.strptime(time, '%Y%m%dT%H%M%SZ') for time in time_seq]).strftime('%Y%m%dT%H%M%SZ')

    dive_regexp = re.match('r\d+_\d+_(\w+)', dive_name)

    with open(csv_output_path, 'w') as outcsv:
        outcsv.write('dive_number,dive_name,dive_metadata_uuid,facility_code,campaign_code,'
                     'dive_code,distance_covered_in_m,number_of_images,image_path,'
                     'abstract,platform_code,pattern,dive_report_path,kml_path,'
                     'geospatial_lat_min, geospatial_lon_min,'
                     'geospatial_lat_max, geospatial_lon_max,'
                     'geospatial_vertical_min, geospatial_vertical_max,'
                     'time_coverage_start, time_coverage_end\n')

        outcsv.write('%d,%s,%s,%s,%s,%s,%f,%d,%s,\"%s\",%s,%s,%s,%s,%f,%f,%f,%f,%f,%f,%s,%s\n'
                     % (dive_number, dive_regexp.group(1).replace('_', ' '),
                        dive_metadata_uuid, 'AUV', campaign_name, dive_name,
                        distance_covered_in_m, len(time_seq), image_path, abstract, platform_code,
                        pattern, dive_report_path, kml_path,
                        geospatial_lat_min, geospatial_lon_min,
                        geospatial_lat_max, geospatial_lon_max,
                        geospatial_vertical_min, geospatial_vertical_max,
                        time_coverage_start, time_coverage_end))


def list_dives(campaign_path):
    """
    lists the dives available for one campaign
    """
    list_dive = []
    for dir in os.listdir(campaign_path):
        if dir.startswith('r20'):
            list_dive.append(dir)

    list_dive.sort()
    return list_dive


def list_recursively_files_abs_path(path):
    """
    return a list containing the absolute path of files recursively found in a path
    :param path:
    :return:
    """
    filelist = []
    for filename in glob.glob('{path}/**'.format(path=path), recursive=True):
        if os.path.isfile(filename):
            filelist.append(os.path.abspath(filename))
    return filelist


def copy_manifest_reports_to_incoming(campaign_path):
    """ copy manifest file containing campaign pdf reports to incoming"""
    campaign_name    = os.path.basename(campaign_path)
    all_reports_path = os.path.join(campaign_path, 'all_reports')

    if os.path.exists(all_reports_path):
        file_list = list_recursively_files_abs_path(all_reports_path)
        with open(os.path.join(os.environ['INCOMING_DIR'], 'AUV', '%s-alldives.pdfreports.manifest' % campaign_name), 'w') as f:
            for item in file_list:
                f.write("%s\n" % item)


def copy_manifest_dive_to_incoming(campaign_dive_name, dive_path):
    """
    create a manifest file for all data in a dive to be pushed to incoming directory. The dive manifest files are split
    per lines_per_file value, ie 4096 files to be pushed at once in the pipeline
    :param campaign_dive_name:
    :param dive_path:
    :return:
    """
    file_list = list_recursively_files_abs_path(dive_path)
    out_filename = os.path.join(os.environ['INCOMING_DIR'], 'AUV',
                                '%s.{}.dive.manifest' % campaign_dive_name)

    lines_per_file = 2**12

    for file_number, lines in groupby(enumerate(file_list), key=lambda x: x[0] // lines_per_file):
        with open(out_filename.format(file_number), 'w') as outfile:
            for item in lines:
                outfile.write("%s\n" % item[1])


def copy_manifest_dive_data_to_incoming(output_data, thumbnail=True):
    """
    A manifest file per type of files to push and per dive can be created.
        1- manifest containing links to both netcdf file (ST and B)
        2- manifest containing links to both DATA_... csv output file
        3- manifest containing links to all generated thumbnails
        4- manifest containing link to full dive folder in order to do async upload vi incoming handler
        5- manifest containing link to report files
    """
    dive_path            = output_data[2]
    campaign_dive_name   = '%s-%s' % (dive_path.split(os.path.sep)[-2], dive_path.split(os.path.sep)[-1])
    global IS_NETCDF_EXISTS

    if IS_NETCDF_EXISTS:
        netcdf_dir_dive_path = _netcdf_dive_path(dive_path)
        nc_files             = []

        for file in os.listdir(netcdf_dir_dive_path):
            if file.endswith('.nc'):
                nc_files.append(os.path.join(netcdf_dir_dive_path, file))

    table_data_csv_ouput = output_data[0]
    thumbnail_path_list  = []
    thumbnail_dir_path   = output_data[1]
    full_res_path_list   = []
    full_res_dir_path    = 'full_res'.join(thumbnail_dir_path.rsplit('thumbnails', 1))

    if thumbnail_dir_path is not None:
        for file in os.listdir(thumbnail_dir_path):
            if file.endswith('.jpg'):
                thumbnail_path_list.append(os.path.join(thumbnail_dir_path, file))

    if full_res_dir_path is not None:
        for file in os.listdir(full_res_dir_path):
            if file.endswith('.jpg'):
                full_res_path_list.append(os.path.join(full_res_dir_path, file))

    logger.info('Move AUV data to INCOMING_DIR')

    if IS_NETCDF_EXISTS:
        with open(os.path.join(os.environ['INCOMING_DIR'], 'AUV', '%s.netcdf.manifest' % campaign_dive_name), 'w') as f:
            [f.write('%s\n' % nc) for nc in nc_files]

    with open(os.path.join(os.environ['INCOMING_DIR'], 'AUV', '%s.csv.manifest' % campaign_dive_name), 'w') as f:
        f.write('%s\n' % table_data_csv_ouput)

    # optional
    if thumbnail is True:
        thumnbail_manifest_filename = '%s.images.manifest' % campaign_dive_name

        with open(os.path.join(AUV_WIP_DIR, thumnbail_manifest_filename), 'w') as f:
            [f.write('%s\n' % thumbnail_path) for thumbnail_path in thumbnail_path_list]
            [f.write('%s\n' % full_res_path) for full_res_path in full_res_path_list]

        shutil.copy(os.path.join(AUV_WIP_DIR, thumnbail_manifest_filename),
                    os.path.join(os.environ['INCOMING_DIR'], 'AUV', thumnbail_manifest_filename))

    # if os.path.exists(os.path.join(AUV_WIP_DIR, 'auvReporting.csv')):
    #     shutil.copy(os.path.join(AUV_WIP_DIR, 'auvReporting.csv'),
    #                 os.path.join(os.environ['INCOMING_DIR'], 'AUV', 'auvReporting.csv'))

    copy_manifest_dive_to_incoming(campaign_dive_name, dive_path)


def reporting(campaign_path, dive_name):
    """TODO : Finish function to write proper values
    Creates reporting information to populate a postgres table and used for
    reporting
    """
    campaign_name = os.path.basename(campaign_path)

    reporting_file_url = 'http://data.aodn.org.au/IMOS/AUV/auv_viewer_data/csv_outputs/auvReporting.csv'
    response           = urlopen(reporting_file_url)
    data               = StringIO.StringIO(response.read())  # removing StringIO wont work with DictReader
    read               = csv.DictReader(data)
    report_data        = []
    for row_read in read:
        report_data.append(row_read)

    already_exist = False
    for x, row in enumerate(report_data):
        if row['campaign_code'] == campaign_name and row['dive_code'] == dive_name:
            already_exist       = True
            already_exist_index = x

    if already_exist:
        index_report = already_exist_index
    else:
        index_report = -1

    report_data[index_report]['campaign_code']           = campaign_name
    report_data[index_report]['campaign_metadata_uuid']  = 'n.a.'
    report_data[index_report]['cdom']                    = 'YES'
    report_data[index_report]['cphl']                    = 'YES'
    report_data[index_report]['csv_track_file']          = 'YES'
    report_data[index_report]['data_folder']             = os.path.join(campaign_name, dive_name)
    report_data[index_report]['data_on_auv_viewer']      = 'YES'
    report_data[index_report]['data_on_portal']          = 'YES'
    report_data[index_report]['dive_code']               = dive_name
    report_data[index_report]['dive_code_metadata_uuid'] = ''
    report_data[index_report]['dive_report']             = 'YES'
    report_data[index_report]['geotiff']                 = 'YES'
    report_data[index_report]['mesh']                    = 'YES'
    report_data[index_report]['multibeam']               = 'YES'
    report_data[index_report]['opbs']                    = 'YES'
    report_data[index_report]['openLink']                = ''
    report_data[index_report]['psal']                    = 'YES'
    report_data[index_report]['temp']                    = 'YES'

    header_order = "campaign_code,campaign_metadata_uuid,dive_code,"\
        "dive_code_metadata_uuid,openLink,data_on_portal,data_on_auv_viewer,"\
        "data_folder,geotiff,mesh,multibeam,cdom,cphl,opbs,psal,temp,"\
        "csv_track_file,dive_report".split(",")
    write_csv_dict_header_reorder(os.path.join(AUV_WIP_DIR, 'auvReporting.csv'), header_order,
                                  report_data, 'write')


def process_campaign(campaign_path, create_thumbnail=True, push_data_to_incoming=False):
    campaign_name    = os.path.basename(campaign_path)
    campaign_wip_dir = os.path.join(AUV_WIP_DIR, campaign_name)
    if not os.path.exists(campaign_wip_dir):
        os.makedirs(os.path.join(AUV_WIP_DIR, campaign_name))

    def process_dive():
        """ sub function to process individual dive
        """
        global IS_NETCDF_EXISTS
        IS_NETCDF_EXISTS = True  # reset value for each new dive

        logger.info('Processing %s - %s' % (campaign_name, dive_name))
        dive_path        = os.path.join(campaign_path, dive_name)

        netcdf_data      = read_netcdf(dive_path)
        csv_track_data   = read_track_csv(dive_path)
        geotiff_list     = list_geotiff_dive(dive_path)
        if not geotiff_list:
            raise Exception('No Geotiffs file matched the following patterns *AC16* *FC16* *LC16* *RC16* *RM16*')

        # order is important, creating geotiff_metadata list of dict, containing
        # matching data between images, track file and netcdf files
        geotiff_metadata = geotiff_list_metadata(geotiff_list)
        geotiff_metadata = match_csv_track_info_with_geotiff(csv_track_data,
                                                             geotiff_metadata,
                                                             campaign_name,
                                                             dive_name)
        geotiff_metadata = match_netcdf_data_geotiff_metadata(netcdf_data,
                                                              geotiff_metadata)

        data_csv_output_path = os.path.join(AUV_WIP_DIR, campaign_name,
                                            'DATA_%s_%s.csv' %
                                            (campaign_name, dive_name))

        table_metadata_csv(geotiff_metadata, campaign_path, dive_name, data_csv_output_path)
        table_data_csv(geotiff_metadata, data_csv_output_path)

        thumbnail_dir_path = None
        if create_thumbnail:
            thumbnail_dir_path = os.path.join(AUV_WIP_DIR, 'thumbnails',
                                              campaign_name, dive_name, 'thumbnails')
            logger.info('Generating thumbnails')
            generate_geotiff_thumbnails_dive(geotiff_list, thumbnail_dir_path)

        # reporting(campaign_path, dive_name)

        if push_data_to_incoming:
            copy_manifest_dive_data_to_incoming([data_csv_output_path, thumbnail_dir_path, dive_path], create_thumbnail)

    dives = list_dives(campaign_path)
    for dive_name in dives:
        try:
            process_dive()
        except Exception as err:
            logger.error('Dive not processed:\n%s' % err)

        # special case for last dive being processed, we move the all_reports
        # folder to S3 as well
        if dive_name == dives[-1]:
            copy_manifest_reports_to_incoming(campaign_path)


def parse_arg():
    """
    create optional script arg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--campaign-path", type=str, help='campaign path', required=True)
    parser.add_argument("-n", "--no-thumbnail-creation", help="process or reprocess campaign without the creation of thumbnails", action="store_false", required=False)
    parser.add_argument("-p", "--push-to-incoming", help="push output data, and ALL AUV CAMPAIGN data to incoming dir for pipeline processing", action="store_true", required=False)
    args               = parser.parse_args()
    args.campaign_path = args.campaign_path.rstrip("//")

    return args


if __name__ == '__main__':
    """ example:
        auv_processing.py -c /vagrant/src/PS201502 -n  -> no creation of thumbnails
        auv_processing.py -c /vagrant/src/PS201502 -p  -> full process of campaign and push to incoming ALL data(viewer plus campaign data)
    """
    os.umask(0o002)
    # setup logging
    log_filepath  = os.path.join(AUV_WIP_DIR, 'auv.log')
    logging       = IMOSLogging()
    global logger
    logger        = logging.logging_start(log_filepath)

    args = parse_arg()
    process_campaign(args.campaign_path,
                     create_thumbnail=args.no_thumbnail_creation,
                     push_data_to_incoming=args.push_to_incoming)
