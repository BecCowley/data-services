# Converts XBT profile recorded by Turo XBT to standardised netCDF format ready for QC with PYQUEST
# A. Walsh V2 4/10/22

# V3 - adjustments to format for compatibility with MQUEST netCDF and changes to PyQUEST-XBT.py.
# For compatibility with MQUEST to IMOS netCDF converter change all LAT,LONG,DEPTH,DEPTH_RAW,TEMP,TEMP_RAW from type 'd' (double) to type 'f' (float)

####Usage####
# Before running this script:
# 1) create 2 folders: xbtdata_raw,xbtdata_standardised
# 2) Copy the raw data files (monthly folder contents from Turo XBT) dropXXX.nc to xbtdata_raw
# Ensure there are no duplicated raw files; include edited version of drop if any, not original unedited version
# General command:
# python TuroXBT2StdNetCDF-V3.py -i xbtdata_raw -o xbtdata_standardised -c YYNNNSS -s "HMAS XXXX"

# Where YYNNNSS = Cruise ID
# For Turo XBT a 'cruise' is normally a 1 month of data from 1 ship, as Turo auto organises data into monthly folders
# YY =last 2 digits of the observation year
# NNN= consecutive number of dataset recived in that year
# XX = 2 character abbreviation of the shipname e.g. SY=HMAS Sydney
# xbtdata_raw = input folder holding raw files from Turo XBT - dropXXX.nc
# xbtdata_standardised = output folder to hold standardised files produced by this script

# Example:
# python TuroXBT2StdNetCDF-V3.py -i xbtdata_raw_3 -o xbtdata_standardised_3 -c 21018AN -s "HMAS ANZAC"


####Output File names####
# General format - YYNNNSS-YYYYMMDDThhmmssZ-CCC.nc
# YYNNNSS = Cruise ID
# CCC = Consecutive No 000, 001, ...
# YYYYMMDDThhmmssZ = XBT drop UTC Date-time
# e.g. 20001SY-20210303T212721Z-000.nc

import string
import re
import os

from netCDF4 import Dataset
import datetime
from time import localtime, gmtime, strftime
from netCDF4 import date2num

import numpy as np

from optparse import OptionParser
import glob


##########FUNCTIONS#######################

def netCDFout(nco, n, cid, sNam, raw_netCDF_file):
    calendar = 'standard'
    units = 'days since 1950-01-01 00:00'
    consec = str(f'{n:03}')

    # Gather the varaibles
    lat = nco.variables['latitude']  # lat[0] is actual value
    lon = nco.variables['longitude']
    # date-time
    '''
	  int woce_date(time);
      string woce_date:long_name = "WOCE date";
      string woce_date:units = "yyyymmdd UTC";
    int woce_time(time);
      string woce_time:long_name = "WOCE time";
      string woce_time:units = "hhmmss UTC";
	'''
    yyyymmdd = str(nco.variables['woce_date'][0])
    yyyy = yyyymmdd[0:4]
    yyyyI = int(yyyy)
    mm = yyyymmdd[4:6]
    mmI = int(mm)
    dd = yyyymmdd[6:8]
    ddI = int(dd)
    # >>> n = 4
    # >>> print(f'{n:03}') # Preferred method for formatted string literals for python >= 3.6
    # 004
    # dtstr=f'{yyyymmdd:s}T{timeInt:06}'
    timeInt = nco.variables['woce_time'][0]
    timeStr = str(f'{timeInt:06}')
    hh = timeStr[0:2]
    hhI = int(hh)
    mins = timeStr[2:4]
    minsI = int(mins)
    ss = timeStr[4:6]
    ssI = int(ss)

    dt = "%s%s%sT%s%s%sZ" % (yyyy, mm, dd, hh, mins, ss)
    dtISO8601 = "%s-%s-%sT%s:%s:%sZ" % (yyyy, mm, dd, hh, mins, ss)

    # Time in days since 1950 for netCDF file

    d = datetime.datetime(yyyyI, mmI, ddI, hhI, minsI, ssI)
    depoch = date2num(d, units=units, calendar=calendar)

    depths = nco.variables['depth'][:]  # 1D array
    # Should use 'temperature' array not 'procTemperature' as that may be a filtered version if the Turo XBT despike option was used.
    # temps = nco.variables['procTemperature'][0,:,0,0] #4D array access
    temps = nco.variables['temperature'][0, :, 0, 0]  # 4D array access

    # Global attributes
    '''
	if n==0:
		for name in nco.ncattrs():
			print("Global attr {} = {}".format(name, getattr(nco, name)))
	'''
    # Needed globals
    # string :ReleaseVersion = "Version: 5.03.00";
    # string :Code = "052"; - Probe type + fall rate code
    # string :HardwareSerialNo = "69";
    # string :SerialNo = "1302539"; - Probe serial #
    # string :PreDropComments = "depth 632.1m";
    # string :PostDropComments = " ";
    # Access individual global from Turo XBT source file
    relVer = nco.getncattr('ReleaseVersion')
    ptype = nco.getncattr('Code')
    hardwareSerNo = nco.getncattr('HardwareSerialNo')
    pSerNo = nco.getncattr('SerialNo')
    preDropCmnt = nco.getncattr('PreDropComments')
    postDropCmnt = nco.getncattr('PreDropComments')

    # Construct output (standardised) netCDF filename

    ncfilen = os.path.join(outDir, cid + '_' + dt + '_' + consec + '.nc')

    print(ncfilen, n, dt, lat[0], lon[0], depths[0], temps[0], relVer, ptype, pSerNo, hardwareSerNo, preDropCmnt,
          postDropCmnt)

    # Open the netCDF file

    ncoutf = Dataset(ncfilen, 'w', format='NETCDF4')

    # Create Dimensions

    ncoutf.createDimension('DEPTH', 0)  # 0 means unlimited size for depth dimension
    depthdim = ('DEPTH',)

    '''
	ncoutf.createDimension('TIME',1)
	timedim=('TIME',) #a tuple
	'''

    # Create variables
    # The createVariable method has two mandatory arguments,
    # the variable name (a Python string),
    # and the variable datatype. and an optional dimension(s)
    # The variable's dimensions are given by a tuple containing the dimension names
    # (defined previously with createDimension)

    # Keep raw arrays of any variables (as backup) which may be changed through QC

    TIME_RAW = ncoutf.createVariable('TIME_RAW', 'd', fill_value=999999.)
    TIME = ncoutf.createVariable('TIME', 'd', fill_value=999999.)
    TIME_quality_control = ncoutf.createVariable('TIME_quality_control', 'b', fill_value=np.byte(99))

    # For compatibility with MQUEST converter change all LAT,LONG,DEPTH,DEPTH_RAW,TEMP,TEMP_RAW from type 'd' (double) to type 'f' (float)
    LATITUDE_RAW = ncoutf.createVariable('LATITUDE_RAW', 'f', fill_value=999999.)
    LATITUDE = ncoutf.createVariable('LATITUDE', 'f', fill_value=999999.)
    LATITUDE_quality_control = ncoutf.createVariable('LATITUDE_quality_control', 'b', fill_value=np.byte(99))

    LONGITUDE_RAW = ncoutf.createVariable('LONGITUDE_RAW', 'f', fill_value=999999.)
    LONGITUDE = ncoutf.createVariable('LONGITUDE', 'f', fill_value=999999.)
    LONGITUDE_quality_control = ncoutf.createVariable('LONGITUDE_quality_control', 'b', fill_value=np.byte(99))

    DEPTH = ncoutf.createVariable('DEPTH', 'f', depthdim, fill_value=999999.)
    DEPTH_quality_control = ncoutf.createVariable('DEPTH_quality_control', 'b', depthdim, fill_value=np.byte(99))

    # For compatibility with MQUEST converter change 'TEMPERATURE' to 'TEMP'
    TEMP_RAW = ncoutf.createVariable('TEMP_RAW', 'f', depthdim, fill_value=999999.)
    TEMP = ncoutf.createVariable('TEMP', 'f', depthdim, fill_value=999999.)
    TEMP_quality_control = ncoutf.createVariable('TEMP_quality_control', 'b', depthdim, fill_value=np.byte(99))

    XBT_fault_type = ncoutf.createVariable('XBT_fault_type', 'b', depthdim, fill_value=np.byte(99))

    # Set variable attributes

    TIME_RAW.standard_name = "time"
    TIME_RAW.long_name = "time uncorrected"
    TIME_RAW.units = "days since 1950-01-01 00:00:00Z"
    TIME_RAW.axis = "T"
    TIME_RAW.valid_min = 0.
    TIME_RAW.valid_max = 999999.

    TIME.standard_name = "time"
    TIME.units = "days since 1950-01-01 00:00:00Z"
    TIME.axis = "T"
    TIME.valid_min = 0.
    TIME.valid_max = 999999.
    TIME.ancillary_variables = "TIME_quality_control"
    TIME.calendar = 'gregorian'

    TIME_quality_control.long_name = "quality flags for time"
    TIME_quality_control.standard_name = "time status_flag"
    TIME_quality_control.quality_control_conventions = "IMOS standard flags"
    TIME_quality_control.valid_min = np.byte(0)
    TIME_quality_control.valid_max = np.byte(9)
    TIME_quality_control.flag_values = np.byte(list(range(0, 10)))
    TIME_quality_control.flag_meanings = "No_QC_performed Good_data Probably_good_data Bad_data_that_are_potentially_correctable Bad_data Value_changed Not_used Not_used Not_used Missing_value"

    LATITUDE_RAW.reference_datum = "geographical coordinates, WGS84 projection"
    LATITUDE_RAW.axis = "Y"
    LATITUDE_RAW.standard_name = "latitude"
    LATITUDE_RAW.long_name = "latitude uncorrected"

    LATITUDE_RAW.units = "degrees_north"

    LATITUDE.reference_datum = "geographical coordinates, WGS84 projection"
    LATITUDE.ancillary_variables = "LATITUDE_quality_control"
    LATITUDE.axis = "Y"
    LATITUDE.standard_name = "latitude"
    LATITUDE.long_name = "latitude"
    LATITUDE.units = "degrees_north"

    LATITUDE_quality_control.long_name = "quality flags for latitude"
    LATITUDE_quality_control.standard_name = "latitude status_flag"
    LATITUDE_quality_control.quality_control_conventions = "IMOS standard flags"
    LATITUDE_quality_control.valid_min = np.byte(0)
    LATITUDE_quality_control.valid_max = np.byte(9)
    LATITUDE_quality_control.flag_values = np.byte(list(range(0, 10)))
    LATITUDE_quality_control.flag_meanings = "No_QC_performed Good_data Probably_good_data Bad_data_that_are_potentially_correctable Bad_data Value_changed Not_used Not_used Not_used Missing_value"

    LONGITUDE_RAW.reference_datum = "geographical coordinates, WGS84 projection"
    LONGITUDE_RAW.axis = "X"
    LONGITUDE_RAW.standard_name = "longitude"
    LONGITUDE_RAW.long_name = "longitude uncorrected"
    LONGITUDE_RAW.units = "degrees_east"

    LONGITUDE.reference_datum = "geographical coordinates, WGS84 projection"
    LONGITUDE.ancillary_variables = "LONGITUDE_quality_control"
    LONGITUDE.axis = "X"
    LONGITUDE.standard_name = "longitude"
    LONGITUDE.long_name = "longitude"
    LONGITUDE.units = "degrees_east"

    LONGITUDE_quality_control.long_name = "quality flags for longitude"
    LONGITUDE_quality_control.standard_name = "longitude status_flag"
    LONGITUDE_quality_control.quality_control_conventions = "IMOS standard flags"
    LONGITUDE_quality_control.valid_min = np.byte(0)
    LONGITUDE_quality_control.valid_max = np.byte(9)
    LONGITUDE_quality_control.flag_values = np.byte(list(range(0, 10)))
    LONGITUDE_quality_control.flag_meanings = "No_QC_performed Good_data Probably_good_data Bad_data_that_are_potentially_correctable Bad_data Value_changed Not_used Not_used Not_used Missing_value"

    DEPTH.positive = "down"
    DEPTH.ancillary_variables = "DEPTH_quality_control"
    DEPTH.valid_min = 0.
    DEPTH.valid_max = 12000.
    DEPTH.standard_name = "depth"
    DEPTH.units = "m"

    DEPTH.axis = "Z"
    DEPTH.long_name = "depth"

    DEPTH_quality_control.long_name = "quality flags for depth"
    DEPTH_quality_control.standard_name = "depth status_flag"
    DEPTH_quality_control.quality_control_conventions = "IMOS standard flags"
    DEPTH_quality_control.valid_min = np.byte(0)
    DEPTH_quality_control.valid_max = np.byte(9)
    DEPTH_quality_control.flag_values = np.byte(list(range(0, 10)))
    DEPTH_quality_control.flag_meanings = "No_QC_performed Good_data Probably_good_data Bad_data_that_are_potentially_correctable Bad_data Value_changed Not_used Not_used Not_used Missing_value"

    # For compatibility with MQUEST converter change 'TEMPERATURE' to 'TEMP'
    TEMP_RAW.positive = "down"
    TEMP_RAW.valid_min = -2.5
    TEMP_RAW.valid_max = 40.
    TEMP_RAW.axis = "Z"
    TEMP_RAW.coordinates = "TIME LATITUDE LONGITUDE DEPTH"
    TEMP_RAW.long_name = "sea_water_temperature"
    TEMP_RAW.standard_name = "sea_water_temperature"
    TEMP_RAW.units = "Celsius"

    TEMP.positive = "down"
    TEMP.ancillary_variables = "TEMP_quality_control XBT_fault_type"
    TEMP.valid_min = -2.5
    TEMP.valid_max = 40.
    TEMP.axis = "Z"
    TEMP.coordinates = "TIME LATITUDE LONGITUDE DEPTH"
    TEMP.long_name = "sea_water_temperature"
    TEMP.standard_name = "sea_water_temperature"
    TEMP.units = "Celsius"

    TEMP_quality_control.long_name = "quality flag for sea_water_temperature"
    TEMP_quality_control.standard_name = "sea_water_temperature status_flag"
    TEMP_quality_control.quality_control_conventions = "IMOS standard flags"
    TEMP_quality_control.valid_min = np.byte(0)
    TEMP_quality_control.valid_max = np.byte(9)
    TEMP_quality_control.flag_values = np.byte(list(range(0, 10)))
    TEMP_quality_control.flag_meanings = "No_fault_check Good_data Probably_good_data Bad_data_that_are_potentially_correctable Bad_data Value_changed Not_used Not_used Not_used Missing_value"

    # XBT Fault type			Code	GTSPP/IMOS QC Flag -(TEMPERATURE_quality_control)
    # No_fault_check 		0		0
    # Good_data				1		1
    # Hit_bottom				2		3
    # Wire_break				3		4
    # Wire_stretch			4		3
    # Insulation_penetration	5		3
    # Spike_accept			6		2
    # Electrical_noise		7		4
    # Temperature_offset		8		3
    # Unknown_fault			9		4
    # Probably_good			10		2 (set on depths deeper than interp/spike_accept segment) - REMOVED - NOT a fault but is QC class

    XBT_fault_type.long_name = "XBT fault type code"
    XBT_fault_type.valid_min = np.byte(0)
    XBT_fault_type.valid_max = np.byte(9)
    XBT_fault_type.flag_values = np.byte(list(range(0, 10)))
    XBT_fault_type.flag_meanings = "No_QC_performed Good_data Hit_bottom Wire_break Wire_stretch Insulation_penetration Spike_accept Electrical_noise Temperature_offset Unknown_fault"

    # Set Global variables
    ncoutf.source_filename = raw_netCDF_file
    ncoutf.cruiseID = cid
    # Profile Id
    pid = "%s_%s_%03d" % (cid, dt, n)
    ncoutf.XBT_uniqueid = pid
    ncoutf.shipname = sNam

    ncoutf.release_version = relVer
    ncoutf.probe_type = ptype
    ncoutf.hardware_serial_no = hardwareSerNo
    ncoutf.probe_serial_no = pSerNo
    ncoutf.preDropCmnt = preDropCmnt
    ncoutf.postDropCmnt = postDropCmnt
    ncoutf.qcstatus = 0  # 0=not qced, 1=qc in progress
    ncoutf.hit_bottom_flag = 'N'  # default to N,ill be set to "Y" in PyQUEST XBT QC if probe hit bottom
    ncoutf.featureType = "profile"
    ncoutf.Conventions = "CF-1.9"  # run output through the CF-Checker

    ncoutf.geospatial_lat_min = lat[0]
    ncoutf.geospatial_lat_max = lat[0]
    ncoutf.geospatial_lon_min = lon[0]
    ncoutf.geospatial_lon_max = lon[0]

    ncoutf.geospatial_vertical_min = depths[0]
    ncoutf.geospatial_vertical_max = depths[-1]
    ncoutf.time_coverage_start = dtISO8601
    ncoutf.time_coverage_end = dtISO8601

    # title = "RAN XBT data, Ship: %s, ID: %s" % (sNam,pid)
    title = "RAN XBT data, Ship: {0}, ID: {1}".format(sNam, pid)
    # print("title:"+title+":")

    ncoutf.title = title

    localnow = strftime("%Y-%m-%dT%H:%M:%SL", localtime())
    ncoutf.date_created = localnow

    # Set variable arrays

    TIME_RAW[0] = depoch
    TIME[0] = depoch
    TIME_quality_control[0] = np.byte(1)  # By default good=1 (maybe set to bad(3)/changed(5) on L1 metadata QC)

    LATITUDE_RAW[0] = lat[0]
    LATITUDE[0] = lat[0]
    LATITUDE_quality_control[0] = np.byte(1)  # By default good=1 (maybe set to bad(3)/changed(5) on L1 metadata QC)

    # Longitude in Turo netCDF is degrees east 0 -360 deg convention
    LONGITUDE_RAW[0] = lon[0]
    LONGITUDE[0] = lon[0]
    LONGITUDE_quality_control[0] = np.byte(1)  # By default good=1 (maybe set to bad(3)/changed(5) on L1 metadata QC)

    DEPTH[:] = depths[:]
    DEPTH_quality_control[:] = np.ones(len(DEPTH), dtype=np.byte)  # for XBT assume depths all good - rarely changed

    TEMP_RAW[:] = temps[:]
    TEMP[:] = temps[:]
    TEMP_quality_control[:] = np.zeros(len(DEPTH), dtype=np.byte)  # set QC flag initially to 0=no qc
    XBT_fault_type[:] = np.zeros(len(DEPTH), dtype=np.byte)  # set fault type initially to 0=no qc


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", dest="inputDir",
                      help="the input raw netCDF file directory", metavar="INDIR")

    parser.add_option("-o", dest="outputDir",
                      help="the output file directory path (relative)", metavar="OUTDIR")

    parser.add_option("-c", dest="cruiseID",
                      help="the cruise ID", metavar="CID")

    parser.add_option("-s", dest="sName",
                      help="the ship name", metavar="sNam")

    (options, args) = parser.parse_args()

    inDir = options.inputDir
    outDir = options.outputDir
    cid = options.cruiseID
    sNam = options.sName

    # Read the cruise and translate

    files_pattern = os.path.join(inDir, "*.nc")
    # print files_pattern

    files = sorted(glob.glob(files_pattern))

    n = 0
    for file in files:  # read/write loop
        nco = Dataset(file, 'r')
        nco.set_auto_maskandscale(False)
        raw_netCDF_file = os.path.basename(file)
        # Write function
        netCDFout(nco, n, cid, sNam, raw_netCDF_file)
        n += 1
