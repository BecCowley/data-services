#!/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import DatafabricConnection,folderCopier

if __name__ == "__main__":

    df = DatafabricConnection.DatafabricConnection()

    if df.isDfMounted():
       print "Connected to the datafabric"
       try:
           os.system("/usr/local/bin/matlab -nodisplay -r  \"cd '/usr/local/harvesters/matlab_3/svn/acorn/trunk'; addpath(fullfile('.','Util')); acorn_summary('WERA', true)\"")
       except Exception, e:
           print ("ERROR: " + str(e))
           sys.exit()
       
       f = folderCopier.folderCopier()

       f.processFiles("/var/lib/matlab_3/ACORN/WERA/radial_QC/output/datafabric/gridded_1havg_currentmap_QC/CBG","/home/matlab_3/df_root/opendap/ACORN/gridded_1h-avg-current-map_QC/CBG",'nc')
       f.processFiles("/var/lib/matlab_3/ACORN/WERA/radial_QC/output/datafabric/gridded_1havg_currentmap_QC/SAG","/home/matlab_3/df_root/opendap/ACORN/gridded_1h-avg-current-map_QC/SAG",'nc')
       f.processFiles("/var/lib/matlab_3/ACORN/WERA/radial_QC/output/datafabric/gridded_1havg_currentmap_QC/ROT","/home/matlab_3/df_root/opendap/ACORN/gridded_1h-avg-current-map_QC/ROT",'nc')
       f.processFiles("/var/lib/matlab_3/ACORN/WERA/radial_QC/output/datafabric/gridded_1havg_currentmap_QC/COF","/home/matlab_3/df_root/opendap/ACORN/gridded_1h-avg-current-map_QC/COF",'nc')

       f.close()
    else:
       print "Failed to connect to the datafabric so exiting."
       sys.exit()