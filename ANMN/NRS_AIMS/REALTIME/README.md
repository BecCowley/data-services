NRS AIMS Darwin Yongala Beagle realtime data download
=============

This script downloads data from the AIMS web service and modify the netcdf files to be
CF and IMOS compliant. The data is NRT, and comes from the following stations : Darwin, Yongala and Beagle.
The script can be run as many times as desired.


## Usage

Type in your shell ```./anmn_nrs_aims.py```

## Data debug
A test is in place for each run of the script. We download a part of a channel, 
run a md5 checksum and compare with what we should have. 

If the md5 value is different, the script won't run. This test is necessary as 
we had in the past unwanted changes from AIMS. 

If this happens: 
1) go to $WIP_DIR/[script_output_dir],
2) list the latest **nc_unittest_*** files ```ll -lrt nc_unittest_*```
3) check the latest ```nc_unittest...nc``` content with vimdiff/ncdump to assure the new file is mostly the same (and is not introducing renamed variables/gatts ...)
4) if the file is all good, grab the ```md5sum``` value from the filename (example: nc_unittest_**78c6386529faf9dc2272e9bed5ed7fa2**.nc) and replace it in https://github.com/aodn/data-services/blob/master/ANMN/NRS_AIMS/REALTIME/anmn_nrs_aims.py#L305

## Delete platform or channel for re-download/re-process
```bash
$ export data_wip_path=$WIP_DIR/ANMN/NRS_AIMS_Darwin_Yongala_data_rss_download_temporary
$ cd $DATA_SERVICES_DIR/lib/aims
$ mkdir $WIP_DIR/ANMN/NRS_AIMS_Darwin_Yongala_data_rss_download_temporary/.bckp
# backup pickle files 
cp $WIP_DIR/ANMN/NRS_AIMS_Darwin_Yongala_data_rss_download_temporary/*.pickle $WIP_DIR/ANMN/NRS_AIMS_Darwin_Yongala_data_rss_download_temporary/.bckp
$ ipython                                                                
```

Then in the python console
```python
# delete all entries from pickle file (qc level 0 and 1) for the platform Beagle
from realtime_util import delete_platform_entries_from_pickle
delete_platform_entries_from_pickle(0, 'Beagle')
delete_platform_entries_from_pickle(1, 'Beagle')

# also can delete a specific channe
from realtime_util import delete_channel_id_from_pickle
delete_channel_id_from_pickle(0, '261')
```


Files need to be removed from s3 bucket storage(remove ```echo``` and ```head -1``` once happy
```bash
for f in `s3_ls_recur IMOS/ANMN/NRS/REAL_TIME/NRSBEA/ | head -1` ; do
    echo po_s3_del $f
done
```

## Contact Support
for support contact:
Email: laurent.besnard@utas.edu.au
