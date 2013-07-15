#!/bin/bash
# rsync ACORN radial FV01 data from STAGING to OPENDAP

date
tic=$(date +%s.%N)
echo ' '

# Need to set the environment variables relevant for ACORN
source /home/ggalibert/DEFAULT_PATH.env
source /home/ggalibert/STORAGE.env
source /home/ggalibert/ACORN.env

# No need to delete empty files/directories, done by FV00 process before
#find $STAGING/ACORN/radial/ -type f -empty -delete
#find $STAGING/ACORN/radial/ -type d -empty -delete

# we need to prevent from copying growing files
# (files still being uploaded and not finished at the time we launch rsync)
# so we look for files last accessed for greater than 5min ago
find $STAGING/ACORN/radial/ -type f -amin +5 -name "*FV01_radial.nc" -printf %P\\0 | rsync -va --remove-source-files --files-from=- --from0 $STAGING/ACORN/radial/ $OPENDAP/ACORN/radial_quality_controlled/

echo ' '
date
toc=$(date +%s.%N)
printf "%6.1Fs\tFV01 radial files moved from STAGING to OPENDAP\n"  $(echo "$toc - $tic"|bc )
