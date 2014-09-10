#!/bin/bash

# test the number of input arguments
if [ $# -ne 3 ]
then
	echo "Usage: $0 \"{'WERA_SITE_CODE', ...}\" \"'yyymmddTHH3000'\" \"'yyymmddTHH3000'\""
	exit
fi

# Need to set the environment variables relevant for data-services jobs
source /home/ggalibert/DEFAULT_PATH.env
source /home/ggalibert/STORAGE.env

# run the job
logFile=$DATA/ACORN/radar_QC-$$.log
./radar_QC.sh "$1" "$2" "$3" &> $logFile

# check the log file size before sending the mail
maxFileSize=8000000
actualFileSize=$(wc -c "$logFile" | cut -d ' ' -f 1)
if [ $actualFileSize -ge $maxFileSize ]; then
	gzip $logFile
	echo "" | mailx -s '<ggalibert@imos-5> $ACORN_EXP/BASH/radar_QC.sh' -c sebastien.mancini@utas.edu.au guillaume.galibert@utas.edu.au -a "$logFile".gz
else
	cat $logFile | mailx -s '<ggalibert@imos-5> $ACORN_EXP/BASH/radar_QC.sh' -c sebastien.mancini@utas.edu.au guillaume.galibert@utas.edu.au
fi

# if mail is successful we can delete the logfile
rm -f $logFile "$logFile".gz
