#!/usr/bin/env sh
#
# These functions are useful for quick checking
# the status of AODN hf-radar entries in the database
# and error directory.
# I use them for cronjobs and automatic daily reporting.
#
# Assumptions:
#
# 1.configured ssh keys to servers 
# 2.configured ~/.pgpass for prod db access via psql.
# 3.configured ~/.sensitive_variables with following variables:
#    SCRATCH_DIR= #folder
#    PROD_IP_ADDR= #production ip address
#    PROD_DB_ADDR= #production db end-point
#    DBUSER= #production db user
#
# Dependencies:
#
# sh
# coreutils & friends (grep,sort,uniq,awk,sed,cut,xargs)
# psql
#
# author: hugo.oliveira@utas.edu.au
#
. "$HOME"/.sensitive_variables

get_hf_error_files() {
   help="Usage: get_hf_error_files {dm|nrt|hourly_nonqc|hourly_qc|radial_nonqc|radial_qc|all}"
   case $1 in
   "dm")
      acorn_folder='ACORN_DM'
      ;;
   "nrt")
      acorn_folder='ACORN'
      ;;
   "hourly_nonqc")
      get_hf_error_files all | grep FV00_1-hour-avg | sort | uniq
      return
      ;;
   "hourly_qc")
      get_hf_error_files all | grep FV01_1-hour-avg | sort | uniq
      return
      ;;
   "radial_nonqc")
      get_hf_error_files all | grep FV00_radial | sort | uniq
      return
      ;;
   "radial_qc")
      get_hf_error_files all | grep FV01_radial | sort | uniq
      return
      ;;
   "all")
      (get_hf_error_files dm ; get_hf_error_files nrt ) | sort | uniq
      return 
      ;;
   *)
      echo "$help" && return 1
   esac
   remote_folder="/mnt/ebs/error/$acorn_folder"
   ssh $PROD_IP_ADDR ls $remote_folder | cut -d "." -f1,2 | uniq
}

get_hf_station_from_file() {
   help="Usage: get_hf_station_from_file filename ; command | get_hf_station_from_file"
   if [ "$#" -gt 1 ]; then
      echo "$help" && return 1
   elif [ "$#" -eq 1 ]; then
      input=$1
      echo "$input" | sed -e "s/_FV[0-9][0-9]_/@/g" | cut -d "@" -f1 | sed -e "s/T[0-9].*_/@/g" | cut -d "@" -f2
   else
      while read -r input; do
         echo "$input" | sed -e "s/_FV[0-9][0-9]_/@/g" | cut -d "@" -f1 | sed -e "s/T[0-9].*_/@/g" | cut -d "@" -f2
      done
   fi
}

get_hf_time_from_file() {
   help="Usage: get_hf_time_from_file filename ; command | grep_hf_time_from_file"
   if [ "$#" -gt 1 ]; then
      echo "$help" && return 1
   elif [ "$#" -eq 1 ]; then
	 input=$1
	 date_time=$(echo "$input" | sed -e "s/Z_[A-Z]*/@/g" | cut -d "@" -f1 | rev | cut -d "_" -f1 | rev)
	 date=$(echo "$date_time" | cut -d "T" -f1 | sed -e "s/\(....\)\(..\)\(..\)/\1-\2-\3/")
	 time=$(echo "$date_time" | cut -d "T" -f2 | sed -e "s/\(..\)\(..\)\(..\)/\1:\2:\3/")
	 echo "$date" "$time"
   else
      while read -r input; do
	 date_time=$(echo "$input" | sed -e "s/Z_[A-Z]*/@/g" | cut -d "@" -f1 | rev | cut -d "_" -f1 | rev)
	 date=$(echo "$date_time" | cut -d "T" -f1 | sed -e "s/\(....\)\(..\)\(..\)/\1-\2-\3/")
	 time=$(echo "$date_time" | cut -d "T" -f2 | sed -e "s/\(..\)\(..\)\(..\)/\1:\2:\3/")
	 echo "$date" "$time"
      done
   fi
}

report_hf_files() {
   help="Usage: report_hf_files {missing|present} {hourly_avg_qc|hourly_avg_nonqc|radial_qc|radial_nonqc}"

   input_tmpfile=$(mktemp);
   result_tmpfile=$(mktemp);
   cleanup() { 
      [ -f "$input_tmpfile" ] && rm "$input_tmpfile" > /dev/null
      [ -f "$result_tmpfile" ] && rm "$result_tmpfile" > /dev/null
   }
   trap cleanup EXIT


   case $1 in
      "missing")
	 mode=$1;
      ;;
   "present")
      mode=$1;
      ;;
   *)
      echo "$help"
   esac
   shift;

   case $1 in
      "hourly_avg_qc")
	 hftype=$1;
	 get_hf_error_files hourly_qc > "$input_tmpfile"
		 ;;
      "hourly_avg_nonqc")
	 hftype=$1;
	 get_hf_error_files hourly_nonqc > "$input_tmpfile"
	 ;;
      "radial_qc")
	 hftype=$1;
	 get_hf_error_files radial_qc > "$input_tmpfile"
	 ;;
      "radial_nonqc")
	 hftype=$1;
	 get_hf_error_files radial_nonqc | sort > "$input_tmpfile"
	 ;;
      *)
      echo "$help" && return 1
   esac

   
   schema_name="acorn_$hftype";
   view_name="acorn_"$hftype"_timeseries_url";
   address="$schema_name"."$view_name"
   queryfile=$(mktemp)
   counter=0;
   while read -r file; do 
      extract_query="SELECT file_url FROM $address WHERE file_url LIKE '%$file'"
      if [ "$counter" = 0 ]; then   
	 echo "$extract_query" >> "$queryfile"
	 #query="$extract_query"
      else
	 echo "UNION $extract_query" >> "$queryfile"
	 #query="$query UNION $extract_query"
      fi
      counter=$((counter+1));
   done < "$input_tmpfile"

   echo "Missing files are in $input_tmpfile"
   echo "Query request for individual files is at $queryfile"
   # query | remove empty line of psql output | reverse string to filter prefix path out | sort | remove empty lines | use one row/line per file
#   psql -U $DBUSER -w -t -h $PROD_DB_ADDR harvest -c "$query" | xargs | rev | cut -d "/" -f 1 | rev | sort | xargs | sed -e "s/ /\n/g" > "$result_tmpfile"
   psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -f "$queryfile" | xargs | rev | cut -d "/" -f 1 | rev | sort | xargs | sed -e "s/ /\n/g" > "$result_tmpfile"
   if [ "$mode" = "missing" ]; then
      # filter differences only, print error file name, remove empty lines
      missing_files=$(diff -a -w --suppress-common-lines -y "$input_tmpfile" "$result_tmpfile" | awk '{ print $1 }' | xargs)
      if [ -z "$missing_files" ]; then
	 return 0
      else
	 echo "DBprod::$schema_name do not contain the following files:"
	 echo "$missing_files" | sed -e "s/ /\n/g"
      fi
   else
      # filter equality only, print error file name, remove empty lines
      included_files=$(diff -a -w -y "$input_tmpfile" "$result_tmpfile" | grep -v -e ">" -e "<" | awk '{ print $1 }' | xargs)
      if [ -z "$included_files" ];then
	 return 0
      else
	 echo "DBProd::$schema_name contain the following files:"
	 echo "$included_files" | sed -e "s/ /\n/g"
      fi
   fi
   cleanup 
}

report_all_error_hf_files_present_in_db() {
   (
   report_hf_files present hourly_avg_qc;
   report_hf_files present hourly_avg_nonqc;
   report_hf_files present radial_qc;
   report_hf_files present radial_nonqc;
   )
}

report_all_error_hf_files_missing_in_db() {
   (
   report_hf_files missing hourly_avg_qc;
   report_hf_files missing hourly_avg_nonqc;
   report_hf_files missing radial_qc;
   report_hf_files missing radial_nonqc;
   )
}

# Shell utilities related.
last_login() {
	last_login_iso=$(last --time-format iso "$USER" | head -n 1 | awk '{print $4}')
	last_login=$(date -d "$last_login_iso" +"%s")
	echo "$last_login"
}

last_shell_session_time() {
   #use the second most recent shell, since this fucntion is called form a session already.
   if [ "$SHELL" = '/bin/bash' ];then
      shell_pid_pattern="[0-9] $SHELL [-]" #need to match a bash login shell
   else
      shell_pid_pattern="[0-9] $SHELL";
   fi
   last_shell_started_pid=$(ps kstart_time -U "$USER" -o pid,cmd | grep "$shell_pid_pattern" | tail -n 2 | head -n 1 | awk '{print $1}')
   if [ -z "$last_shell_started_pid" ]; then
      last_login
   else
      shell_pid_start_date=$(ps --cols 1000 --rows 1000 -p "$last_shell_started_pid" -o lstart=);
      date -d "$shell_pid_start_date" +"%s"
   fi
}

need_daily_greeting() {
   if [ -e "$SCRATCH_DIR"/last_greeting ]; then
      old_date=$(cat "$SCRATCH_DIR"/last_greeting);
      current_date=$(date +"%s");
      diff_date=$(echo "$current_date" -"$old_date" | bc);
      if [ "$diff_date" -gt 21600 ]; then
	 echo "$current_date" > "$SCRATCH_DIR"/last_greeting
	 echo "yes"
      else
	 echo "no"
      fi
   else
      date +"%s" > "$SCRATCH_DIR"/last_greeting
      echo "yes"
   fi
}

report_hf_all_schemas() {
   help="Usage: report_hf_all_schemas {station|platform_code|site_code|time|file_url}"
   case $1 in
      "site_code")
	 col=$1
	 ;;
      "platform_code")
	 col=$1
	 ;;
      "station")
	 col="site_code, platform_code"
	 ;;
      "time")
	 col=$1
	 ;;
      "file_url")
	 col=$1
	 ;;
      *)
      echo "$help" && return 1
   esac

   counter=0;
   if [ "$1" = 'platform_code' ] || [ "$1" = 'station' ];then
      sources=$(hf_all_radial_sources_in_db)
   else
      sources=$(hf_all_sources_in_db)
   fi

   for source in $sources; do
      pick="select distinct $col from $source"
      if [ "$counter" -eq 0 ]; then
	 query="$pick"
      else
	 query="$query UNION $pick"
      fi
      counter=$((counter+1))
   done
   psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query" | sort | uniq
}


report_hf_time_per_site() {
   col="time"
   echo "Discovering sites..."
   sources=$(hf_all_sources_in_db)
   sites=$(hf_all_sites_in_db)
   for site in $sites; do
      for src in $sources; do
	 query="select min($col),max($col) from $src WHERE site_code LIKE '%$site%'"
         time_range=$(psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query")
	 not_avail=$(echo "$time_range" | cut -d "|" -f1)
	 if [ -n "$not_avail" ]; then
            echo "DBprod::table=$src:site=$site:time_range=$time_range"
	 fi
      done
      echo
   done
}

report_hf_time_per_station() {
   col="time";
   echo "Data reporting for HF-radar individual Stations:"
   echo "Discovering site_codes and platform_codes..."
   all_sites_and_stations=$(mktemp)
   hf_all_stations_in_db > "$all_sites_and_stations"
   echo "Discovering radial views.."
   sources=$(hf_all_radial_sources_in_db)
   echo "Filtering time ranges."
   echo
   for src in $sources; do
      prev_site=""
      while read -r site_and_stations; do
	 site_code=$(echo "$site_and_stations" | cut -d " " -f1)
	 if [ -n "$site_code" ]; then
	    if [ "$prev_site" != "$site_code" ]; then
	       echo
            fi
	    platform_code=$(echo "$site_and_stations" | cut -d " " -f2)
	    query="select min($col),max($col) from $src WHERE platform_code like '%$platform_code'"
	    time_range=$(psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query")
	    not_avail=$(echo "$time_range" | cut -d "|" -f1)
	    if [ -n "$not_avail" ]; then
	       echo "DBprod::table=$src:site_code=$site_code:station=$platform_code:time_range=$time_range"
	    fi
	    prev_site=$site_code;
	 fi
      done < "$all_sites_and_stations"
	 echo
   done
}

hf_schemas_in_db() {
    query="select schema_name from information_schema.schemata where schema_name like '%acorn%'"
    psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query" | sort | xargs
 }

hf_views_in_db() {
    query="select table_name from information_schema.views where table_name like '%acorn%'" 
    psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query" | sort | xargs
}

hf_all_sources_in_db() {
   schemas=$(hf_schemas_in_db)
   views=$(hf_views_in_db)
   counter1=0;
   counter2=0;
   for schema in $schemas;do
      counter1=$((counter1+1))
      for view in $views; do
	 counter2=$((counter2+1))
	 if [ "$counter1" = "$counter2" ]; then
	 	 echo "$schema"."$view"
	 fi
      done
      counter2=0;
   done
}

hf_all_radial_sources_in_db() {
   hf_all_sources_in_db | grep -v hourly
}

hf_all_hourly_sources_in_db() {
   hf_all_sources_in_db | grep -v radial
}

hf_all_site_code_in_db() {
   report_hf_all_schemas site_code | awk '{ print $1 }' | tr -d '“' | tr -d '”' | sed -e 's/,//g' | sort | uniq
}

hf_all_platform_code_in_db() {
   report_hf_all_schemas platform_code | tr -d '“' | tr -d '”' | sed -e "s/ /\n/g" | sort | uniq
}

hf_all_stations_in_db() {
   report_hf_all_schemas station | awk '{print $1, $NF}' | tr -d '“' | tr -d '”' |  sed -e "s/,//g" | sort | uniq
}

find_hf_file_in_db() {
   file=$(echo "$1" | cut -d "." -f1,2)
   for address in $(hf_all_sources_in_db);do
      extract_query="SELECT file_url FROM $address WHERE file_url LIKE '%$file'"
      psql -U "$DBUSER" -w -t -h "$PROD_DB_ADDR" harvest -c "$query" | rev | cut -d "/" -f 1 | rev | sort | xargs
   done
}

hf_status_greeting() {
   if [ "$1" = 'show' ]; then
      print_out='show'
   else
      print_out=$(need_daily_greeting)
   fi
   if [ "$print_out" = 'no' ]; then
		return 0
   fi
   n_dm_files=$(get_hf_error_files dm | wc -l)
   n_nrt_files=$(get_hf_error_files nrt | wc -l)
   total_files=$(echo "$n_dm_files" + "$n_nrt_files" | bc )
   echo "
=====     HF-Radar status           ======
===== $(date) ======

     . . .||. .  .
       . .||. .
     .   |  |   .
         |__|
         /||\\
        //||\\\\
       // || \\\\
    __//__||__\\\\__
   '--------------'
   $total_files erroed files"
}
