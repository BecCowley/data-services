#!/bin/bash

export PYTHONPATH=$DATA_SERVICES_DIR/lib/python

incoming_dir() {
    cd $INCOMING_DIR/"$@"
}
_cd_incoming() {
    cd $INCOMING_DIR; _cd
}
complete -F _cd_incoming incoming_dir

error_dir() {
    cd $ERROR_DIR/"$@"
}
_cd_error() {
    cd $ERROR_DIR; _cd
}
complete -F _cd_error error_dir

declare -x -r PROJECTOFFICER_USER=`stat --printf="%U" $DATA_SERVICES_DIR/env`
alias sudo_project_officer="sudo -u $PROJECTOFFICER_USER -i"

# delete a file, using opportunistic index deletion. if file is supposed to be
# unindexed, it will, otherwise, just delete the file
po_s3_del() {
    local object_name=$1; shift

    if [ x"$object_name" = x ]; then
        echo "Usage: po_s3_del object_name" 1>&2; return 1
    fi

    if can_be_indexed $object_name; then
        echo "Deleting '$object_name' with index deletion"
        _po_command s3_del $object_name
    else
        echo "Deleting '$object_name'"
        _po_command s3_del_no_index $object_name
    fi
}

# run command in manual mode
# "$@" command to run
_po_command() {
    local tmp_command_file=`mktemp`
    local _user=`whoami`

    # build a file with the specified command, we'll run it with sudo
    echo "#!/bin/bash"                             >> $tmp_command_file
    echo "source $DATA_SERVICES_DIR/env"           >> $tmp_command_file
    echo "export JOB_NAME='MANUAL_COMMAND_'$_user" >> $tmp_command_file
    echo "log_info \"Running: '$@'\""              >> $tmp_command_file
    echo "$@"                                      >> $tmp_command_file

    chmod 755 $tmp_command_file
    sudo -u $PROJECTOFFICER_USER $tmp_command_file
    local -i retval=$?

    rm -f $tmp_command_file

    return $retval
}
