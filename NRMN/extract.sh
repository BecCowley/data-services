#!/bin/bash

# exit when any command fails
set -e

# Avoid running this script if variables are undefined
set -u

export PGHOST=$IMOS_PO_CREDS_NRMN_EXTRACT_HOST
export PGUSER=$IMOS_PO_CREDS_NRMN_EXTRACT_USER
export PGPASSWORD=$IMOS_PO_CREDS_NRMN_EXTRACT_PASSWORD
export PGDATABASE=$IMOS_PO_CREDS_NRMN_EXTRACT_DATABASE

SOURCE_SCHEMA=$IMOS_PO_CREDS_NRMN_EXTRACT_SCHEMA
DESTINATION_DIR=$INCOMING_DIR/NRMN

RELATIONS_TO_EXTRACT="
  ep_site_list_public
  ep_survey_list_public
  ep_species_list_public
  ep_m1_public
  ep_m2_inverts_public
  ep_m2_cryptic_fish_public
  ep_m0_off_transect_sighting_public
  ep_m3_isq_public
  ep_m4_macrocystis_count_public
  ep_m5_limpet_quadrats_public
  ep_m11_off_transect_measurement_public
  ep_m13_pq_scores_public
  ep_species_survey
  ep_species_survey_observation
  ep_species_list
  ep_tpac
"

# main
main() {
    mkdir -p $DESTINATION_DIR

    # Don't run if previous extract has not been processed

    if [ "$(ls -A $DESTINATION_DIR)" ]; then
        echo "Unable to run extract as $DESTINATION_DIR is not empty"
        exit 1
    fi

    # Create a temporary working directory and ensure its deleted under normal circumstances

    EXTRACT_DIR=$(mktemp -dt nrmn_extract_XXXXXXXX)
    trap 'rm -rf "$EXTRACT_DIR"' EXIT

    # Extract csv for each relation to the temporary directory

    for relation in $RELATIONS_TO_EXTRACT ; do
      echo "Downloading ${relation}.csv... "
      psql -c "\copy (select * from ${SOURCE_SCHEMA}.${relation}) to ${EXTRACT_DIR}/${relation}.csv csv header"
    done

    # Zip extracted csv's

    zip -j $EXTRACT_DIR/extract.zip $EXTRACT_DIR/*.csv

    # Set permissions for pipeline to update

    chmod 0664 $EXTRACT_DIR/extract.zip

    # Move the zipped csv's to the WIP directory

    mv $EXTRACT_DIR/extract.zip $DESTINATION_DIR
}

main "$@"
