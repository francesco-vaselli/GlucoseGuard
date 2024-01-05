#!/bin/bash

# Script settings
set -eu

# Loop through directories
ls -d [0-9]* | while read dir; do

    echo "Processing directory: $dir"
    cd $dir/direct-sharing-31/
    
    # Unzip and process entries.json
    if [[ -f entries.json.gz ]]; then
        gzip -cd entries.json.gz > ${dir}_entries.json || echo "No entries.json.gz found; continuing"
    else
        echo "No entries.json.gz found; continuing"
        continue
    fi
    mkdir -p ${dir}_entries_csv

    echo 'DateString,Direction,SGV' > ${dir}_entries_csv/${dir}_entries.data.csv
    # Process entries.json
    jq -r '.[] | "\(.dateString),\(.direction),\(.sgv)"' ${dir}_entries.json >> ${dir}_entries_csv/${dir}_entries.data.csv || echo "Could not process ${dir}_entries.json; continuing"

    # Process additional entries files if present
    ls entries*.gz | sed "s/entries//" | sed "s/.json.gz//" | grep _ | while read datestring; do
        gzip -cd entries${datestring}.json.gz > ${dir}_entries${datestring}.json
        jq -r '.[] | "\(.dateString),\(.direction),\(.sgv)"' ${dir}_entries${datestring}.json >> ${dir}_entries_csv/${dir}_entries.data.csv
    done
    
    echo 'IOB,timestamp,COB,ISF, CR' > ${dir}_entries_csv/${dir}_devicestatus.data.csv
    # Unzip and process devicestatus.json
    if [[ -f devicestatus.json.gz ]]; then
        gzip -cd devicestatus.json.gz > ${dir}_devicestatus.json || echo "No devicestatus.json.gz found; continuing"
        jq -r '.[] | select(.openaps.enacted) | [(try (.openaps.enacted.IOB) // "null"), (try (.openaps.enacted.timestamp) // "null"), (try (.openaps.enacted.reason | capture("COB: (?<COB>[0-9]+)") | .COB) // "null"), (try (.openaps.enacted.reason | capture("ISF: (?<ISF>[0-9.]+)") | .ISF) // "null"), (try (.openaps.enacted.reason | capture("CR: (?<CR>[0-9.]+)") | .CR) // "null")] | @csv' ${dir}_devicestatus.json >> ${dir}_entries_csv/${dir}_devicestatus.data.csv || echo "Could not process ${dir}_devicestatus.json; continuing"
        # (try (.openaps.enacted.reason | capture("CR: (?<CR>[0-9.]+)") | .CR) // "null") not every patient has CR
    else
        echo "No devicestatus.json.gz found; continuing"
    fi

    # Process additional devicestatus files if present
    ls devicestatus*.gz | sed "s/devicestatus//" | sed "s/.json.gz//" | grep _ | while read datestring; do
        gzip -cd devicestatus${datestring}.json.gz > ${dir}_devicestatus${datestring}.json
        jq -r '.[] | select(.openaps.enacted) | [(try (.openaps.enacted.IOB) // "null"), (try (.openaps.enacted.timestamp) // "null"), (try (.openaps.enacted.reason | capture("COB: (?<COB>[0-9]+)") | .COB) // "null"), (try (.openaps.enacted.reason | capture("ISF: (?<ISF>[0-9.]+)") | .ISF) // "null"), (try (.openaps.enacted.reason | capture("CR: (?<CR>[0-9.]+)") | .CR) // "null")] | @csv' ${dir}_devicestatus.json >> ${dir}_entries_csv/${dir}_devicestatus.data.csv

    done

    # Combine entries and devicestatus data into one file
    # cat ${dir}_entries_csv/${dir}_entries.data.csv ${dir}_entries_csv/${dir}_devicestatus.data.csv > ${dir}_entries_csv/${dir}_combined.csv
    paste -d, ${dir}_entries_csv/${dir}_entries.data.csv ${dir}_entries_csv/${dir}_devicestatus.data.csv > ${dir}_entries_csv/${dir}_combined.csv


    # Sort and de-duplicate data
    sort -r ${dir}_entries_csv/${dir}_combined.csv | uniq > ${dir}_entries_csv/${dir}_entries.csv

    # Print the csv to confirm it was created
    ls ${dir}_entries_csv/${dir}_entries.csv

    # Create a copy at the top level for easier analyzing
    mkdir -p ../../EntriesCopies
    cp ${dir}_entries_csv/${dir}_combined.csv ../../EntriesCopies/

    # Cleanup and return to parent directory
    rm ${dir}_entries.json ${dir}_devicestatus.json
    cd ../../

    echo "Aggregation complete for $dir"
done

echo "All directories processed."
