#!/bin/bash
# bash script to download dataset 
str="not_finished"
while [ "$str" == "$(cat status.txt)" ]
do
    timeout 3600 python3 -m src.data_extraction.downloader download_config.json
    echo "-- killing --"
done

echo "-- finished --"

# python -m src.data_extraction.downloader download_config.json
