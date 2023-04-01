#!/bin/bash
# bash script to download dataset 
str="not_finished"
while [ "$str" == "$(cat status.txt)" ]
do
    timeout 1800 python3 -m src.data_extraction.downloader download_config.json
    echo "-- killing --"
done

echo "-- finished --"
