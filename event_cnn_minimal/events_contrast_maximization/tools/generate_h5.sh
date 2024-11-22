#!/bin/bash

# for filename in $PWD/event_data/ECNN_TRAIN/*.bag; do
for filename in /mnt/mhd/esim_coco/event_data/ECNN_TRAIN/bag/*_out.bag; do
    echo $filename
 	python events_contrast_maximization/tools/rosbag_to_h5.py $filename --output_dir /mnt/mhd/esim_coco/event_data/ECNN_TRAIN/h5 --event_topic /cam0/events --image_topic /cam0/image_raw
done
