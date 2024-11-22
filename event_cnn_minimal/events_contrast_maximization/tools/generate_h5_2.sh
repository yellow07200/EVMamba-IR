#!/bin/bash

# for filename in $PWD/event_data/ECNN_TRAIN/*.bag; do
## mvsec // e2vid
# for filename in /media/kucarst3-dlws/HDD11/eFlow_avgstamps_noRNN/testing_datasets/HQF/rosbags/*.bag; do
#     echo $filename
#  	python events_contrast_maximization/tools/rosbag_to_h5.py $filename --output_dir /media/kucarst3-dlws/HDD11/eFlow_avgstamps_noRNN/testing_datasets/HQF/h5 --event_topic /cam0/events --image_topic /cam0/image_raw
# done

for filename in /media/kucarst3-dlws/HDD11/eFlow_avgstamps_noRNN/testing_datasets/ijrr/rosbags/*.bag; do
    echo $filename
 	python event_cnn_minimal/events_contrast_maximization/tools/rosbag_to_h5.py $filename --output_dir /media/kucarst3-dlws/HDD11/eFlow_avgstamps_noRNN/testing_datasets/ijrr/h5 --event_topic /dvs/events --image_topic /dvs/image_raw
done
