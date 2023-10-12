#!/bin/bash
pathDatasetEvEthz='/ev_dataset/event' #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular Examples
echo "Launching shapes_6dof with Monocular sensor"
./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EvETHZ.yaml "$pathDatasetEvEthz"/shapes_6dof_euroc ./Examples/Monocular/EvETHZ_TimeStamps/shapes_6dof.txt dataset-shapes_6dof_mono

echo "Launching hdr_boxes with Monocular sensor"
./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EvETHZ.yaml "$pathDatasetEvEthz"/hdr_boxes_euroc ./Examples/Monocular/EvETHZ_TimeStamps/hdr_boxes.txt dataset-hdr_boxes_mono

echo "Launching slider_depth with Monocular sensor"
./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EvETHZ.yaml "$pathDatasetEvEthz"/slider_depth_euroc ./Examples/Monocular/EvETHZ_TimeStamps/slider_depth.txt dataset-slider_depth_mono

#------------------------------------
# Monocular-Inertial Examples
echo "Launching MH01 with Monocular-Inertial sensor"
./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EvETHZ.yaml "$pathDatasetEvEthz"/shapes_6dof_euroc ./Examples/Monocular/EvETHZ_TimeStamps/shapes_6dof.txt dataset-shapes_6dof_monoi


