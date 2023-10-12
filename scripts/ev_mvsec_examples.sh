#!/bin/bash
pathDatasetTUM_VI='/ev_dataset/event' #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular Examples
echo "Launching Room 1 with Monocular sensor"
./Examples/Monocular/mono_tum_vi Vocabulary/ORBvoc.txt Examples/Monocular/EvMVSEC.yaml "$pathDatasetTUM_VI"/indoor_flying1_euroc/mav0/cam0/data Examples/Monocular/EvMVSEC_TimeStamps/indoor_flying1.txt dataset-indoor_flying1
