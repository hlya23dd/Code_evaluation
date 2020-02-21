#!/bin/bash
mkdir -p ./features


patient_index=1
n_uvar_runs=10


for i in $(seq 0 $n_uvar_runs)
do

           cat ./feature_generation/uvar_Patient"$patient_index"_part"$i"/2020*/patient"$patient_index"_label_feature.dat >> ./features/patient"$patient_index"_label_feature.dat

done


