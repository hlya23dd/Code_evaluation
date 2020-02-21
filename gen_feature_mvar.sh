#!/bin/bash

mkdir -p ./feature_generation


patient_index=1
segment_length_minutes_Yang=10
n_mvar_runs=10
n_segment_per_run=500

	for i in $(seq 0 $n_mvar_runs)
	do

           cp -r ./templates/mvar_Patient_part_tmp ./feature_generation/mvar_Patient"$patient_index"_part"$i"
           cd ./feature_generation/mvar_Patient"$patient_index"_part"$i"

           j=$((i*$n_segment_per_run))
           k=$((j+n_segment_per_run-1)) 

           sed -e "s/startYang/${j}/g" Input.py > tmp.py
           sed -e "s/endYang/${k}/g" tmp.py > tmp2.py
           sed -e "s/patYang/${patient_index}/g"  tmp2.py > tmp.py
           sed -e "s/segment_length_minutes_Yang/${segment_length_minutes_Yang}/g"  tmp.py > Input.py
           rm tmp*
           #sbatch my_job.sh

           cd ../../

	done


