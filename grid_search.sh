#!/bin/bash
mkdir -p grid_search

flevel=(a b ab)
mvar=(mpcH mpcR lincoh corrfAmp corrfPha corr)
uvar=(c e E p y cE ce cp cy ey Ey ep Ep py cpe cpE epy Epy cey cEy cpy cepy cEpy)

for patient_index in {1..2}
do
	for i in {0..5}
	do
	    for ilevel in {0..2}
	    do
	        for j in {0..22}
	        do
                                mvari=${mvar[$i]}
                                uvari=${uvar[$j]}
                                fli=${flevel[$ilevel]}
				cp -r templates/grid_search_uvar_mvar_mlp_tmp/ grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp
				cd grid_search/patient"$patient_index"_"$uvari"_"$mvari"_"$fli"_mlp

                                sed -e "s/patYang/p${patient_index}/g"  Input.py > tmp.py
				sed -e "s/mvar2Yang/${mvari}/g"  tmp.py > tmp2.py
				sed -e "s/uYang/${uvari}/g"  tmp2.py > tmp.py
				sed -e "s/levelyang/${fli}/g"  tmp.py > Input.py
				rm tmp*
				sbatch my_job.sh
				cd ../..
	       done
           done
        done
done

