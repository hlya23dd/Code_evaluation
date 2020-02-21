import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import timeit
import pandas as pd

from AR_v160_svm import *
from Input import *



############################################################
# make a directory according to time
############################################################

mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
file = open(os.path.join(mydir,'run'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.log'),'w') 
file.write(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 

if mode <= 1: # train & feature_generation
   segment_file_name_lables='../../CSVs/train_filenames_labels_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 2: # valid
   segment_file_name_lables='../../CSVs/validation+_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 3: # test
   segment_file_name_lables='../../CSVs/test_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'

df = pd.read_csv(segment_file_name_lables)
n_files=len(df)
segment_fnames=df['image']

if n_start <0:
   n_start=0
if n_end > n_files:
   n_end = n_files
i_file_range=range(n_start,n_end+1)
#i_file_range=range(n_files)

if mode == 1: # train
   labels=df['class']
else:
   labels=np.zeros(n_files)

for i_mat_segment in i_file_range:
    pat_gData,pat_gLabel = eeg_pca_ica_mat_segment_Melborne(segment_fnames[i_mat_segment],labels[i_mat_segment])
    print(pat_gData.shape)
    if pat_gData.shape[0] < 1:
       continue

############################################################
# get mvar
####
    pat_gData=pat_gData*1.0  # pat_gData seems to be integer, this cause trouble, 2018.11.28

    for feature, feature_function in zip(mvar_feature_lib, mvar_feature_function_lib):
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  print(feature)
                  pat_gData2=pat_gData.copy()*1.0

                  exec("dict_return = "+ feature_function)   
                  exec(feature+'_m' + " = dict_return['corr_y']")   
                  exec(feature+'_max' + " = dict_return['corr_max']")   
                  exec(feature+'_eva' + " = dict_return['e_values']")   
                  exec(feature+'_eve' + " = dict_return['e_vector']")   
                  exec(feature+'b' + " = np.c_["+feature+'_eva,' +feature+'_eve,' +feature+"_max]")   
                  exec(feature+'a' + " = "+feature+'_m')   
      
#####
# calculation is done
#####

############################################################
# collect and write feature 
    print('feature collection')

    if i_mat_segment == i_file_range[0]:
       my_label = pat_gLabel
       for feature in mvar_feature_lib:
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature
                  exec(my_feature_name_a + "="+feature+'a')
                  exec(my_feature_name_b + "="+feature+'b')
              
        
    else:
       my_label = np.r_[my_label,pat_gLabel]
       for feature in mvar_feature_lib:
           feature_ = '_'+feature+'_'
           if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature
                  exec(my_feature_name_a + "=np.r_[" +my_feature_name_a +',' +feature+'a'+"]")
                  exec(my_feature_name_b + "=np.r_[" +my_feature_name_b +',' +feature+'b'+"]")



# save features

for feature in mvar_feature_lib:
       feature_ = '_'+feature+'_'
       if feature_ in mvar_feature_select:
                  my_feature_name_a='featurea'+'_'+feature
                  my_feature_name_b='featureb'+'_'+feature

                  f = open(os.path.join(mydir,pat+'_'+my_feature_name_a[7:]+'_label_feature.dat'), 'w')  # write label & feature before shuffle
                  exec("np.savetxt(f, np.c_[my_label,"+my_feature_name_a+"])")
                  f.close()
                  f = open(os.path.join(mydir,pat+'_'+my_feature_name_b[7:]+'_label_feature.dat'), 'w')  # write label & feature before shuffle
                  exec("np.savetxt(f, np.c_[my_label,"+my_feature_name_b+"])")
                  f.close()

print("this is done!")

file.write(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 


