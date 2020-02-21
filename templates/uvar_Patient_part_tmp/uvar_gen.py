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
    channel_range=range(0,num_channel) # univarinat
    pat_gData,pat_gLabel = eeg_pca_ica_mat_segment_Melborne(segment_fnames[i_mat_segment],labels[i_mat_segment])
    print(pat_gData.shape)
    if pat_gData.shape[0] < 1:
       continue

############################################################
# generate uvar
#####

    if True:

        ar_poly_input={'poly_order': poly_order, 
        'time_delay': time_delay, 
        'regression_mode': regression_mode, 
        'num_neigh': Num_neigh,
        'pat_gData': pat_gData,
        'pat_gLabel': pat_gLabel, 
        'f_threshold': F_threshold,
        'n_noisy': n_noisy,
        'i_low_pass': i_low_pass,
        'noise_level': Noise_level
         }

        my_feature_all_channel, my_label_all_channel = ar_ps(ar_poly_input)
 
    if i_mat_segment == i_file_range[0]:
       my_feature = my_feature_all_channel # add samples for each file portions, train& test or blocs
       my_label =my_label_all_channel
    else:
       my_feature = np.r_[my_feature,my_feature_all_channel]  # add samples for each file portions, train& test or blocs
       my_label = np.r_[my_label,my_label_all_channel]

# replace NaN of samples
my_feature = feature_NaN(my_feature,my_feature.shape[1])
my_feature = feature_inf(my_feature,my_feature.shape[1])

# feature standalization
if i_standardization == 1:
     for i_feature in range(my_feature.shape[1]):  
      yy=my_feature[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         my_feature[:,i_feature]= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization

f = open(os.path.join(mydir,pat+'_label_feature.dat'), 'w')  # write label & feature before shuffle
np.savetxt(f, np.c_[my_label,my_feature])
f.close()

file.write(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 


