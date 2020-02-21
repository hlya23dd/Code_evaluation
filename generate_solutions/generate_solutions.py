import numpy as np


import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
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


pat='p'+str(patient_index)

if mode == 1: # train
   segment_file_name_lables='../CSVs/train_filenames_labels_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 2: # valid
   segment_file_name_lables='../CSVs/validation+_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'
if mode == 3: # test
   segment_file_name_lables='../CSVs/test_filenames_patient['+str(patient_index)+']_segment_length_['+str(segment_length_minutes)+'].csv'

df = pd.read_csv(segment_file_name_lables)
n_files=len(df)
segment_fnames=df['image']
i_file_range=range(n_files)

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

        my_feature, my_label = ar_ps(ar_poly_input)

        print(my_feature.shape)
    dim_feature_channelwise=int(my_feature.shape[1]/num_channel)
    mask=get_feature_mask(feature_select,time_delay,dim_feature_channelwise)
    mask=np.asarray(mask)
    dim_feature=mask.sum()
    my_feature = (my_feature.T[mask>0.5]).T

    if sum(mask)>0:    
     feature_flag=1
     print(my_feature.shape)
    else:
     feature_flag=0

####
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
      
                  if feature_flag==0:
                     feature_flag=1
                     if 'a' in feature_level:
                        exec("my_feature ="+feature+'a')
                     if 'b' in feature_level:
                        exec("my_feature ="+feature+'b')
                     if 'ab' in feature_level:
                        exec("my_feature =np.c_["+feature+'a,'+feature+'b]')
                  else:
                     if 'a' in feature_level:
                        exec("my_feature =np.c_[my_feature,"+feature+'a]')
                     if 'b' in feature_level:
                        exec("my_feature =np.c_[my_feature,"+feature+'b]')

    print("feature calculation is done!")

    print(my_feature.shape)

############################################################
# collect features for different segments 
    print('feature collection')

    if i_mat_segment == i_file_range[0]:
       my_label_all = pat_gLabel
       my_feature_all=my_feature              
        
    else:
       my_label_all = np.r_[my_label_all,pat_gLabel]
       my_feature_all=np.r_[my_feature_all,my_feature]

####################################################### 2018-12-5
  
# replace NaN of samples
my_feature_all = feature_NaN(my_feature_all,my_feature_all.shape[1])
my_feature_all = feature_inf(my_feature_all,my_feature_all.shape[1])
    # mean centralization & standalization

if i_standarization >0:  
    for i_feature in range(my_feature_all.shape[1]):  
      yy=my_feature_all[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         yy= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization
      my_feature_all[:,i_feature]=yy

######################################################

start = timeit.default_timer()


###
# mlp-prb
###

roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(my_feature_all, my_label_all, model_file_name)

print(roc_auc, pr_auc, roc_auc3, pr_auc3)

if not os.path.exists('../solutions'):
    os.makedirs('../solutions')

solution_fname='../solutions/solution_['+Seer_Username+']_pat['+str(patient_index)+']_seg['+str(segment_length_minutes)+']_mode['+str(mode)+']_subtract['+str(subtract_mean)+'].csv'

solutions = pd.DataFrame({'image': df['image'], 'class': probas2_10})
solutions = solutions[['image','class']]

solutions.to_csv(solution_fname,index=0)

stop = timeit.default_timer()

print(pat+' Time: ', stop - start)  


file.write(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 


