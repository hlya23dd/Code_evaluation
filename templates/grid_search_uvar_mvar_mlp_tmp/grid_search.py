import numpy as np
import sys
import os, datetime
import errno
import scipy.io as sio  #for mat file


import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import timeit

from AR_v160_svm import *
from Input import *
import pandas as pd

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


for feature_select in feature_range:
  print(feature_select)

  channel_range=range(0,num_channel) # univarinat
 

###
# read features
###
  mask=get_feature_mask(feature_select,time_delay,dim_feature_channelwise)
  if sum(mask)>0:  
   feature_flag=1
   my_label_feature = np.loadtxt(os.path.join(feature_dir,pat+'_label_feature.dat'))
   my_label = my_label_feature[:,0]
   my_feature = my_label_feature[:,1:]
   print(my_label.shape,my_feature.shape)

# use mask to select univarant feature
   mask=np.asarray(mask)
   dim_feature=mask.sum()
   my_feature = (my_feature.T[mask>0.5]).T
   print(my_feature.shape)
  else:
   feature_flag=0

######################################################
# multivariant feature
####
# read features
  for feature in mvar_feature_lib:
       feature_ = '_'+feature+'_'
       if feature_ in mvar_feature_select:
                  print(feature)
                  feature_a = np.loadtxt(os.path.join(feature_dir,pat+'_a_'+feature+'_label_feature.dat'))
                  if 'b' in feature_level:
                     feature_b = np.loadtxt(os.path.join(feature_dir,pat+'_b_'+feature+'_label_feature.dat'))
                  if feature_flag==0:
                     feature_flag=1
                     if 'a' in feature_level:
                        my_feature = feature_a[:,1:]
                     if 'b' in feature_level:
                        my_feature = feature_b[:,1:]
                     if 'ab' in feature_level:
                        my_feature = np.c_[feature_a[:,1:],feature_b[:,1:]]
                  else:
                     if 'a' in feature_level:
                        my_feature = np.c_[my_feature,feature_a[:,1:]]
                     if 'b' in feature_level:
                        my_feature = np.c_[my_feature,feature_b[:,1:]]
  print("this is done!")

  print(my_feature.shape)

####################################################### 2018-12-5
  
  # replace NaN of samples
  my_feature = feature_NaN(my_feature,my_feature.shape[1])
  my_feature = feature_inf(my_feature,my_feature.shape[1])
  # mean centralization & standalization
  
  if i_standardization>0:
     for i_feature in range(my_feature.shape[1]):  
      yy=my_feature[:,i_feature]
      yy= yy-sum(yy)/len(yy)  # mean centralization
      if sum(yy*yy)>0:
         yy= yy/np.sqrt(sum(yy*yy)/len(yy))  # normalization
      my_feature[:,i_feature]=yy

######################################################

  my_feature3=my_feature.copy()  # 
  my_label3=my_label.copy()  # 

  test_feature, test_label, train_feature, train_label  = feature_train_test_split(my_feature3,my_label3, split_time, n_train)

  print(test_feature.shape, train_feature.shape)

  n_1=sum(test_label)+sum(train_label)
  n_0=len(test_label)+len(train_label)-n_1


  start = timeit.default_timer()


###
# mlp-auc
###
  print('# mean_auc_train std_auc_train mean_auc_valid std_auc_valid mean_auc_test std_auc_test C cw_1')

  auc_valid_best=0
  auc_test_best=0

  pr_auc_valid_best=0
  pr_auc_test_best=0

  feature_used='_'+feature_select+'_'+mvar_feature_select+'_'

  class_1_weight=n_ii[PA]/n_pi[PA]

  print(class_1_weight)

  rocpr=0
  for kk in range(n_model_ensemble):
      reset_keras() 
      roc_auc, roc_auc3, pr_auc, pr_auc3, model = keras_mlp_10m_prb(test_feature, test_label, train_feature, train_label, mydir, pat, class_1_weight, batch_size, epochs, kernel_constraint_weight, verbose, hidden_layer_size, i_feature_selection)
      print(roc_auc, pr_auc, roc_auc3, pr_auc3)
      if roc_auc3*pr_auc3>rocpr and roc_auc*pr_auc>0.7:
         rocpr=roc_auc3*pr_auc3
         mlp=model
      f = open(os.path.join(mydir,'auc_best_coeff_linear.dat'), 'a')
      f.write("%s %f %f %f %f\n" % (pat, roc_auc, pr_auc, roc_auc3, pr_auc3))
      f.close()

  model_file_name= pat+'_best.hd5'
  mlp.save(model_file_name)   # HDF5 file, you have to pip3 install h5py if don't have it

  roc_auc3, pr_auc3, test_label_10, probas2_10, model_file_name = keras_mlp_10m_prb_oldmodel_test(test_feature, test_label, model_file_name)

  print(roc_auc, pr_auc, roc_auc3, pr_auc3)

  stop = timeit.default_timer()

  print(pat+' Time: ', stop - start)  


file.write(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 


