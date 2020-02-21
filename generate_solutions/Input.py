patient_index=1 
segment_length_minutes=10

mode=3

Seer_Username='hlya23dd'
subtract_mean=0

model_file_name='.hd5'
n_start=1
n_end=1


num_channel=4
Num_neigh=1  # 1 channel prediction, univariant
time_delay=3 # order of AR model
poly_order=1 # polynomial order
regression_mode = 0 # 0:   model2 = LinearRegression(fit_intercept = False)






n_noisy=1 # 1 is used to show the noise part can be modeled as well
i_low_pass=1 # surrogate
F_threshold=0.0 # [4, 8, 12, 30] frequency threshold for filtering the EEG signal, f_threshold*noise_level = 0
#noise strength for generating surrogate signal, f_threshold*noise_level = 0
Noise_level=0.0 # do not change this, only if you want to add noise
# to randomize phase while keeping power spectrum, please set f_range to negative values !!! 


####
i_PCA = -1 # no preprocessing
i_ICA = -1 # no preprocessing
i_PCA_standardization = -1 # no normalization
####
sw_output_signal_prediction = 0 # no output of predicted signals
ii_index=100000
pi_index=100000

feature_type = 'ar_new_ps'

get_feature_method = 'calculate'
get_feature_method = 'read'

data_set='kaggle2016'
Data_type='ContestRaw'
data_path='taurus'

data_portion='0' 
i_saved_model =0

pat='p'+str(patient_index)

###############################
if data_set == 'kaggle2016':
       pat_PA={'p1': '1', 'p2': '2', 'p3':'3'}
       n_ii={'1': 570, '2': 1836, '3': 1908}
       n_pi={'1': 256, '2': 222, '3': 255}
       n_test_public={'1': 61, '2': 297, '3': 206}
       n_test_private={'1': 144, '2': 697, '3': 483}

if data_set == 'kaggle2016' or data_set == 'kaggle2014' :
      PA=pat_PA[pat]   
      n_sample=n_ii[PA]+n_pi[PA]
      n_sample+=n_test_public[PA]
      n_sample+=n_test_private[PA]

feature_dir='/scratch/hyang/neuroesp/features/t3'
mvar_feature_dir='/scratch/hyang/neuroesp/features/mvar'

###############################
feature_range =['c','e','E','p','y','cE','ce','cp','cy','ey','Ey','ep','Ep','py','cpe','cpE','epy','Epy','cey','cEy','cpy','cepy','cEpy']
feature_range =['uYang']
feature_range =['e']
feature_select= 'e'

mvar_feature_lib=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr','granger']
mvar_feature_function_lib=['mpc_H(pat_gData2, lowcut, highcut, fs=400)',
'mpc_R(pat_gData2, lowcut, highcut, fs=400)',
'coherence_f(pat_gData2, lowcut, highcut, fs=400)', 
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='amp', fs=400)",
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='phase', fs=400)",
'corr_y(pat_gData2)',
'ar_granger(ar_poly_input)']
mvar_feature_range =['_mpcH_','_mpcR_','_lincoh_','_corrfAmp_','_corrfPha_','_corr_','_granger_']
mvar_feature_range =['_mvar2Yang_']
mvar_feature_range =['_mpcH_']

mvar_feature_select= mvar_feature_range[0]

feature_level = 'levelyang'
feature_level = 'a'

feature_band_range = ['bandYang']

band_pass_method='butter'   # should be used with caution, order >4 is unstable !!!
band_pass_method='square_wave'

lowcut = 0.0
highcut = 200.0
###############################

i_standarization=0
###############################
# mlp parameters


batch_size=800
epochs=500

kernel_constraint_weight=0

i_feature_selection=0

hidden_layer_size=[16,8,4]

verbose =0 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

i_class_weight = 'balanced'

