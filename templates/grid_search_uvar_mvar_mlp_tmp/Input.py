patient_index=1 
segment_length_minutes=10

mode=1

Seer_name='hlya23dd'



num_channel=16
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


feature_type = 'ar_new_ps'

get_feature_method = 'read'


pat='patient'+str(patient_index)


feature_dir='../../features/'
i_standardization = -1 # no normalization

###############################
feature_range =['c','e','E','p','y','cE','ce','cp','cy','ey','Ey','ep','Ep','py','cpe','cpE','epy','Epy','cey','cEy','cpy','cepy','cEpy']
feature_range =['uYang']

dim_feature_channelwise=14


mvar_feature_lib=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr','granger']
mvar_feature_range =['_mpcH_','_mpcR_','_lincoh_','_corrfAmp_','_corrfPha_','_corr_','_granger_']

mvar_feature_select= '_mvar2Yang_'

feature_level = 'levelyang'

feature_band_range = ['bandYang']
###############################
# mlp parameters


batch_size=800
epochs=500

kernel_constraint_weight=0

i_feature_selection=0

hidden_layer_size=[16,8,4]
n_model_ensemble=3

verbose =0 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

i_class_weight = 'balanced'

i_sample_weight =1
 
my_class_1_weight =100

split_time='[PATH]/UTC_AB_CD_EF.mat'
split_time=None 
n_train=None

