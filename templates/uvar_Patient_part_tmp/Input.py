patient_index=patYang 
segment_length_minutes=segment_length_minutes_Yang

mode=0

n_start=startYang
n_end=endYang



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

i_standardization = -1 # no normalization

feature_type = 'ar_new_ps'

get_feature_method = 'calculate'

pat='patient'+str(patient_index)

