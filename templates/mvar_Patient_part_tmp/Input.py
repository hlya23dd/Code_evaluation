import numpy as np

patient_index=patYang 
segment_length_minutes=segment_length_minutes_Yang

mode=0

n_start=startYang
n_end=endYang




mvar_feature_lib=['mpcH','mpcR','lincoh','corrfAmp','corrfPha','corr','granger']
mvar_feature_function_lib=['mpc_H(pat_gData2, lowcut, highcut, fs=400)',
'mpc_R(pat_gData2, lowcut, highcut, fs=400)',
'coherence_f(pat_gData2, lowcut, highcut, fs=400)', 
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='amp', fs=400)",
"corr_f(pat_gData2, lowcut, highcut, phase_or_amp='phase', fs=400)",
'corr_y(pat_gData2)',
'ar_granger(ar_poly_input)']

feature_range =['_mpcH_mpcR_lincoh_corrfAmp_corrfPha_corr_']
mvar_feature_select= feature_range[0]


get_feature_method = 'calculate'

band_pass_method='butter'   # should be used with caution, order >4 is unstable !!!
band_pass_method='square_wave'

lowcut = 0.0
highcut = 200.0

pat='patient'+str(patient_index)

