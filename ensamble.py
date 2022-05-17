import pandas as pd

results_1 = pd.read_csv('result_swin_base_gemm_90.34.csv')
results_2 = pd.read_csv('result_swin_base_gemm_flip_90.53.csv')
results_3 = pd.read_csv('result_swin_large_gemm_91.07.csv')
results_4 = pd.read_csv('result_swin_large_gemm_flip_91.18.csv')
results_1['prediction'] = (results_1['prediction'].values + results_2['prediction'].values +
                           results_3['prediction'].values + results_4['prediction'].values) / 4
results_1.to_csv('result_ensamble.csv', index=False)
