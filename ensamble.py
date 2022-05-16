import pandas as pd

results_1 = pd.read_csv('result_large_90.45.csv')
results_2 = pd.read_csv('result_large_90.63.csv')
results_3 = pd.read_csv('result_large_flip_90.30.csv')
results_4 = pd.read_csv('result_base_89.67.csv')
results_5 = pd.read_csv('result_no_flip_open_all_90.06.csv')
results_6 = pd.read_csv('result_open_all_89.93.csv')
results_1['prediction'] = (results_1['prediction'].values + results_2['prediction'].values +
                           results_3['prediction'].values + results_4['prediction'].values +
                           results_5['prediction'].values + results_6['prediction'].values) / 6
results_1.to_csv('result_ensamble.csv', index=False)
