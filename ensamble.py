import pandas as pd

results_1 = pd.read_csv('result_large_90.63.csv')
results_2 = pd.read_csv('result_large_flip.csv')
results_1['prediction'] = (results_1['prediction'].values + results_2['prediction'].values) / 2
results_1.to_csv('result_ensamble.csv', index=False)
