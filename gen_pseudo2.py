# %%
import pandas as pd
# %%
# df = pd.read_csv('validation/valid_data_prediction_fast_reid_moco_R101-ibn_exp0_599.csv')
df = pd.read_csv('result_ensamble_swinbase_swinlarge_resnet101-ibn_92.46.csv')
# %%
df.describe()
# %%
import numpy as np
from scipy.optimize import linear_sum_assignment
# %%
df_image_list = pd.DataFrame(np.concatenate([df['imageA'].values, df['imageB'].values]), columns=['image'])
# %%
df_image_list.drop_duplicates(inplace=True)
# %%
df_image_list.reset_index(drop=True, inplace=True)
# %%
size = len(df_image_list)
cost_matrix = np.ones((size, size))
# %%
image_index = {rec.image: rec.Index for rec in df_image_list.itertuples()}
# %%
for rec in df.itertuples():
    cost_matrix[image_index[rec.imageA], image_index[rec.imageB]] = -rec.prediction
    cost_matrix[image_index[rec.imageB], image_index[rec.imageA]] = -rec.prediction
# %%
matches = linear_sum_assignment(cost_matrix)
# %%
imageAs = []
imageBs = []
predictions = []
for i, (imageA_index, imageB_index) in enumerate(zip(matches[0], matches[1])):
    if -cost_matrix[imageA_index, imageB_index] > -1:
        imageAs.append(df_image_list.loc[imageA_index, "image"])
        imageBs.append(df_image_list.loc[imageB_index, "image"])
        predictions.append(-cost_matrix[imageA_index, imageB_index])
# %%
df_select = pd.DataFrame({'imageA': imageAs, 'imageB': imageBs, 'prediction': predictions})
# %%
df_select.describe()
# %%
df_select.to_csv('pseudo_list_hungary.csv', index=False)
# %%
df_select[(df_select['imageA'] == 'A*3G3yTa_odWgAAAAAAAAAAAAAAQAAAQ.jpg') | (df_select['imageB'] == 'A*3G3yTa_odWgAAAAAAAAAAAAAAQAAAQ.jpg')]
# %%
len(df_select.drop_duplicates('prediction'))
# %%
df_select_half = df_select.drop_duplicates('prediction')
# %%
for rec in df_select_half.itertuples():
    tmpA = df_select_half[(df_select_half['imageA'] == rec.imageA) | (df_select_half['imageB'] == rec.imageA)]
    tmpB = df_select_half[(df_select_half['imageA'] == rec.imageB) | (df_select_half['imageB'] == rec.imageB)]
    if len(tmpA) > 1 or len(tmpB) > 1:
        print(rec)
# %%
df_select_half.to_csv('pseudo_list_hungary.csv', index=False)
# %%
df_select_half.describe()
# %%
pos_threshold = np.percentile(df_select_half['prediction'].values, 15)
# %%
df_select_half_pos = df_select_half.loc[df_select_half['prediction'] > pos_threshold]
# %%
df_select_half_pos.reset_index(drop=True, inplace=True)
# %%
known = 6000
# %%
with open('pseudo_list_hungary.csv', 'w') as f:
    f.write('dog ID, image_name\n')
    for rec in df_select_half_pos.itertuples():
        f.write(f'{known + rec.Index},{rec.imageA}\n')
        f.write(f'{known + rec.Index},{rec.imageB}\n')
# %%
