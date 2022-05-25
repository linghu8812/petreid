import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm


data_path = '../data/pet_biometric_challenge_2022/train/images'
csv_file = '../data/pet_biometric_challenge_2022/train/train_data.csv'
p_data_path = '../data/pet_biometric_challenge_2022/validation/images'
p_csv_file = 'pseudo_label.csv'
final_path = './datasets/market1501/Market-1501-v15.09.15'
train_path = 'bounding_box_train'
query_path = 'query'
test_path = 'bounding_box_test'
val_length = 1000
query_length = 1

dataset = pd.read_csv(csv_file)
p_dataset = pd.read_csv(p_csv_file)

os.makedirs(os.path.join(final_path, train_path), exist_ok=True)
os.makedirs(os.path.join(final_path, query_path), exist_ok=True)
os.makedirs(os.path.join(final_path, test_path), exist_ok=True)

query_num = np.zeros(val_length)

total_dataset = np.concatenate([dataset.values, p_dataset.values])

for frame_id, (image_id, image_name) in tqdm(enumerate(total_dataset), total=len(total_dataset)):
    src_img = cv2.imread(os.path.join(data_path, image_name))
    if src_img is None:
        src_img = cv2.imread(os.path.join(p_data_path, image_name))
        if src_img is None:
            print(image_name)
            continue
    rst_name = f"{int(image_id):08d}_c1s{frame_id % 10}_{frame_id:08d}_{image_name}"
    cv2.imwrite(os.path.join(final_path, train_path, rst_name), src_img)
    if int(image_id) < val_length:
        if query_num[int(image_id)] < query_length:
            rst_name = rst_name.replace('c1', 'c2')
            cv2.imwrite(os.path.join(final_path, query_path, rst_name), src_img)
            query_num[int(image_id)] += 1
        else:
            cv2.imwrite(os.path.join(final_path, test_path, rst_name), src_img)
