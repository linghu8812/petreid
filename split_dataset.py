import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm


data_path = '../data/pet_biometric_challenge_2022/train/images'
csv_file = '../data/pet_biometric_challenge_2022/train/train_data.csv'
bad_file = 'train_bad_list.txt'
final_path = './datasets/market1501/Market-1501-v15.09.15'
train_path = 'bounding_box_train'
query_path = 'query'
test_path = 'bounding_box_test'
val_length = 1000
query_length = 1

dataset = pd.read_csv(csv_file)
with open(bad_file, 'r') as f:
    data = f.readlines()

bad_data_list = {}
for line in data:
    name, class_id = line.strip().split(' ')
    bad_data_list[name] = int(class_id)

os.makedirs(os.path.join(final_path, train_path), exist_ok=True)
os.makedirs(os.path.join(final_path, query_path), exist_ok=True)
os.makedirs(os.path.join(final_path, test_path), exist_ok=True)

query_num = np.zeros(val_length)

for frame_id, (image_id, image_name) in tqdm(enumerate(dataset.values), total=len(dataset.values)):
    src_img = cv2.imread(os.path.join(data_path, image_name))
    if src_img is None:
        print(image_name)
        continue
    if image_name in bad_data_list:
        rotate_time = bad_data_list[image_name]
        src_img = np.rot90(src_img, rotate_time, axes=(1, 0))
    rst_name = f"{int(image_id):08d}_c1s{frame_id % 10}_{frame_id:08d}_{image_name}"
    cv2.imwrite(os.path.join(final_path, train_path, rst_name), src_img)
    if int(image_id) < val_length:
        if query_num[int(image_id)] < query_length:
            rst_name = rst_name.replace('c1', 'c2')
            cv2.imwrite(os.path.join(final_path, query_path, rst_name), src_img)
            query_num[int(image_id)] += 1
        else:
            cv2.imwrite(os.path.join(final_path, test_path, rst_name), src_img)
