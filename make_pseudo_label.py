"""
--resume ./logs/resnet101_256/checkpoint.pth ./logs/swin_base_gemm_flip_blur/checkpoint.pth ./logs/swin_large_gemm_flip_blur/checkpoint.pth
--config ./logs/resnet101_256/config.yaml ./logs/swin_base_gemm_flip_blur/config.yaml ./logs/swin_large_gemm_flip_blur/config.yaml
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
import pandas as pd
from openunreid.core.utils.compute_dist import build_dist

from dataset import TestData, build_transform
from predict_ensamble import parge_args, setup_model


def model_inference(model, test_loader, flip=False):
    features, names = [], []
    for images1, images2, names1, names2 in tqdm(test_loader):
        if flip:
            images1, images2 = torch.flip(images1, dims=[3]), torch.flip(images2, dims=[3])
        features1 = F.normalize(model(images1.cuda()))
        features2 = F.normalize(model(images2.cuda()))
        features.append(features1.cpu())
        features.append(features2.cpu())
        names.extend(list(names1))
        names.extend(list(names2))
    return features, names


def compute_dist(model_config, features1, features2):
    similarities = torch.mm(features1, features2)
    jaccard_dist = 1 - build_dist(model_config['TEST'], features1, features2.t(), dist_m="jaccard")
    results_dist = 0.7 * similarities + 0.3 * jaccard_dist
    return 1 - results_dist


def compute_result(model, test_loader, model_config):
    features, names = model_inference(model, test_loader)
    features_f, _ = model_inference(model, test_loader, True)

    label_dict, indexes = {}, []
    for index, name in enumerate(names):
        if name not in label_dict:
            indexes.append(index)
            label_dict[name] = -1

    features = torch.cat(features)[indexes]
    features_f = torch.cat(features_f)[indexes]

    dist_1 = compute_dist(model_config, features, features.t())
    dist_2 = compute_dist(model_config, features, features_f.t())
    dist_3 = compute_dist(model_config, features_f, features.t())
    dist_4 = compute_dist(model_config, features_f, features_f.t())
    results_dist = (dist_1 + dist_2 + dist_3 + dist_4) / 4
    return results_dist, np.array(names)[indexes]


def label_data():
    batch_size = 32
    args = parge_args()
    assert len(args.configs) == len(args.resumes)

    model_list, config_list = [], []
    for config, resume in zip(args.configs, args.resumes):
        reid_model, model_cfg = setup_model(config, resume)
        model_list.append(reid_model)
        config_list.append(model_cfg)

    with torch.no_grad():
        results_dist = 0
        for model, config in zip(model_list, config_list):
            test_transform = build_transform(config['DATA']['height'])
            test_dataset = TestData('../data/pet_biometric_challenge_2022/validation/images',
                                    '../data/pet_biometric_challenge_2022/validation/valid_data.csv',
                                    test_transform, 'validation_bad_list.txt')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            model_reuslt, names = compute_result(model, test_loader, config)
            results_dist = results_dist + model_reuslt

        results_dist /= len(model_list)

    test_pairs = pd.read_csv('../data/pet_biometric_challenge_2022/validation/valid_data.csv')
    matrix_dist = np.ones_like(results_dist)

    for row in test_pairs.iterrows():
        index1 = list(names).index(row[1]['imageA'])
        index2 = list(names).index(row[1]['imageB'])
        matrix_dist[index1, index2] = results_dist[index1, index2]
        matrix_dist[index2, index1] = results_dist[index2, index1]

    matches = linear_sum_assignment(matrix_dist)

    imageAs = []
    imageBs = []
    predictions = []
    for i, (imageA_index, imageB_index) in enumerate(zip(matches[0], matches[1])):
        if matrix_dist[imageA_index, imageB_index] < 1:
            imageAs.append(imageA_index)
            imageBs.append(imageB_index)
            predictions.append(matrix_dist[imageA_index, imageB_index].item())

    thresh = np.percentile(predictions, 85)

    start_index = 6000

    test_labels = {}
    step = 0
    for match in zip(imageAs, imageBs, predictions):
        if match[2] < thresh:
            if names[match[0]] not in test_labels and names[match[1]] not in test_labels:
                test_labels[names[match[0]]] = start_index + step
                test_labels[names[match[1]]] = start_index + step
                indexes1 = np.where(matrix_dist[match[0]] < 0.5)[0]
                indexes2 = np.where(matrix_dist[match[1]] < 0.5)[0]
                for index in np.concatenate([indexes1, indexes2]):
                    if names[index] not in test_labels:
                        test_labels[names[index]] = start_index + step
                step += 1
            elif names[match[0]] in test_labels:
                test_labels[names[match[1]]] = test_labels[names[match[0]]]
                indexes = np.where(matrix_dist[match[1]] < 0.5)[0]
                for index in indexes:
                    if names[index] not in test_labels:
                        test_labels[names[index]] = test_labels[names[match[0]]]
            elif names[match[1]] in test_labels:
                test_labels[names[match[0]]] = test_labels[names[match[1]]]
                indexes = np.where(matrix_dist[match[0]] < 0.5)[0]
                for index in indexes:
                    if names[index] not in test_labels:
                        test_labels[names[index]] = test_labels[names[match[1]]]

    with open('pseudo_label.csv', 'w') as f:
        f.write('dog ID, image_name\n')
        for name in test_labels:
            f.write(f'{test_labels[name]},{name}\n')

    print('Label Image Done')


if __name__ == '__main__':
    label_data()
