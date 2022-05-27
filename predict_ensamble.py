"""
#source command line
--resume ./logs/resnet_attack/checkpoint.pth ./logs/swin_base_attack/checkpoint.pth ./logs/swin_large_attack/checkpoint.pth
--config ./logs/resnet_attack/config.yaml ./logs/swin_base_attack/config.yaml ./logs/swin_large_attack/config.yaml
--resume ./logs/swin_base_attack/checkpoint.pth ./logs/swin_large_attack/checkpoint.pth
--config ./logs/swin_base_attack/config.yaml ./logs/swin_large_attack/config.yaml
#pseudo command line
--resume ./logs/resnet_pseudo_ema/checkpoint.pth ./logs/swin_base_pseudo_ema/checkpoint.pth ./logs/swin_large_pseudo_ema/checkpoint.pth
--config ./logs/resnet_pseudo_ema/config.yaml ./logs/swin_base_pseudo_ema/config.yaml ./logs/swin_large_pseudo_ema/config.yaml
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from openunreid.models import build_model
from openunreid.utils.config import (
    cfg,
    cfg_from_yaml_file,
)
from openunreid.core.utils.compute_dist import build_dist

from dataset import TestData, build_transform


def parge_args():
    parser = argparse.ArgumentParser(description="Testing Re-ID models")
    parser.add_argument("--resumes", nargs="+", help="the checkpoint file to test")
    parser.add_argument("--configs", nargs="+", help="test config file path")
    args = parser.parse_args()
    return args


def setup_model(config_file, resume_file):
    cfg_from_yaml_file(config_file, cfg)
    reid_model = build_model(
        cfg, 0, resume_file
    )  # use num_classes=0 since we do not need classifier for testing
    reid_model.cuda()
    reid_model.eval()
    return reid_model, cfg.copy()


def model_inference(model, test_loader, flip=False):
    features_1, features_2, names_1, names_2 = [], [], [], []
    for images1, images2, names1, names2 in tqdm(test_loader):
        if flip:
            images1, images2 = torch.flip(images1, dims=[3]), torch.flip(images2, dims=[3])
        features1 = F.normalize(model(images1.cuda()))
        features2 = F.normalize(model(images2.cuda()))
        features_1.append(features1.cpu())
        features_2.append(features2.cpu())
        names_1.extend(list(names1))
        names_2.extend(list(names2))
    return features_1, features_2, names_1, names_2


def compute_dist(model_config, features1, features2):
    similarities = np.diagonal(torch.mm(torch.cat(features1), torch.cat(features2).t()))
    # jaccard_dist = 1 - build_dist(model_config['TEST'], torch.cat(features1), torch.cat(features2), dist_m="jaccard")
    # results_dist = np.diagonal(0.7 * similarities + 0.3 * jaccard_dist)
    return similarities


def compute_result(model, test_loader, model_config, flip=False):
    features_1, features_2, names_1, names_2 = model_inference(model, test_loader)
    if flip:
        features_1_f, features_2_f, _, _ = model_inference(model, test_loader, True)
    dist_1 = compute_dist(model_config, features_1, features_2)
    if flip:
        dist_2 = compute_dist(model_config, features_1_f, features_2)
        dist_3 = compute_dist(model_config, features_1, features_2_f)
        dist_4 = compute_dist(model_config, features_1_f, features_2_f)
        results_dist = (dist_1 + dist_2 + dist_3 + dist_4) / 4
    else:
        results_dist = dist_1
    return results_dist, names_1, names_2


def extract():
    batch_size = 32
    args = parge_args()
    assert len(args.configs) == len(args.resumes)

    model_list, config_list = [], []
    for config, resume in zip(args.configs, args.resumes):
        reid_model, model_cfg = setup_model(config, resume)
        model_list.append(reid_model)
        config_list.append(model_cfg)

    with torch.no_grad():
        with open('result.csv', 'w') as f:
            f.write(f'imageA,imageB,prediction\n')
            results_dist = 0
            for model, config in zip(model_list, config_list):
                test_transform = build_transform(config['DATA']['height'])
                test_dataset = TestData('../data/pet_biometric_challenge_2022/test/test',
                                        '../data/pet_biometric_challenge_2022/test/test_data.csv',
                                        test_transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
                model_reuslt, names_1, names_2 = compute_result(model, test_loader, config, flip=True)
                results_dist = results_dist + model_reuslt

            results_dist /= len(model_list)
            for image_name1, image_name2, result in zip(names_1, names_2, results_dist):
                f.write(f'{image_name1},{image_name2},{result}\n')

    print('Extraction Done')


if __name__ == '__main__':
    extract()
