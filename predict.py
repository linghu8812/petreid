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
    cfg_from_list,
    cfg_from_yaml_file,
)
from openunreid.core.utils.compute_dist import build_dist

from dataset import TestData, build_transform


def parge_config():
    parser = argparse.ArgumentParser(description="Testing Re-ID models")
    parser.add_argument("resume", help="the checkpoint file to test")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    args.resume = Path(args.resume)
    cfg.work_dir = args.resume.parent
    if not args.config:
        args.config = cfg.work_dir / "config.yaml"
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.MODEL.sync_bn = False  # not required for inference
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def extract():
    batch_size = 32
    args, cfg = parge_config()
    reid_model = build_model(
        cfg, 0, args.resume
    )  # use num_classes=0 since we do not need classifier for testing
    reid_model.cuda()
    reid_model.eval()
    test_transform = build_transform(cfg.DATA['height'])
    test_dataset = TestData('../data/pet_biometric_challenge_2022/validation/images',
                            '../data/pet_biometric_challenge_2022/validation/valid_data.csv',
                            test_transform, 'validation_bad_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    with torch.no_grad():
        with open('result.csv', 'w') as f:
            f.write(f'imageA,imageB,prediction\n')
            features_1, features_2, names_1, names_2 = [], [], [], []
            for images1, images2, names1, names2 in tqdm(test_loader):
                features1 = F.normalize(reid_model(images1.cuda()))
                features2 = F.normalize(reid_model(images2.cuda()))
                features_1.append(features1.cpu())
                features_2.append(features2.cpu())
                names_1.extend(list(names1))
                names_2.extend(list(names2))
            similarities = torch.mm(torch.cat(features_1), torch.cat(features_2).t())
            # jaccard_dist = 1 - build_dist(cfg.TEST, torch.cat(features_1), torch.cat(features_2), dist_m="jaccard")
            # results_dist = np.diagonal(0.7 * similarities + 0.3 * jaccard_dist)
            for image_name1, image_name2, result in zip(names_1, names_2, np.diagonal(similarities)):
                f.write(f'{image_name1},{image_name2},{result}\n')

    print('Extraction Done')


if __name__ == '__main__':
    extract()
