"""This example is testing domain adaptation for action recognition, using PyTorch Lightning.
We can load and test different trained models without training.

"""

import argparse
import logging

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from thop import profile


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Action Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=[0],
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
             "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--ckpt", default="", help="pre-trained parameters for the model (ckpt files)", type=str)
    args = parser.parse_args()
    return args


def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup output ----
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    # ---- setup dataset ----
    seed = cfg.SOLVER.SEED
    source, target, dict_num_classes = VideoDataset.get_source_target(
        VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    )
    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        random_state=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
        num_workers=cfg.SOLVER.WORKERS,
    )

    model, train_params = get_model(cfg, dataset, dict_num_classes)

    input = {"rgb": torch.randn(16, 3, 16, 224, 224), }

    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


if __name__ == "__main__":
    main()
