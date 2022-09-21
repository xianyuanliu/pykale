"""Test for action recognition example."""

import argparse
import logging

import pytorch_lightning as pl
import torch

from config import get_cfg_defaults
from model import get_model, get_train_valid_test_loaders
# from pytorch_lightning import loggers as pl_loggers

# from examples.action_recog.pytorchvideo_data import (
#     get_hmdb51_dataset_ptvideo,
#     get_train_valid_test_loaders_ptvideo,
#     get_ucf101_dataset_ptvideo,
# )

# from examples.action_recog.torchvision_data import (
#     collate_video_label,
#     get_hmdb51_dataset,
#     get_ucf101_dataset,
# )
from kale.loaddata.video_access import VideoDataset


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Action Recognition on Video Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
             "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--ckpt", default="", help="pre-trained parameters for the model (ckpt files)", type=str)
    args = parser.parse_args()
    return args


def weights_update(model, checkpoint):
    """Load the pre-trained parameters to the model."""
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


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

    if cfg.DATASET.NAME in ["ADL", "KITCHEN", "GTEA", "EPIC"]:
        dataset, num_classes = VideoDataset.get_dataset(
            VideoDataset(cfg.DATASET.NAME.upper()), cfg.MODEL.METHOD, seed, cfg
        )
        train_dataset, valid_dataset = dataset.get_train_valid(valid_ratio=cfg.DATASET.VALID_RATIO)
        test_dataset = dataset.get_test()
        _, _, test_loader = get_train_valid_test_loaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            cfg.SOLVER.TRAIN_BATCH_SIZE,
            cfg.SOLVER.TEST_BATCH_SIZE,
            cfg.SOLVER.NUM_WORKERS,
        )

    elif cfg.DATASET.NAME in ["HMDB51", "UCF101"]:
        dataset, num_classes = VideoDataset.get_dataset(
            VideoDataset(cfg.DATASET.NAME.upper()), cfg.MODEL.METHOD, seed, cfg
        )
        train_dataset = dataset.get_train()
        test_dataset = dataset.get_test()
        _, _, test_loader = get_train_valid_test_loaders(
            train_dataset,
            "None",
            test_dataset,
            cfg.SOLVER.TRAIN_BATCH_SIZE,
            cfg.SOLVER.TEST_BATCH_SIZE,
            cfg.SOLVER.NUM_WORKERS,
        )
    else:
        raise ValueError("Dataset not supported")

    # ---- setup model and logger ----
    model = get_model(cfg, num_classes)
    # tb_logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR, name="seed{}".format(cfg.SOLVER.SEED))

    ### Set the lightning trainer.
    trainer = pl.Trainer(
        logger=False,
        gpus=args.gpus,
    )

    ### Training/validation process
    model_test = weights_update(model=model, checkpoint=torch.load(args.ckpt))

    ### Evaluation
    trainer.test(model=model_test, dataloaders=test_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
