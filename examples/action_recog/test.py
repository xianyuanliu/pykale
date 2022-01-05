"""Test for action recognition example."""

import argparse
import logging

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from examples.action_recog.pytorchvideo_data import (
    get_hmdb51_dataset_ptvideo,
    get_train_valid_test_loaders_ptvideo,
    get_ucf101_dataset_ptvideo,
)
from examples.action_recog.torchvision_data import (
    collate_video_label,
    get_hmdb51_dataset,
    get_train_valid_test_loaders,
    get_ucf101_dataset,
)
from kale.loaddata.video_access import VideoDataset
from kale.utils.seed import set_seed


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
    logging.basicConfig(format=format_str, level=logging.DEBUG)
    # ---- setup dataset ----
    if cfg.DATASET.NAME in ["ADL", "KITCHEN", "GTEA", "EPIC"]:
        dataset, num_classes = VideoDataset.get_dataset(
            VideoDataset(cfg.DATASET.NAME.upper()), cfg.MODEL.METHOD, cfg.SOLVER.SEED, cfg
        )
        train_dataset, valid_dataset = dataset.get_train_val(val_ratio=cfg.DATASET.VALID_RATIO)
        test_dataset = dataset.get_test()
        train_loader, valid_loader, test_loader = get_train_valid_test_loaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            cfg.SOLVER.TRAIN_BATCH_SIZE,
            cfg.SOLVER.TEST_BATCH_SIZE,
            cfg.SOLVER.NUM_WORKERS,
        )

    elif cfg.DATASET.NAME in ["HMDB51", "UCF101"]:
        if cfg.DATASET.NAME == "HMDB51":
            # train_dataset, valid_dataset, test_dataset, num_classes = get_hmdb51_dataset(
            #     cfg.DATASET.ROOT + "hmdb51/",
            #     cfg.DATASET.FRAMES_PER_SEGMENT,
            #     cfg.DATASET.VALID_RATIO,
            #     step_between_clips=16,
            #     fold=1,
            # )

            train_dataset, valid_dataset, test_dataset, num_classes = get_hmdb51_dataset_ptvideo(
                cfg.DATASET.ROOT + "hmdb51/", cfg.MODEL.METHOD, cfg.DATASET.FRAMES_PER_SEGMENT, cfg.DATASET.VALID_RATIO,
            )

        else:
            # train_dataset, valid_dataset, test_dataset, num_classes = get_ucf101_dataset(
            #     cfg.DATASET.ROOT + "ucf101/",
            #     cfg.DATASET.FRAMES_PER_SEGMENT,
            #     cfg.DATASET.VALID_RATIO,
            #     step_between_clips=16,
            #     fold=1,
            # )

            train_dataset, valid_dataset, test_dataset, num_classes = get_ucf101_dataset_ptvideo(
                cfg.DATASET.ROOT + "ucf101/", cfg.MODEL.METHOD, cfg.DATASET.FRAMES_PER_SEGMENT, cfg.DATASET.VALID_RATIO,
            )

        # train_loader, valid_loader, test_loader = get_train_valid_test_loaders(
        #     train_dataset,
        #     valid_dataset,
        #     test_dataset,
        #     cfg.SOLVER.TRAIN_BATCH_SIZE,
        #     cfg.SOLVER.TEST_BATCH_SIZE,
        #     cfg.SOLVER.NUM_WORKERS,
        #     collate_fn=collate_video_label,
        # )

        train_loader, valid_loader, test_loader = get_train_valid_test_loaders_ptvideo(
            train_dataset,
            valid_dataset,
            test_dataset,
            cfg.SOLVER.TRAIN_BATCH_SIZE,
            cfg.SOLVER.TEST_BATCH_SIZE,
            cfg.SOLVER.NUM_WORKERS,
        )

    else:
        raise ValueError("Dataset not supported")

    # ---- setup model and logger ----
    model = get_model(cfg, num_classes)
    tb_logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.TB_DIR, name="seed{}".format(cfg.SOLVER.SEED))

    ### Set the lightning trainer.
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        gpus=args.gpus,
        logger=tb_logger,
        limit_train_batches=0.0,
        limit_val_batches=0.0,
        # limit_test_batches=0.06,
    )

    ### Training/validation process
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ### Evaluation
    trainer.test(ckpt_path=args.ckpt, dataloaders=test_loader)


if __name__ == "__main__":
    main()
