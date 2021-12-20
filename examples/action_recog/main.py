"""This example is about action recognition for videos, using PyTorch Lightning."""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from torchvision.datasets import HMDB51

from config import get_cfg_defaults
from kale.prepdata.video_transform import get_transform
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from kale.loaddata.video_access import VideoDataset
from torch.utils.data import DataLoader

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
    name = cfg.DATASET.NAME
    if name == "HMDB51":
        root = cfg.DATASET.ROOT
        train_dataset = torchvision.datasets.HMDB51(
            root=root + "hmdb51_videos",
            annotation_path=root + "annotation",
            frames_per_clip=16,
            fold=1,
            train=True,
            transform=get_transform(kind="hmdb51", image_modality="rgb")["train"],
        )
        valid_dataset = torchvision.datasets.HMDB51(
            root=root + "hmdb51_videos",
            annotation_path=root + "annotation",
            frames_per_clip=16,
            fold=1,
            train=False,
            transform=get_transform(kind="hmdb51", image_modality="rgb")["valid"],
        )
        test_dataset = torchvision.datasets.HMDB51(
            root=root + "hmdb51_videos",
            annotation_path=root + "annotation",
            frames_per_clip=16,
            fold=1,
            train=False,
            transform=get_transform(kind="hmdb51", image_modality="rgb")["test"],
        )
        num_classes = 51
    else:
        dataset, num_classes = VideoDataset.get_dataset(VideoDataset(name.upper()), cfg.MODEL.METHOD, seed, cfg)
        train_dataset, valid_dataset = dataset.get_train_val(val_ratio=0.1)
        test_dataset = dataset.get_test()

    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=cfg.SOLVER.WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE, shuffle=False,
                              num_workers=cfg.SOLVER.WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=cfg.SOLVER.TEST_BATCH_SIZE, shuffle=False,
                             num_workers=cfg.SOLVER.WORKERS)

    # ---- training and evaluation ----
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i * 10
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")

        # ---- setup model and logger ----
        model = get_model(cfg, num_classes)
        tb_logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.TB_DIR, name="seed{}".format(seed))
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}",
            save_last=True,
            monitor="valid_loss",
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        ### Set the lightning trainer.
        trainer = pl.Trainer(
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            gpus=args.gpus,
            logger=tb_logger,
            callbacks=[checkpoint_callback, lr_monitor, progress_bar],
            # limit_train_batches=0.005,
            # limit_val_batches=0.1,
            # limit_test_batches=0.06,
        )

        ### Find learning_rate
        # import os
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # lr_finder = trainer.tuner.lr_find(
        #     model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=valid_loader,
        #     max_lr=1e-3,
        #     min_lr=1e-8)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # print(lr_finder.suggestion())

        ### Training/validation process
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        ### Evaluation
        trainer.test(ckpt_path="best", dataloaders=test_loader)
        trainer.test(ckpt_path=checkpoint_callback.last_model_path, dataloaders=test_loader)


if __name__ == "__main__":
    main()
