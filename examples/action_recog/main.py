"""This example is about action recognition for videos, using PyTorch Lightning."""

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
    # seed = cfg.SOLVER.SEED
    # set_seed(seed)
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

        train_loader, _, test_loader = get_train_valid_test_loaders_ptvideo(
            train_dataset,
            valid_dataset,
            test_dataset,
            cfg.SOLVER.TRAIN_BATCH_SIZE,
            cfg.SOLVER.TEST_BATCH_SIZE,
            cfg.SOLVER.NUM_WORKERS,
        )

    else:
        raise ValueError("Dataset not supported")

    print(f"Train samples: {len(train_dataset)}")
    # print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    # print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # ---- training and evaluation ----
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i * 10
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")

        # ---- setup model and logger ----
        model = get_model(cfg, num_classes)
        if cfg.COMET.ENABLE:
            logger = pl_loggers.CometLogger(api_key=cfg.COMET.API_KEY, project_name=cfg.COMET.PROJECT_NAME, save_dir=cfg.OUTPUT.TB_DIR)
        else:
            logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.TB_DIR, name="seed{}".format(seed))
        # checkpoint_callback = ModelCheckpoint(
        # filename="{epoch}-{step}-{val_loss:.4f}", save_last=True, monitor="valid_loss", mode="min",
        # )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        ### Set the lightning trainer.
        trainer = pl.Trainer(
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            gpus=args.gpus,
            logger=logger,
            # callbacks=[checkpoint_callback, lr_monitor, progress_bar],
            callbacks=[lr_monitor, progress_bar],
            # limit_train_batches=0.005,
            # limit_val_batches=0.01,
            # limit_test_batches=0.001,
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
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        # trainer.fit(model, train_dataloaders=train_loader)

        ### Evaluation
        # trainer.test(ckpt_path="best", dataloaders=test_loader)
        # trainer.test(ckpt_path=checkpoint_callback.last_model_path, dataloaders=test_loader)
        trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()
