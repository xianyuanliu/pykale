# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
Define the learning model and configure training parameters.
References from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
"""

from copy import deepcopy

from kale.embed.video_feature_extractor import get_video_feat_extractor
from kale.pipeline.base_trainer import ActionRecogTrainer
from kale.predict.class_domain_nets import ClassNetVideo, ClassNetVideoC3D


def get_config(cfg):
    """
    Sets the hyper parameter for the optimizer and experiment using the config file

    Args:
        cfg: A YACS config object.
    """

    config_params = {
        "train_params": {
            "init_lr": cfg.SOLVER.BASE_LR,
            "batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
            "image_modality": cfg.DATASET.IMAGE_MODALITY,
            "optimizer": {
                "type": cfg.SOLVER.TYPE,
                "optim_params": {
                    #     "momentum": cfg.SOLVER.MOMENTUM,
                    #     "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                    #     "nesterov": cfg.SOLVER.NESTEROV,
                },
            },
        },
    }
    return config_params


def get_model(cfg, num_classes):
    """
    Builds and returns a model and associated hyper parameters according to the config object passed.

    Args:
        cfg: A YACS config object.
        num_classes: The class number of specific dataset.
    """

    # setup feature extractor
    feature_network, class_feature_dim, _ = get_video_feat_extractor(
        cfg.MODEL.METHOD.upper(), cfg.DATASET.IMAGE_MODALITY, cfg.MODEL.ATTENTION, num_classes
    )
    # setup classifier
    if cfg.MODEL.METHOD.upper() == "C3D":
        classifier_network = ClassNetVideoC3D(input_size=class_feature_dim, n_class=num_classes)
    else:
        classifier_network = ClassNetVideo(input_size=class_feature_dim, n_class=num_classes)

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)

    model = ActionRecogTrainer(
        feature_extractor=feature_network,
        task_classifier=classifier_network,
        **train_params_local,
    )

    return model
