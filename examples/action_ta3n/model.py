# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
Define the learning model and configure training parameters.
References from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
"""

from copy import deepcopy

from kale.embed.video_feature_extractor import get_extractor_feat, get_extractor_ta3n, get_extractor_video
from kale.embed.video_ta3n import get_classnet_ta3n, get_domainnet_ta3n
from kale.pipeline import domain_adapter, video_domain_adapter
from kale.predict.class_domain_nets import ClassNetVideo, DomainNetVideo


def get_config(cfg):
    """
    Sets the hyper parameter for the optimizer and experiment using the config file

    Args:
        cfg: A YACS config object.
    """

    # if cfg.DAN.METHOD == "TA3N":
    config_params = {
        "train_params": {
            # "adapt_lambda": cfg.SOLVER.AD_LAMBDA,
            # "adapt_lr": cfg.SOLVER.AD_LR,
            # "lambda_init": cfg.SOLVER.INIT_LAMBDA,
            "nb_adapt_epochs": cfg.SOLVER.MAX_EPOCHS,
            # "nb_init_epochs": cfg.SOLVER.MIN_EPOCHS,
            "init_lr": cfg.SOLVER.BASE_LR,
            "batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
            "optimizer": {"type": cfg.SOLVER.TYPE, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY},},
            # "num_source": cfg.DATASET.NUM_SOURCE,
            # "num_target": cfg.DATASET.NUM_TARGET,
            "num_segments": cfg.DATASET.NUM_SEGMENTS,
            "baseline_type": cfg.DATASET.BASELINE_TYPE,
            "frame_aggregation": cfg.DATASET.FRAME_AGGREGATION,
            "add_fc": cfg.MODEL.ADD_FC,
            "fc_dim": cfg.MODEL.FC_DIM,
            "arch": cfg.MODEL.ARCH,
            "use_target": cfg.MODEL.USE_TARGET,
            "share_params": cfg.MODEL.SHARE_PARAMS,
            "pred_normalize": cfg.MODEL.PRED_NORMALIZE,
            # "weighted_class_loss_da": cfg.MODEL.WEIGHTED_CLASS_LOSS_DA,
            # "weighted_class_loss": cfg.MODEL.WEIGHTED_CLASS_LOSS,
            "dropout_i": cfg.MODEL.DROPOUT_I,
            "dropout_v": cfg.MODEL.DROPOUT_V,
            # "no_partialbn": cfg.MODEL.NO_PARTIALBN,
            # "exp_da_name": cfg.MODEL.EXP_DA_NAME,
            # "dis_da": cfg.MODEL.DIS_DA,
            # "adv_pos_0": cfg.MODEL.ADV_POS_0,
            "adv_da": cfg.MODEL.ADV_DA,
            "add_loss_da": cfg.MODEL.ADD_LOSS_DA,
            # "ens_da": cfg.MODEL.ENS_DA,
            "use_attn": cfg.MODEL.USE_ATTN,
            "use_attn_frame": cfg.MODEL.USE_ATTN_FRAME,
            "use_bn": cfg.MODEL.USE_BN,
            "n_attn": cfg.MODEL.N_ATTN,
            # "place_dis": cfg.MODEL.PLACE_DIS,
            "place_adv": cfg.MODEL.PLACE_ADV,
            "n_rnn": cfg.MODEL.N_RNN,
            "rnn_cell": cfg.MODEL.RNN_CELL,
            "n_directions": cfg.MODEL.N_DIRECTIONS,
            "n_ts": cfg.MODEL.N_TS,
            # "flow_prefix": cfg.MODEL.FLOW_PREFIX,
            "alpha": cfg.HYPERPARAMETERS.ALPHA,
            "beta": cfg.HYPERPARAMETERS.BETA,
            "gamma": cfg.HYPERPARAMETERS.GAMMA,
            "mu": cfg.HYPERPARAMETERS.MU,
            # "pretrain_source": cfg.SOLVER.PRETRAIN_SOURCE,
            "verbose": cfg.OUTPUT.VERBOSE,
            "dann_warmup": cfg.SOLVER.DANN_WARMUP,
            # Learning configs
            # "loss_type": cfg.SOLVER.LOSS_TYPE,
            "lr_adaptive": cfg.SOLVER.LR_ADAPTIVE,
            "lr_steps": cfg.SOLVER.LR_STEPS,
            "lr_decay": cfg.SOLVER.LR_DECAY,
            # "clip_gradient": cfg.SOLVER.CLIP_GRADIENT,
            # "pretrained": cfg.SOLVER.PRETRAINED,
            # "resume": cfg.SOLVER.RESUME,
            # "resume_hp": cfg.SOLVER.RESUME_HP,
            # "accelerator": cfg.SOLVER.ACCELERATOR,
            # "workers": cfg.SOLVER.WORKERS,
            # "ef": cfg.SOLVER.EF,
            # "pf": cfg.SOLVER.PF,
            # "sf": cfg.SOLVER.SF,
            # "copy_list": cfg.SOLVER.COPY_LIST,
            # "save_model": cfg.SOLVER.SAVE_MODEL,
        },
        # "test_params": {
        #     "noun_weights": cfg.TESTER.NOUN_WEIGHTS,
        #     "batch_size": cfg.TESTER.BATCH_SIZE,
        #     "dropout_i": cfg.TESTER.DROPOUT_I,
        #     "dropout_v": cfg.TESTER.DROPOUT_V,
        #     "noun_target_data": cfg.TESTER.NOUN_TARGET_DATA,
        #     "result_json": cfg.TESTER.RESULT_JSON,
        #     "verbose": cfg.OUTPUT.VERBOSE,
        # },
    }
    data_params = {
        "data_params": {
            # "dataset_group": cfg.DATASET.NAME,
            "dataset_name": cfg.DATASET.SOURCE + "2" + cfg.DATASET.TARGET,
            "source": cfg.DATASET.SOURCE,
            "target": cfg.DATASET.TARGET,
            "size_type": cfg.DATASET.SIZE_TYPE,
            "weight_type": cfg.DATASET.WEIGHT_TYPE,
            "input_type": cfg.DATASET.INPUT_TYPE,
            "class_type": cfg.DATASET.CLASS_TYPE,
        }
    }
    config_params.update(data_params)
    if config_params["train_params"]["optimizer"]["type"] == "SGD":
        config_params["train_params"]["optimizer"]["optim_params"]["momentum"] = cfg.SOLVER.MOMENTUM
        config_params["train_params"]["optimizer"]["optim_params"]["nesterov"] = cfg.SOLVER.NESTEROV

    return config_params


# Based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
def get_model(cfg, dataset, dict_num_classes):
    """
    Builds and returns a model and associated hyper parameters according to the config object passed.

    Args:
        cfg: A YACS config object.
        dataset: A multi domain dataset consisting of source and target datasets.
        dict_num_classes (dict): The dictionary of class number for specific dataset.
    """

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    # test_params = config_params["test_params"]
    # test_params_local = deepcopy(test_params)
    data_params = config_params["data_params"]
    data_params_local = deepcopy(data_params)
    input_type = data_params_local["input_type"]
    class_type = data_params_local["class_type"]

    # setup feature extractor
    feature_network, class_feature_dim, domain_feature_dim = get_extractor_ta3n(
        cfg.DAN.METHOD.upper(),
        cfg.DATASET.IMAGE_MODALITY,
        dict_num_classes,
        cfg.DATASET.FRAME_AGGREGATION,
        cfg.DATASET.NUM_SEGMENTS,
        input_size=1024,
        output_size=512,
    )

    classifier_network = get_classnet_ta3n(
        input_size_frame=class_feature_dim,
        input_size_video=class_feature_dim,
        dict_n_class=dict_num_classes,
        dropout_rate=0.5,
    )

    # if input_type == "image":
    #     feature_network, class_feature_dim, domain_feature_dim = get_extractor_video(
    #         cfg.MODEL.METHOD.upper(), cfg.DATASET.IMAGE_MODALITY, cfg.MODEL.ATTENTION, dict_num_classes
    #     )
    # else:
    #     feature_network, class_feature_dim, domain_feature_dim = get_extractor_feat(
    #         cfg.DAN.METHOD.upper(),
    #         cfg.DATASET.IMAGE_MODALITY,
    #         dict_num_classes,
    #         cfg.DATASET.FRAME_AGGREGATION,
    #         cfg.DATASET.NUM_SEGMENTS,
    #         input_size=1024,
    #         output_size=256,
    #     )

    # setup classifier
    # if cfg.DAN.METHOD == "TA3N":
    #     classifier_network = get_classnet_ta3n(
    #         input_size_frame=class_feature_dim,
    #         input_size_video=class_feature_dim,
    #         dict_n_class=dict_num_classes,
    #         dropout_rate=0.5,
    #     )
    # else:
    #     classifier_network = ClassNetVideo(
    #         input_size=class_feature_dim, dict_n_class=dict_num_classes, class_type=class_type.lower()
    #     )

    method_params = {}

    method = domain_adapter.Method(cfg.DAN.METHOD)

    # if method.is_mmd_method():
    #     model = video_domain_adapter.create_mmd_based_video(
    #         method=method,
    #         dataset=dataset,
    #         image_modality=cfg.DATASET.IMAGE_MODALITY,
    #         feature_extractor=feature_network,
    #         task_classifier=classifier_network,
    #         input_type=input_type,
    #         class_type=class_type,
    #         **method_params,
    #         **train_params_local,
    #     )
    # elif method.is_ta3n_method():
    # TODO: add domain nets
    critic_network = get_domainnet_ta3n(input_size_frame=512, input_size_video=256)
    model = video_domain_adapter.create_dann_like_video(
        method=method,
        dataset=dataset,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        feature_extractor=feature_network,
        task_classifier=classifier_network,
        critic=critic_network,
        input_type=input_type,
        class_type=class_type,
        dict_n_class=dict_num_classes,
        **method_params,
        **train_params_local,
    )
    # else:
    #     critic_input_size = domain_feature_dim
    #     # setup critic network
    #     if method.is_cdan_method():
    #         if cfg.DAN.USERANDOM:
    #             critic_input_size = cfg.DAN.RANDOM_DIM
    #         else:
    #             critic_input_size = domain_feature_dim * dict_num_classes["verb"]
    #     critic_network = DomainNetVideo(input_size=critic_input_size)
    #
    #     if cfg.DAN.METHOD == "CDAN":
    #         method_params["use_random"] = cfg.DAN.USERANDOM
    #
    #     # The following calls kale.loaddata.dataset_access for the first time
    #     model = video_domain_adapter.create_dann_like_video(
    #         method=method,
    #         dataset=dataset,
    #         image_modality=cfg.DATASET.IMAGE_MODALITY,
    #         feature_extractor=feature_network,
    #         task_classifier=classifier_network,
    #         critic=critic_network,
    #         input_type=input_type,
    #         class_type=class_type,
    #         **method_params,
    #         **train_params_local,
    #     )

    return model, train_params
