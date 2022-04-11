# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
Action video dataset loading for EPIC-Kitchen, ADL, GTEA, KITCHEN. The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

from copy import deepcopy
from enum import Enum
from pathlib import Path

import pandas as pd
import torch

import kale.prepdata.video_transform as video_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.video_datasets import BasicVideoDataset, EPIC, HMDB51_UCF101
from kale.loaddata.videos import VideoFrameDataset


def get_image_modality(image_modality):
    """Change image_modality (string) to rgb (bool), flow (bool) and audio (bool) for efficiency"""

    if image_modality.lower() == "all":
        rgb = flow = audio = True
    elif image_modality.lower() == "joint":
        rgb = flow = True
        audio = False
    elif image_modality.lower() in ["rgb", "flow", "audio"]:
        rgb = image_modality == "rgb"
        flow = image_modality == "flow"
        audio = image_modality == "audio"
    else:
        raise Exception("Invalid modality option: {}".format(image_modality))
    return rgb, flow, audio


def get_class_type(class_type):
    """Change class_type (string) to verb (bool) and noun (bool) for efficiency. Only noun is NA because we
    work on action recognition."""

    verb = True
    if class_type.lower() == "verb":
        noun = False
    elif class_type.lower() == "verb+noun":
        noun = True
    else:
        raise ValueError("Invalid class type option: {}".format(class_type))
    return verb, noun


def get_domain_adapt_config(cfg):
    """Get the configure parameters for video data for action recognition domain adaptation from the cfg files"""

    config_params = {
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_src_name": cfg.DATASET.SOURCE,
            "dataset_src_trainlist": cfg.DATASET.SRC_TRAINLIST,
            "dataset_src_testlist": cfg.DATASET.SRC_TESTLIST,
            "dataset_tgt_name": cfg.DATASET.TARGET,
            "dataset_tgt_trainlist": cfg.DATASET.TGT_TRAINLIST,
            "dataset_tgt_testlist": cfg.DATASET.TGT_TESTLIST,
            "dataset_image_modality": cfg.DATASET.IMAGE_MODALITY,
            "dataset_input_type": cfg.DATASET.INPUT_TYPE,
            "dataset_class_type": cfg.DATASET.CLASS_TYPE,
            "dataset_num_segments": cfg.DATASET.NUM_SEGMENTS,
            "frames_per_segment": cfg.DATASET.FRAMES_PER_SEGMENT,
        }
    }
    return config_params


def get_action_recog_config(cfg):
    """Get the configure parameters for video data for action recognition from the cfg files"""

    config_params = {
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_name": cfg.DATASET.NAME,
            "dataset_trainlist": cfg.DATASET.TRAINLIST,
            "dataset_testlist": cfg.DATASET.TESTLIST,
            "dataset_image_modality": cfg.DATASET.IMAGE_MODALITY,
            "dataset_num_segments": cfg.DATASET.NUM_SEGMENTS,
            "frames_per_segment": cfg.DATASET.FRAMES_PER_SEGMENT,
        },
    }
    return config_params


def generate_list(data_name, data_params_local, domain=None):
    """

    Args:
        data_name (string): name of dataset
        data_params_local (dict): hyperparameters from configure file
        domain (string, optional): domain type (source or target)

    Returns:
        data_path (string): image directory of dataset
        train_listpath (string): training list file directory of dataset
        test_listpath (string): test list file directory of dataset
    """

    if data_name == "EPIC":
        dataset_path = Path(data_params_local["dataset_root"]).joinpath(data_name, "EPIC_KITCHENS_2018")
    elif data_name in ["ADL", "GTEA", "KITCHEN", "EPIC100", "HMDB51", "UCF101"]:
        dataset_path = Path(data_params_local["dataset_root"]).joinpath(data_name)
    else:
        raise ValueError("Wrong dataset name. Select from [EPIC, ADL, GTEA, KITCHEN, EPIC100]")

    data_path = Path.joinpath(dataset_path, "frames_rgb_flow")

    if domain is None:
        train_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_trainlist"]
        )
        test_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_testlist"]
        )
    else:
        train_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_trainlist".format(domain)]
        )
        test_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_testlist".format(domain)]
        )

    return data_path, train_listpath, test_listpath


class VideoDataset(Enum):
    EPIC = "EPIC"
    ADL = "ADL"
    GTEA = "GTEA"
    KITCHEN = "KITCHEN"
    EPIC100 = "EPIC100"
    HMDB51 = "HMDB51"
    UCF101 = "UCF101"

    @staticmethod
    def get_source_target(source: "VideoDataset", target: "VideoDataset", seed, params):
        """
        Gets data loaders for source and target datasets
        Sets channel_number as 3 for RGB, 2 for flow.
        Sets class_number as 8 for EPIC, 7 for ADL, 6 for both GTEA and KITCHEN.

        Args:
            source: (VideoDataset): source dataset name
            target: (VideoDataset): target dataset name
            seed: (int): seed value set manually.
            params: (CfgNode): hyper parameters from configure file

        Examples::
            >>> source, target, num_classes = get_source_target(source, target, seed, params)
        """
        config_params = get_domain_adapt_config(params)
        data_params = config_params["data_params"]
        data_params_local = deepcopy(data_params)
        data_src_name = data_params_local["dataset_src_name"].upper()
        src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain="src")
        data_tgt_name = data_params_local["dataset_tgt_name"].upper()
        tgt_data_path, tgt_tr_listpath, tgt_te_listpath = generate_list(data_tgt_name, data_params_local, domain="tgt")
        image_modality = data_params_local["dataset_image_modality"]
        input_type = data_params_local["dataset_input_type"]
        class_type = data_params_local["dataset_class_type"]
        num_segments = data_params_local["dataset_num_segments"]
        frames_per_segment = data_params_local["frames_per_segment"]

        rgb, flow, audio = get_image_modality(image_modality)
        verb, noun = get_class_type(class_type)

        transform_names = {
            VideoDataset.EPIC: "epic",
            VideoDataset.GTEA: "gtea",
            VideoDataset.ADL: "adl",
            VideoDataset.KITCHEN: "kitchen",
            VideoDataset.EPIC100: None,
        }

        verb_class_numbers = {
            VideoDataset.EPIC: 8,
            VideoDataset.GTEA: 6,
            VideoDataset.ADL: 7,
            VideoDataset.KITCHEN: 6,
            VideoDataset.EPIC100: 97,
        }

        noun_class_numbers = {
            VideoDataset.EPIC100: 300,
        }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
            VideoDataset.GTEA: GTEADatasetAccess,
            VideoDataset.ADL: ADLDatasetAccess,
            VideoDataset.KITCHEN: KITCHENDatasetAccess,
            VideoDataset.EPIC100: EPIC100DatasetAccess,
        }

        rgb_source = rgb_target = flow_source = flow_target = audio_source = audio_target = None
        num_verb_classes = num_noun_classes = None

        if verb:
            num_verb_classes = min(verb_class_numbers[source], verb_class_numbers[target])
        if noun:
            num_noun_classes = min(noun_class_numbers[source], noun_class_numbers[target])

        source_tf = transform_names[source]
        target_tf = transform_names[target]

        if input_type == "image":

            if rgb:
                rgb_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                rgb_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )

            if flow:
                flow_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                flow_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )
            if audio:
                raise ValueError("Not support {} for input_type {}.".format(image_modality, input_type))

        elif input_type == "feature":
            # Input is feature vector, no need to use transform.
            if rgb:
                rgb_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                rgb_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )
            if flow:
                flow_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                flow_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )
            if audio:
                audio_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="audio",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                audio_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="audio",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )

        else:
            raise Exception("Invalid input type option: {}".format(input_type))

        return (
            {"rgb": rgb_source, "flow": flow_source, "audio": audio_source},
            {"rgb": rgb_target, "flow": flow_target, "audio": audio_target},
            {"verb": num_verb_classes, "noun": num_noun_classes},
        )

    @staticmethod
    def get_dataset(name: "VideoDataset", method, seed, params):

        config_params = get_action_recog_config(params)
        data_params = config_params["data_params"]
        data_params_local = deepcopy(data_params)
        data_name = data_params_local["dataset_name"].upper()
        data_path, tr_listpath, te_listpath = generate_list(data_name, data_params_local)
        image_modality = data_params_local["dataset_image_modality"]
        num_segments = data_params_local["dataset_num_segments"]
        frames_per_segment = data_params_local["frames_per_segment"]

        transform_names = {
            VideoDataset.EPIC: "epic",
            VideoDataset.GTEA: "gtea",
            VideoDataset.ADL: "adl",
            VideoDataset.KITCHEN: "kitchen",
            VideoDataset.HMDB51: "hmdb51",
            VideoDataset.UCF101: "ucf101",
        }

        class_numbers = {
            VideoDataset.EPIC: 8,
            VideoDataset.GTEA: 6,
            VideoDataset.ADL: 7,
            VideoDataset.KITCHEN: 6,
            VideoDataset.HMDB51: 51,
            VideoDataset.UCF101: 101,
        }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
            VideoDataset.GTEA: GTEADatasetAccess,
            VideoDataset.ADL: ADLDatasetAccess,
            VideoDataset.KITCHEN: KITCHENDatasetAccess,
            VideoDataset.HMDB51: HMDB51DatasetAccess,
            VideoDataset.UCF101: UCF101DatasetAccess,
        }

        dataset = factories[name](
            data_path=data_path,
            train_list=tr_listpath,
            test_list=te_listpath,
            image_modality=image_modality,
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            n_classes=class_numbers[name],
            transform=transform_names[name],
            seed=seed,
        )

        return dataset, class_numbers[name]


class VideoDatasetAccess(DatasetAccess):
    """
    Common API for video dataset access

    Args:
        data_path (string): image directory of dataset
        train_list (string): training list file directory of dataset
        test_list (string): test list file directory of dataset
        image_modality (string): image type (RGB or Optical Flow)
        num_segments (int): number of segments the video should be divided into to sample frames from.
        frames_per_segment (int): length of each action sample (the unit is number of frame)
        n_classes (int): number of class
        transform (string): types of video transforms
        seed: (int): seed value set manually.
    """

    def __init__(
        self,
        data_path,
        train_list,
        test_list,
        image_modality,
        num_segments,
        frames_per_segment,
        n_classes,
        transform,
        seed,
    ):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._image_modality = image_modality
        self._num_segments = num_segments
        self._frames_per_segment = frames_per_segment
        self._transform = video_transform.get_transform(transform, self._image_modality)
        self._seed = seed

    def get_train_valid(self, valid_ratio):
        """Get the train and validation dataset with the fixed random split. This is used for joint input like RGB and
        optical flow, which will call `get_train_valid` twice. Fixing the random seed here can keep the seeds for twice
        the same."""
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - valid_ratio) * ntotal)
        return torch.utils.data.random_split(
            train_dataset, [ntrain, ntotal - ntrain], generator=torch.Generator().manual_seed(self._seed)
        )


class EPICDatasetAccess(VideoDatasetAccess):
    """EPIC data loader"""

    def get_train(self):
        return EPIC(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=True,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return EPIC(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class GTEADatasetAccess(VideoDatasetAccess):
    """GTEA data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class ADLDatasetAccess(VideoDatasetAccess):
    """ADL data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class KITCHENDatasetAccess(VideoDatasetAccess):
    """KITCHEN data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class HMDB51DatasetAccess(VideoDatasetAccess):
    """HMDB51 data loader"""

    def get_train(self):
        return HMDB51_UCF101(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="img_{:05d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:05d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return HMDB51_UCF101(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="img_{:05d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:05d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class UCF101DatasetAccess(VideoDatasetAccess):
    """UCF101 data loader"""

    def get_train(self):
        return HMDB51_UCF101(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="img_{:05d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:05d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return HMDB51_UCF101(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="img_{:05d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:05d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class EPIC100DatasetAccess(VideoDatasetAccess):
    """EPIC-100 video feature data loader"""

    def __init__(
        self,
        domain,
        data_path,
        train_list,
        test_list,
        image_modality,
        num_segments,
        frames_per_segment,
        n_classes,
        transform,
        seed,
        input_type,
    ):
        super(EPIC100DatasetAccess, self).__init__(
            data_path,
            train_list,
            test_list,
            image_modality,
            num_segments,
            frames_per_segment,
            n_classes,
            transform,
            seed,
        )
        self._input_type = input_type
        self._domain = domain
        self._num_train_dataload = len(pd.read_pickle(self._train_list).index)
        self._num_test_dataload = len(pd.read_pickle(self._test_list).index)

    def get_train(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on train subset for EPIC UDA 2021 challenge
            # root_path=Path(self._data_path, self._input_type, "{}_train.pkl".format(self._domain)),
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 5
            frames_per_segment=self._frames_per_segment,  # 1
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=False,
            input_type="feature",
            num_data_load=self._num_train_dataload,
        )

    def get_train_valid(self, valid_ratio):
        train_dataset = self.get_train()
        valid_dataset = self.get_test()
        return train_dataset, valid_dataset

    def get_test(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on test subset for EPIC UDA 2021 challenge
            # root_path=Path(self._data_path, self._input_type, "{}_test.pkl".format(self._domain)),
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 5
            frames_per_segment=self._frames_per_segment,  # 1
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=True,
            input_type="feature",
            num_data_load=self._num_test_dataload,
        )
