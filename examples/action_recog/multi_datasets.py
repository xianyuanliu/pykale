# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Construct a dataset for videos with multiple data sources."""

import logging

import numpy as np
from sklearn.utils import check_random_state

from kale.loaddata.multi_domain import DatasetSizeType, MultiDomainDatasets, WeightingType
from kale.loaddata.sampler import FixedSeedSamplingConfig, MultiDataLoader
from kale.loaddata.video_access import get_image_modality


class VideoMultiModalDatasets(MultiDomainDatasets):
    def __init__(
        self,
        access_dict,
        image_modality,
        config_weight_type="natural",
        config_size_type="max",
        valid_split_ratio=0.1,
        num_workers=0,
        random_state=None,
        class_ids=None,
    ):
        """The class controlling how the source and target domains are iterated over when the input is joint.
            Inherited from MultiDomainDatasets.
        Args:
            access_dict (dictionary): dictionary of source RGB and flow dataset accessors
            image_modality (string): image type (RGB or Optical Flow)
            config_weight_type (WeightingType, optional): The weight type for sampling. Defaults to 'natural'.
            config_size_type (string, optional): Which dataset size to use to define the number of epochs vs
                batch_size. Defaults to "max".
            valid_split_ratio (float, optional): ratio for the validation part of the train dataset. Defaults to 0.1.
            num_workers (int, optional): number of workers for data loading. Defaults to 1.
            random_state ([int|np.random.RandomState], optional): Used for deterministic sampling/few-shot label
                selection. Defaults to None.
            class_ids (list, optional): List of chosen subset of class ids. Defaults to None (=> All Classes).
        """

        self._image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self._image_modality)

        # if self.rgb:
        #     access = access_dict["rgb"]
        # if self.flow:
        #     access = access_dict["flow"]
        # if self.audio:
        #     access = access_dict["audio"]

        # weight_type = WeightingType(config_weight_type)
        size_type = DatasetSizeType(config_size_type)
        #
        # if weight_type is WeightingType.PRESET0:
        #     self._sampling_config = FixedSeedSamplingConfig(
        #         class_weights=np.arange(access.n_classes(), 0, -1)
        #     )
        # elif weight_type is WeightingType.BALANCED:
        #     self._sampling_config = FixedSeedSamplingConfig(balance=True)
        # elif weight_type not in WeightingType:
        #     raise ValueError(f"Unknown weighting method {weight_type}.")
        # else:
        #     self._sampling_config = FixedSeedSamplingConfig(
        #         seed=random_state, num_workers=num_workers, size_type=config_size_type
        #     )
        self._sampling_config = FixedSeedSamplingConfig(
            seed=random_state, num_workers=num_workers, size_type=config_size_type
        )

        self._access_dict = access_dict
        self._valid_split_ratio = valid_split_ratio
        self._rgb_by_split = {}
        self._flow_by_split = {}
        self._audio_by_split = {}
        self._size_type = size_type
        self._random_state = check_random_state(random_state)
        self.class_ids = class_ids

    def prepare_data_loaders(self):
        if self.rgb:
            logging.debug("Load RGB train and valid")
            (self._rgb_by_split["train"], self._rgb_by_split["valid"]) = self._access_dict[
                "rgb"
            ].get_train_valid(self._valid_split_ratio)

            logging.debug("Load RGB Test")
            self._rgb_by_split["test"] = self._access_dict["rgb"].get_test()

        if self.flow:
            logging.debug("Load flow train and valid")
            (self._flow_by_split["train"], self._flow_by_split["valid"]) = self._access_dict[
                "flow"
            ].get_train_valid(self._valid_split_ratio)

            logging.debug("Load flow Test")
            self._flow_by_split["test"] = self._access_dict["flow"].get_test()

        if self.audio:
            logging.debug("Load audio train and val")
            (self._audio_by_split["train"], self._audio_by_split["valid"]) = self._access_dict[
                "audio"
            ].get_train_valid(self._valid_split_ratio)

            logging.debug("Load RGB Test")
            self._audio_by_split["test"] = self._access_dict["audio"].get_test()

    def get_multi_loaders(self, split="train", batch_size=32):
        rgb_ds = flow_ds = audio_ds = None
        rgb_loader = flow_loader = audio_loader = None

        if self.rgb:
            rgb_ds = self._rgb_by_split[split]
            rgb_loader = self._sampling_config.create_loader(rgb_ds, batch_size)
            n_dataset, _ = DatasetSizeType.get_size(self._size_type, batch_size, rgb_ds)

        if self.flow:
            flow_ds = self._flow_by_split[split]
            flow_loader = self._sampling_config.create_loader(flow_ds, batch_size)
            n_dataset, _ = DatasetSizeType.get_size(self._size_type, batch_size, flow_ds)

        if self.audio:
            audio_ds = self._audio_by_split[split]
            audio_loader = self._sampling_config.create_loader(audio_ds, batch_size)
            n_dataset, _ = DatasetSizeType.get_size(self._size_type, batch_size, audio_ds)

        dataloaders = [
            rgb_loader,
            flow_loader,
            audio_loader,
        ]
        dataloaders = [x for x in dataloaders if x is not None]

        return MultiDataLoader(dataloaders=dataloaders, n_batches=max(n_dataset // batch_size, 1))

    def __len__(self):
        if self.rgb:
            source_ds = self._rgb_by_split["train"]
        if self.flow:
            source_ds = self._flow_by_split["train"]
        if self.audio:
            source_ds = self._audio_by_split["train"]

        return len(source_ds)
