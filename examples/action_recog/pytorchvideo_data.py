import itertools

import torch
from pytorchvideo.data import Ucf101, Hmdb51, make_clip_sampler
from pathlib import Path

from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Normalize, RandomShortSideScale, \
    ShortSideScale, RemoveKey
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop

from examples.action_recog.torchvision_data import get_validation_dataset


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def video_transform(mode):
    """
    This function contains example transforms using both PyTorchVideo and TorchVision
    in the same Callable. For 'train' mode, we use augmentations (prepended with
    'Random'), for 'val' mode we use the respective determinstic function.
    """
    if mode == "train":
        transform = Compose(
            [
                UniformTemporalSubsample(16),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(224),
                RandomHorizontalFlip(p=0.5),
                # RemoveKey("audio"),
            ]
        )
    else:
        transform = Compose(
            [
                UniformTemporalSubsample(16),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ShortSideScale(size=256),
                CenterCrop(224),
                # RemoveKey("audio"),
            ]
        )

    return ApplyTransformToKey(key="video", transform=transform)


def get_hmdb51_dataset_ptvideo(root, frame_per_segment, valid_ratio, fold=1):
    train_dataset = LimitDataset(
        Hmdb51(
            data_path=Path(root).joinpath("annotation"),
            clip_sampler=make_clip_sampler("uniform", frame_per_segment),
            decode_audio=False,
            split_id=fold,
            split_type="train",
            transform=video_transform(mode="train"),
            video_path_prefix=str(Path(root).joinpath("video")),
            video_sampler=RandomSampler,
        )
    )

    test_dataset = LimitDataset(
        Hmdb51(
            data_path=Path(root).joinpath("annotation"),
            clip_sampler=make_clip_sampler("uniform", frame_per_segment),
            decode_audio=False,
            split_id=fold,
            split_type="test",
            transform=video_transform(mode="test"),
            video_path_prefix=str(Path(root).joinpath("video")),
            video_sampler=RandomSampler,
        )
    )
    train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    num_classes = 51
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_ucf101_dataset_ptvideo(root, frame_per_segment, valid_ratio, fold=1):
    train_dataset = LimitDataset(
        Ucf101(
            data_path=str(Path(root).joinpath("annotation")),
            clip_sampler=make_clip_sampler("random", frame_per_segment),
            decode_audio=False,
            transform=video_transform(mode="train"),
            video_path_prefix=str(Path(root).joinpath("video")),
        )
    )

    test_dataset = LimitDataset(
        Ucf101(
            data_path=str(Path(root).joinpath("annotation")),
            clip_sampler=make_clip_sampler("uniform", frame_per_segment),
            decode_audio=False,
            transform=video_transform(mode="test"),
            video_path_prefix=str(Path(root).joinpath("video")),
        )
    )
    train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    num_classes = 101
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_train_valid_test_loaders_ptvideo(train_dataset, valid_dataset, test_dataset, train_batch_size, test_batch_size,
                                         num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

# if __name__ == '__main__':
#     root = Path("J:/Datasets/Video/")
#     frame_per_segment = 16
#     valid_ratio = 0.1
#
#     train_dataset, valid_dataset, test_dataset = get_hmdb51_dataset_ptvideo(root, frame_per_segment, valid_ratio)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)
