import itertools
from pathlib import Path

import torch
from pytorchvideo.data import Hmdb51, labeled_video_dataset, LabeledVideoDataset, make_clip_sampler, Ucf101
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomCrop, RandomHorizontalFlip

from examples.action_recog.torchvision_data import get_validation_dataset


def hmdb51_with_ucf101_list(
    data_path,
    clip_sampler,
    video_sampler=torch.utils.data.RandomSampler,
    transform=None,
    video_path_prefix="",
    decode_audio=True,
    decoder="pyav",
) -> LabeledVideoDataset:
    return labeled_video_dataset(
        data_path, clip_sampler, video_sampler, transform, video_path_prefix, decode_audio, decoder,
    )


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset, clips_per_video=1.0):
        super().__init__()
        self.dataset = dataset
        self.clips_per_video = clips_per_video

        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return int(self.dataset.num_videos * self.clips_per_video)


def video_transform(mode, dataset, method, frame_per_segment=16):
    """
    This function contains example transforms using both PyTorchVideo and TorchVision
    in the same Callable. For 'train' mode, we use augmentations (prepended with
    'Random'), for 'val' mode we use the respective determinstic function.
    """
    if dataset == "ucf101":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "hmdb51":
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError("Wrong DATASET.NAME. Current:{}".format(dataset))

    if method.upper() == "I3D":
        min_size = 256
        max_size = 320
        crop_size = 224
    elif method.upper() in ["C3D", "R3D_18", "R2PLUS1D_18", "MC3_18"]:
        min_size = 128
        max_size = 160
        crop_size = 112
    else:
        raise ValueError("Wrong MODEL.METHOD. Current:{}".format(method))

    if mode == "train":
        transform = Compose(
            [
                UniformTemporalSubsample(frame_per_segment),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                RandomShortSideScale(min_size=min_size, max_size=max_size),
                RandomCrop(crop_size),
                RandomHorizontalFlip(p=0.5),
                # RemoveKey("audio"),
            ]
        )
    else:
        transform = Compose(
            [
                UniformTemporalSubsample(frame_per_segment),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                ShortSideScale(size=min_size),
                CenterCrop(crop_size),
                # ShortSideScale(size=128),
                # CenterCrop(112),
                # RemoveKey("audio"),
            ]
        )

    return ApplyTransformToKey(key="video", transform=transform)


def get_hmdb51_dataset_ptvideo(root, method, frame_per_segment, valid_ratio, fold=1):
    second_per_segment = frame_per_segment / 30
    train_dataset = LimitDataset(
        # Hmdb51(
        #     data_path=Path(root).joinpath("annotation_org"),
        hmdb51_with_ucf101_list(
            # data_path=Path(root).joinpath("annotation", "dummy_trainlist0{}.txt".format(fold)),
            data_path=Path(root).joinpath("annotations", "trainlist0{}.txt".format(fold)),
            # clip_sampler=make_clip_sampler("constant_clips_per_video", frame_per_segment, 5),
            clip_sampler=make_clip_sampler("random", second_per_segment),
            decode_audio=False,
            transform=video_transform("train", "hmdb51", method, frame_per_segment),
            video_path_prefix=str(Path(root).joinpath("video")),
        ),
        # clips_per_video=5,
    )

    test_dataset = LimitDataset(
        # Hmdb51(
        #     data_path=Path(root).joinpath("annotation_org"),
        hmdb51_with_ucf101_list(
            # data_path=Path(root).joinpath("annotation", "dummy_trainlist0{}.txt".format(fold)),
            data_path=Path(root).joinpath("annotations", "testlist0{}.txt".format(fold)),
            # clip_sampler=make_clip_sampler("constant_clips_per_video", frame_per_segment, 5),
            # clip_sampler=make_clip_sampler("random", frame_per_segment),
            clip_sampler=make_clip_sampler("uniform", second_per_segment),
            decode_audio=False,
            transform=video_transform("test", "hmdb51", method, frame_per_segment),
            video_path_prefix=str(Path(root).joinpath("video")),
        ),
        clips_per_video=5.61,
    )
    # train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    valid_dataset = None
    num_classes = 51
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_ucf101_dataset_ptvideo(root, method, frame_per_segment, valid_ratio, fold=1):
    second_per_segment = frame_per_segment / 30
    train_dataset = LimitDataset(
        Ucf101(
            data_path=str(Path(root).joinpath("annotations", "trainlist0{}.txt".format(fold))),
            # clip_sampler=make_clip_sampler("constant_clips_per_video", frame_per_segment, 5),
            clip_sampler=make_clip_sampler("random", second_per_segment),
            decode_audio=False,
            transform=video_transform("train", "ucf101", method, frame_per_segment),
            video_path_prefix=str(Path(root).joinpath("video")),
        ),
        # clips_per_video=5,
    )

    test_dataset = LimitDataset(
        Ucf101(
            data_path=str(Path(root).joinpath("annotations", "testlist0{}.txt".format(fold))),
            # clip_sampler=make_clip_sampler("constant_clips_per_video", frame_per_segment, 5),
            # clip_sampler=make_clip_sampler("random", frame_per_segment),
            clip_sampler=make_clip_sampler("uniform", second_per_segment),
            decode_audio=False,
            transform=video_transform("test", "ucf101", method, frame_per_segment),
            video_path_prefix=str(Path(root).joinpath("video")),
        ),
        clips_per_video=11.05,
    )
    # train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    valid_dataset = None
    num_classes = 101
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_train_valid_test_loaders_ptvideo(
    train_dataset, valid_dataset, test_dataset, train_batch_size, test_batch_size, num_workers=0
):
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers)
    # valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, num_workers=num_workers)
    valid_loader = None
    return train_loader, valid_loader, test_loader


# if __name__ == '__main__':
#     root = Path("J:/Datasets/Video/")
#     frame_per_segment = 16
#     valid_ratio = 0.1
#
#     # train_dataset, valid_dataset, test_dataset = get_hmdb51_dataset_ptvideo(root, frame_per_segment, valid_ratio)
#     # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4)
#     # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)
#
#     train_dataset, valid_dataset, test_dataset, num_classes = get_ucf101_dataset_ptvideo(str(root.joinpath("ucf101")), frame_per_segment, valid_ratio)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4)
#     valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)
#
#     print(f"Total number of train samples: {len(train_dataset)}")
#     print(f"Total number of validation samples: {len(valid_dataset)}")
#     print(f"Total number of test samples: {len(test_dataset)}")
#     print(f"Total number of train batches: {len(train_loader)}")
#     print(f"Total number of validation batches: {len(valid_loader)}")
#     print(f"Total number of test batches: {len(test_loader)}")
