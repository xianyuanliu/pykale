import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import HMDB51, UCF101

from kale.prepdata.video_transform import get_transform


def collate_video_label(batch):
    """
    The function to collate the video frames and labels.
    Original batch is [video, audio, label] but we want to collate them into [video, label]
    """

    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def get_validation_dataset(dataset, valid_ratio):
    """
    Split the dataset into training and validation dataset of valid_ratio.
    """

    num_train = len(dataset)
    num_valid = round(valid_ratio * num_train)
    train_dataset, valid_dataset = random_split(dataset, [num_train - num_valid, num_valid])
    return train_dataset, valid_dataset


def get_hmdb51_dataset(root, frame_per_segment, valid_ratio, step_between_clips=1, fold=1):
    """
    Get the HMDB51 dataset. Using torchvision.datasets.HMDB51.
    """

    train_dataset = HMDB51(
        root=root + "video/",
        annotation_path=root + "annotation_org/",
        frames_per_clip=frame_per_segment,
        step_between_clips=step_between_clips,
        fold=fold,
        train=True,
        transform=get_transform(kind="hmdb51", image_modality="rgb")["train"],
    )

    test_dataset = HMDB51(
        root=root + "video/",
        annotation_path=root + "annotation_org/",
        frames_per_clip=frame_per_segment,
        step_between_clips=step_between_clips,
        fold=fold,
        train=False,
        transform=get_transform(kind="hmdb51", image_modality="rgb")["test"],
    )
    train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    num_classes = 51
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_ucf101_dataset(root, frame_per_segment, valid_ratio, step_between_clips=1, fold=1):
    """
    Get the UCF101 dataset. Using torchvision.datasets.UCF101.
    """

    train_dataset = UCF101(
        root=root + "video/",
        annotation_path=root + "annotation/",
        frames_per_clip=frame_per_segment,
        step_between_clips=step_between_clips,
        fold=fold,
        train=True,
        transform=get_transform(kind="ucf101", image_modality="rgb")["train"],
    )

    test_dataset = UCF101(
        root=root + "video/",
        annotation_path=root + "annotation/",
        frames_per_clip=frame_per_segment,
        step_between_clips=step_between_clips,
        fold=fold,
        train=False,
        transform=get_transform(kind="ucf101", image_modality="rgb")["test"],
    )
    train_dataset, valid_dataset = get_validation_dataset(train_dataset, valid_ratio)
    num_classes = 101
    return train_dataset, valid_dataset, test_dataset, num_classes


def get_train_valid_test_loaders(
    train_dataset, valid_dataset, test_dataset, train_batch_size, test_batch_size, num_workers=0, collate_fn=None
):
    """
    Get the dataloader from the dataset. HMDB51 and UCF101 are using collate_fn but others not.
    """

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
    )
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
        )
    else:
        valid_loader = None
    return train_loader, valid_loader, test_loader
