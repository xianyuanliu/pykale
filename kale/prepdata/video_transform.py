import torch
from torchvision import transforms

# from torchvision.transforms import _transforms_video as transforms_video


def get_transform(kind, image_modality, method=None):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        image_modality (string): image type (RGB or Optical Flow)
        method (string, optional): the method name. Defaults to None.
    """

    mean, std = get_dataset_mean_std(kind)

    if kind in ["epic", "gtea", "adl", "kitchen"]:
        if image_modality == "rgb":
            transform = {
                "train": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.RandomCrop(size=224),
                        transforms.Normalize(mean=mean, std=std),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=mean, std=std),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=mean, std=std),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
            }
        elif image_modality == "flow":
            transform = {
                "train": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.RandomCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        ConvertTCHWtoCTHW(),
                    ]
                ),
            }
        else:
            raise ValueError("Input modality is not in [rgb, flow, joint]. Current is {}".format(image_modality))
    elif kind in ["hmdb51", "ucf101"]:
        min_size, max_size, crop_size = get_dataset_size(method)  # for hmdb51/ucf101 resizing and cropping
        transform = {
            "train": transforms.Compose(
                [
                    # ConvertBHWCtoBCHW(),
                    # ToTensorVideo(),
                    ImglistToTensor(),
                    transforms.Resize((min_size, max_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.RandomCrop((crop_size, crop_size)),
                    ConvertTCHWtoCTHW(),
                ]
            ),
            "valid": transforms.Compose(
                [
                    # ToTensorVideo(),
                    ImglistToTensor(),
                    transforms.Resize((min_size, max_size)),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.CenterCrop((crop_size, crop_size)),
                    ConvertTCHWtoCTHW(),
                ]
            ),
            "test": transforms.Compose(
                [
                    # ToTensorVideo(),
                    ImglistToTensor(),
                    transforms.Resize((min_size, max_size)),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.CenterCrop((crop_size, crop_size)),
                    ConvertTCHWtoCTHW(),
                ]
            ),
        }
    elif kind is None:
        return
    else:
        raise ValueError(f"Unknown transform kind '{kind}'")
    return transform


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``kale.loaddata.videos.VideoFrameDataset``.
    """

    def forward(self, img_list):
        """
        For RGB input, converts each PIL image in a list to a torch Tensor and stacks them into a single tensor.
        For flow input, converts every two PIL images (x(u)_img, y(v)_img) in a list to a torch Tensor and stacks them.
        For example, if input list size is 16, the dimension is [16, 1, 224, 224] and the frame order is
        [frame 1_x, frame 1_y, frame 2_x, frame 2_y, frame 3_x, ..., frame 8_x, frame 8_y]. The output will be
        [[frame 1_x, frame 1_y], [frame 2_x, frame 2_y], [frame 3_x, ..., [frame 8_x, frame 8_y]] and the dimension is
        [8, 2, 224, 224].

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size `` NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        if img_list[0].mode == "RGB":
            return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
        elif img_list[0].mode == "L":
            it = iter([transforms.functional.to_tensor(pic) for pic in img_list])
            return torch.stack([torch.cat((i, next(it)), dim=0) for i in it])
        else:
            raise RuntimeError("Image modality is not in [rgb, flow].")


class ConvertTCHWtoCTHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (TIME x CHANNELS x HEIGHT x WIDTH) to
    a torch.FloatTensor of shape (CHANNELS x TIME x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(1, 0, 2, 3).contiguous()


class ConvertTHWCtoTCHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (TIME x HEIGHT x WIDTH x CHANNEL) to
    a torch.FloatTensor of shape (TIME x CHANNELS x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(0, 3, 1, 2).contiguous()


class ToTensorVideo(torch.nn.Module):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor from (TIME x HEIGHT x WIDTH x CHANNEL).
    to (TIME x CHANNEL x HEIGHT x WIDTH).

    References: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_transforms_video.py
    """

    def forward(self, tensor):
        if not tensor.dtype == torch.uint8:
            raise TypeError("clip tensor should have data type uint8. Got %s" % str(tensor.dtype))
        return tensor.float().permute(0, 3, 1, 2) / 255.0


def get_dataset_mean_std(kind):
    """Get mean and std of a dataset for normalization."""
    if kind == "epic":
        mean = [0.412773, 0.316411, 0.278039]
        std = [0.222826, 0.215075, 0.202924]
    elif kind == "gtea":
        mean = [0.555380, 0.430436, 0.183021]
        std = [0.132028, 0.139590, 0.123337]
    elif kind == "adl":
        mean = [0.411622, 0.354001, 0.246640]
        std = [0.181746, 0.185856, 0.162441]
    elif kind == "kitchen":
        mean = [0.252758, 0.243761, 0.268163]
        std = [0.188945, 0.186148, 0.191553]
    elif kind == "ucf101":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif kind == "hmdb51":
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    elif kind is None:
        mean = std = None
    else:
        raise ValueError(f"Unknown transform for dataset '{kind}'")
    return mean, std


def get_dataset_size(method):
    """Get the size of a dataset for resizing and cropping data."""

    if method.upper() == "I3D":
        min_size = 256
        max_size = 320
        crop_size = 224
    elif method.upper() in ["C3D", "R3D_18", "R2PLUS1D_18", "MC3_18"]:
        min_size = 128
        max_size = 160
        crop_size = 112
    elif method is None:
        pass
    else:
        raise ValueError("Wrong MODEL.METHOD. Current:{}".format(method))
    return min_size, max_size, crop_size
