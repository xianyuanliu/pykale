import torch
from torchvision import transforms


def get_transform(kind, image_modality):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        image_modality (string): image type (RGB or Optical Flow)
    """

    if kind in ["epic", "gtea", "adl", "kitchen"]:
        transform = dict()
        if image_modality == "rgb":
            transform = {
                "train": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.RandomCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ConvertBCHWtoCBHW(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ConvertBCHWtoCBHW(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ConvertBCHWtoCBHW(),
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
                        ConvertBCHWtoCBHW(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        ConvertBCHWtoCBHW(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        ConvertBCHWtoCBHW(),
                    ]
                ),
            }
        else:
            raise ValueError("Input modality is not in [rgb, flow, joint]. Current is {}".format(image_modality))
    elif kind == "hmdb51":
        transform = {
            "train": transforms.Compose(
                [
                    ToFloatTensorInZeroOne(),
                    ConvertBHWCtoBCHW(),
                    transforms.Resize((128, 171)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                    transforms.RandomCrop((112, 112)),
                    ConvertBCHWtoCBHW(),
                    # TensorPermute2(),
                    # transforms.ConvertImageDtype(torch.float),
                    # # ImglistToTensor(),
                    # # transforms.Resize(size=256),
                    # transforms.RandomCrop(size=224),
                    # transforms.Normalize(mean=[128, 128, 128], std=[128, 128, 128]),
                    # TensorPermute(),
                ]
            ),
            "valid": transforms.Compose(
                [
                    ToFloatTensorInZeroOne(),
                    ConvertBHWCtoBCHW(),
                    transforms.Resize((128, 171)),
                    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                    transforms.CenterCrop((112, 112)),
                    ConvertBCHWtoCBHW(),
                ]
            ),
            "test": transforms.Compose(
                [
                    ToFloatTensorInZeroOne(),
                    ConvertBHWCtoBCHW(),
                    transforms.Resize((128, 171)),
                    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                    transforms.CenterCrop((112, 112)),
                    ConvertBCHWtoCBHW(),
                ]
            ),
        }
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


class ConvertBCHWtoCBHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (BATCH x CHANNELS x HEIGHT x WIDTH) to
    a torch.FloatTensor of shape (CHANNELS x BATCH x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(1, 0, 2, 3).contiguous()


class ConvertBHWCtoBCHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (BATCH x HEIGHT x WIDTH x CHANNEL) to
    a torch.FloatTensor of shape (BATCH x CHANNELS x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(0, 3, 1, 2).contiguous()


class ToFloatTensorInZeroOne(torch.nn.Module):
    """
    Convert Tensor to FloatTensor in the range [0,1].
    """

    def forward(self, tensor):
        return tensor.to(torch.float32) / 255
