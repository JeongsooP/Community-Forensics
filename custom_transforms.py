import math
from typing import List, Tuple, Optional, Dict
import os

import torch
from torch import Tensor
import numpy as np
import PIL.Image
import random
from io import BytesIO
import cv2
import numpy as np

from torchvision.transforms import functional as F, InterpolationMode
import torchvision.transforms as T

__all__ = ["AutoAugmentPolicy", "AutoAugment", "RandAugment", "TrivialAugmentWide", "AugMix"]

def get_dimensions(img):
    height, width = F.get_image_size(img)
    channels = F.get_image_num_channels(img)
    return channels, height, width

def cutout(img, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image. 
    
    ### (PyTorch implementation of Google's big_vision cutout) ###
    
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    Args:
        image: A PIL image
        pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.
    Returns:
        A PIL image of type uint8.
    """
    convert_back=False
    if F._is_pil_image(img):
        img = F.pil_to_tensor(img) # convert to tensor for pytorch operations
        convert_back=True
    assert img.dtype == torch.uint8, "PIL to tensor image is expected to have torch.unit8 as dtype."
    channels, height, width = get_dimensions(img)
    cutout_center_height = torch.randint(low=0, high=height, size=(1,)).item()
    cutout_center_width = torch.randint(low=0, high=width, size=(1,)).item()

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, height - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, width - cutout_center_width - pad_size)

    cutout_shape = (height - (lower_pad + upper_pad),
                    width - (left_pad + right_pad)) # cutout this shape
    padding_dims = (left_pad, right_pad, upper_pad, lower_pad)
    cutout_mask = torch.nn.functional.pad(
        torch.zeros(cutout_shape, dtype=img.dtype, device=img.device),
        padding_dims, value=1
    )
    cutout_mask = cutout_mask.unsqueeze(dim=0)
    cutout_mask = torch.tile(cutout_mask, (channels,1,1))
    #replacement = torch.ones_like(img, dtype=torch.float32) * replace[0]
    #replacement = replacement.to(torch.uint8)
    img = torch.where(
        cutout_mask==0, # condition.
        torch.ones_like(img, dtype=img.dtype, device=img.device) * replace, # If true
        #replacement,
        img # If condition is false
    )
    if convert_back:
        return F.to_pil_image(img)
    else:
        return img

def solarize_add(img, addition=0, threshold=128):
    """
    For each pixel in the image less than threshold
    we add 'addition' amount to it and then clip the
    pixel value to be between 0 and 255. The value
    of 'addition' is between -128 and 128.
    
    ### Re-implementation of Google's big_vision in PyTorch ###
    """
    convert_back=False
    if F._is_pil_image(img):
        img = F.pil_to_tensor(img) # convert to tensor for pytorch operations
        convert_back=True
    assert img.dtype == torch.uint8, "PIL to tensor image is expected to have torch.unit8 as dtype."
    added_img = img.to(torch.int) + addition
    added_img = torch.clamp(added_img, min=0,max=255)
    added_img = added_img.to(img.dtype)
    img = torch.where(
        img < threshold, # condition
        added_img, # if true
        img # if false
    )
    if convert_back:
        return F.to_pil_image(img)
    else:
        return img

def chroma_drop(img):
    img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    if torch.rand(1).item() > 0.5:
        Cr = Cr.point(lambda i: 128)
    else:
        Cb = Cb.point(lambda i: 128)
    img = PIL.Image.merge("YCbCr", (Y, Cb, Cr))
    return img.convert("RGB")

def auto_saturation_separate(img):
    img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    Cbmin, Cbmax = Cb.getextrema()
    Crmin, Crmax = Cr.getextrema()
    Cmin = min(Cbmin, Crmin)
    Cmax = max(Cbmax, Crmax)
    Cb = Cb.point(lambda i: ((i-128) / (Cmax - 128) * 127 + 128 if Cmax > 128 else i) if i>127 \
        else ((i - Cmin) / (127 - Cmin) * 127) if Cmin<127 else i) # scale >127 and else separately (they represent different hue)
    #Cb = Cb.point(lambda i: (i-Cbmin) / (Cbmax - Cbmin) * 255)
    Cr = Cr.point(lambda i: ((i-128) / (Cmax - 128) * 127 + 128 if Cmax > 128 else i) if i>127 \
        else ((i - Cmin) / (127 - Cmin) * 127) if Cmin<127 else i)
    #Cr = Cr.point(lambda i: (i-Crmin) / (Crmax - Crmin) * 255)
    img = PIL.Image.merge("YCbCr", (Y, Cb, Cr))
    return img.convert("RGB")


def auto_saturation(img):
    img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    Cbmin, Cbmax = Cb.getextrema()
    Crmin, Crmax = Cr.getextrema()
    Cmin = min(Cbmin, Crmin)
    Cmax = max(Cbmax, Crmax)
    Cb = Cb.point(lambda i: (i-Cmin) / (Cmax - Cmin) * 255 if (Cmax - Cmin) != 0 else i)
    Cr = Cr.point(lambda i: (i-Cmin) / (Cmax - Cmin) * 255 if (Cmax - Cmin) != 0 else i)
    img = PIL.Image.merge("YCbCr", (Y, Cb, Cr))
    return img.convert("RGB")

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    elif op_name == 'Cutout': # added
        img = cutout(img, int(magnitude), replace=fill)
    elif op_name == "SolarizeAdd": # added
        img = solarize_add(img, int(magnitude))
    elif op_name == "Grayscale": # added v2
        img = F.to_grayscale(img, num_output_channels=3)
    elif op_name == "ChromaDrop": #
        img = chroma_drop(img)
    elif op_name == "AutoSaturation":
        #img = auto_saturation(img)
        img = auto_saturation(img) # dct-equivalent
    elif op_name == "AutoSaturation_old": # for compatibility purposes
        img = auto_saturation(img)
    elif op_name == "Rotate90": # magnitude is +- 90
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



class RandAugment_bv(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    ### Re-implementation of Google's Big Vision randaugment in PyTorch ###

    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 10,
        num_magnitude_bins: int = 11,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        ops_list = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd", "Color", "Contrast", "Brightness",
                        "Sharpness", "ShearX", "ShearY", "Cutout", "TranslateX", "TranslateY"]
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        if ops_list==None:
            self.ops_list = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd", "Color", "Contrast", "Brightness",
                        "Sharpness", "ShearX", "ShearY", "Cutout", "TranslateX", "TranslateY"]
        else:
            self.ops_list = ops_list

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            #"Identity": (torch.tensor(0.0), False), not needed
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False), # added
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 110, num_bins), False), # added
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "Cutout": (torch.linspace(0, 40, num_bins), False), #added
            "TranslateX": (torch.linspace(0.0, 150.0 / 336.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 336.0 * image_size[0], num_bins), True),
            "Grayscale": (torch.tensor(0.0), False),
            "ChromaDrop": (torch.tensor(0.0), False),
            "AutoSaturation": (torch.tensor(0.0), False),
            "AutoSaturation_old": (torch.tensor(0.0), False),
            "Rotate90": (torch.tensor(90.0), True),
        }


    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = get_dimensions(img)
        #if isinstance(img, Tensor):
        #    if isinstance(fill, (int, float)):
        #        fill = [float(fill)] * channels
        #    elif fill is not None:
        #        fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(self.ops_list), (1,)).item())
            op_name = list(self.ops_list)[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


class ToTensor_range(torch.nn.Module):
    r"""
    Converts PIL image to Tensor into a specified range

    Args:
        val_min = minimum value after convert
        val_max = maximum value after convert
        dtype = dtype after convert (default=torch.float32)

    Returns:
        Converted Torch Tensor
    """

    def __init__(
        self,
        val_min: float = -1.,
        val_max: float = 1.,
        dtype = torch.float32,
    ) -> Tensor:
        super().__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
            img (PIL Image): Image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        #assert F._is_pil_image(img), "Input should be a PIL image (ToTensor_range transform)"
        if F._is_pil_image(img):
            img = F.to_tensor(img) # to_tensor normalizes data to (0,1)
        img = img.to(self.dtype) # convert dtype
        img = self.val_min + (img * (self.val_max - self.val_min)) # scale to val_min to val_max

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"val_min={self.val_min}"
            f", val_max={self.val_max}"
            f", dtype={self.dtype}"
            f")"
        )
        return s

def apply_PILJPEG(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0) # move pointer to 0 so we can read them
    img = PIL.Image.open(buffer).convert("RGB")
    return img

def apply_cv2JPEG(img, quality):
    # convert PIL image to cv2 image
    img_cv2 = np.array(img)
    img_cv2 = img_cv2[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return PIL.Image.fromarray(decimg[:,:,::-1])

def apply_randomJPEG(img, quality):
    if random.random() < 0.5:
        img = apply_PILJPEG(img, quality) # randomly apply PIL or CV2
    else:
        img = apply_cv2JPEG(img, quality)
    return img

def resize_with_random_intpl(img, size):
    """
    Perform resizing with random interpolation
    """
    #intp_list = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING, InterpolationMode.BOX]
    intp_list = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
    #interp_idx = random.randint(0, len(intp_list)-1)
    interp = random.choice(intp_list)
    # random interpolation somehow doesn't work
    img = F.resize(img, size, interpolation=interp)
    return img

class RandomResizeWithRandomIntpl(torch.nn.Module):
    r"""
    Reads PIL Image. Resizes with random interpolation. Returns torch tensor.
    """

    def __init__(
        self,
        size_range: int=(112,448),
    ) -> Tensor:
        super().__init__()
        self.size_range = size_range

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (RandomResizeWithRandomIntpl transform)"
        # add resize
        img = resize_with_random_intpl(img, random.randint(self.size_range[0], self.size_range[1]))
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}()"
            f" size_range={self.size_range}"
            f")"
        )

class ResizeWithRandomIntpl(torch.nn.Module):
    r"""
    Reads PIL Image. Resizes with random interpolation. Returns torch tensor.
    """

    def __init__(
        self,
        size: int,
    ) -> Tensor:
        super().__init__()
        self.size = size

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (ResizeWithRandomIntpl transform)"
        # add resize
        img = resize_with_random_intpl(img, self.size)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f" size={self.size}"
            f")"
        )
        return s

class RRCWithRandomIntpl(T.RandomResizedCrop):
    r"""
    Reads PIL Image. Randomly resized crop with random interpolation. Returns torch tensor.
    """

    def __init__(
        self,
        size: int,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3./4., 4./3.),
    ) -> Tensor:
        super().__init__(size=size, scale=scale, ratio=ratio)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.intp_list=[InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING, InterpolationMode.BOX]

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (RRCWithRandomIntpl transform)"
        # add resize
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        #interp_idx = random.randint(0, len(self.intp_list)-1)
        interp = random.choice(self.intp_list) # somehow doesn't work. Gives me error: TypeError: resized_crop() got multiple values for argument 'interpolation'
        return F.resized_crop(img, i, j, h, w, self.size, interpolation=interp)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f" size={self.size}"
            f", scale={self.scale}"
            f", ratio={self.ratio}"
            f")"
        )
        return s

class JPEGinMemory(torch.nn.Module):
    r"""
    Reads PIL Image. Compress JPEG in memory. Returns PIL Image.

    """

    def __init__(
        self,
        quality_range = (30, 100),
        method: str = "cv,pil",
        dtype = torch.float32,
    ) -> Tensor:
        super().__init__()
        self.quality_range = quality_range
        self.method = method.lower().split(',')
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.jdt

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (ResizeAndJPEGinMemory transform)"
        if "cv" in self.method and "pil" in self.method:
            img = apply_randomJPEG(img, random.randint(self.quality_range[0], self.quality_range[1]))
        elif "cv" in self.method:
            img = apply_cv2JPEG(img, random.randint(self.quality_range[0], self.quality_range[1]))
        elif "pil" in self.method:
            img = apply_PILJPEG(img, random.randint(self.quality_range[0], self.quality_range[1]))
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f", quality_range={self.quality_range}"
            f", dtype={self.dtype}"
            f")"
        )
        return s

class ResizeAndJPEGinMemory(torch.nn.Module):
    r"""
    Reads PIL Image. Resizes and compresses to JPEG in memory. Returns torch tensor.

    """

    def __init__(
        self,
        size: int,
        quality: int = 95,
        method: str = "cv,pil",
        dtype = torch.float32,
    ) -> Tensor:
        super().__init__()
        self.size = size
        self.quality = quality
        self.method = method.lower().split(',')
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (ResizeAndJPEGinMemory transform)"
        # add resize
        img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR) # this is the right way to resize! If torchvision updates, make sure that this resizes the smaller side to the specified size and keeps the aspect ratio
        if "cv" in self.method and "pil" in self.method:
            img = apply_randomJPEG(img, self.quality)
        elif "cv" in self.method:
            img = apply_cv2JPEG(img, self.quality)
        elif "pil" in self.method:
            img = apply_PILJPEG(img, self.quality)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f" size={self.size}"
            f", quality={self.quality}"
            f", dtype={self.dtype}"
            f")"
        )
        return s
    
class StochasticJPEG(torch.nn.Module):
    r"""
    Stochastically applies multiple JPEG compression and resizing to an image.
    """

    def __init__(
        self,
        size: int, # final output size
        quality: Tuple[int, int] = (50, 100), # quality range
        num_jpeg: Tuple[int, int] = (1, 5), # number of jpegs to apply
        jpeg_p: float = 0.5, # probability of applying JPEG compression
        rrc_p: float = 0.5, # probability of applying random resized crop
        rrc_scale: Tuple[float, float] = (0.75, 1.0), # random resize crop scale
        rrc_ratio: Tuple[float, float] = (3./4., 4./3.), # random resize crop ratio
        no_rrc: bool = False, # if True, no random resized crop is applied
        dtype: type = torch.float32,
    ) -> Tensor:
        """
        Initialize the CustomTransforms class.

        Args:
            size (int): The final output size.
            quality (Tuple[int, int]): The quality range as a tuple of two integers.
            num_jpeg (Tuple[int, int]): The number of jpegs to apply as a tuple of two integers.
            p (float): The probability of applying the transform.
            rrc_scale (Tuple[float, float]): The random resize crop scale as a tuple of two floats.
            rrc_ratio (Tuple[float, float]): The random resize crop ratio as a tuple of two floats.
            no_rrc (bool): If True, no random resized crop is applied.
            dtype (type): The data type of the tensor.

        Returns:
            Tensor: The initialized CustomTransforms object.
        """
        super().__init__()
        self.size = size
        self.quality = quality
        self.num_jpeg = num_jpeg
        self.jpeg_p = jpeg_p
        self.rrc_p = rrc_p
        self.rrc = torch.nn.Identity() if no_rrc else T.RandomResizedCrop(size=size, scale=rrc_scale, ratio=rrc_ratio, interpolation=InterpolationMode.BILINEAR)
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (StochasticJPEG transform)"

        # randomly sample p
        count = self.num_jpeg[0]
        for _ in range(self.num_jpeg[0]): # apply min number of jpegs and RRC first
            img = self.rrc(img)
            img = apply_randomJPEG(img, random.randint(self.quality[0], self.quality[1]))
        
        while count < self.num_jpeg[1]:
            if random.random() < self.p: # apply more jpegs with set probability.
                img = self.rrc(img)
                img = apply_randomJPEG(img, random.randint(self.quality[0], self.quality[1]))
                count += 1
            else:
                break
        
        return img

class RandomJPEG(torch.nn.Module):
    """
    Randomly applies JPEG
    Args:
        quality: tuple of quality value range for JPEG
        p: probability of applying JPEG
    """
    def __init__(
        self,
        quality_list: tuple = (30, 100),
        p: float = 0.5,
    ):
        super().__init__()
        self.quality_list = quality_list
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            img = apply_randomJPEG(img, random.randint(self.quality_list[0], self.quality_list[1]))
        return img

class RandomGaussianBlur(torch.nn.Module):
    """
    Randomly applies Gaussian Blur
    Args:
        p: probability of applying JPEG
        sigma: tuple of sigma values for Gaussian Blur
    """
    def __init__(
        self,
        p: float = 0.5,
        sigma: Tuple[float, float] = (0.0, 3.0),
    ):
        super().__init__()
        self.p = p
        self.sigma = sigma
    
    def forward(self, img):
        if random.random() < self.p:
            sigma=random.uniform(self.sigma[0], self.sigma[1])
            kernel_size=1+2*round(sigma*4.0) # default sigma used in scipy (https://github.com/scipy/scipy/blob/v1.13.1/scipy/ndimage/_filters.py#L286-L390)
            img = F.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        return img

class RandomPaddingAndResize(torch.nn.Module):
    r"""
    Reads PIL Image. Randomly applies padding, and resize it back to original resolution.

    """

    def __init__(
        self,
        pad_percentage_range = (0.1, 0.1), # random padding percentage for x (width) and y (height)
        padding_value_range = (0, 255), # random padding value range
    ) -> Tensor:
        super().__init__()
        self.pad_percentage_range = pad_percentage_range
        self.padding_value_range = padding_value_range

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.jdt

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (ResizeAndJPEGinMemory transform)"
        original_size = img.size
        pad_x_l = random.uniform(0, self.pad_percentage_range[0]/2) # x-axis random padding ratio (left)
        pad_x_r = random.uniform(0, self.pad_percentage_range[0]/2) # x-axis random padding ratio (right)
        pad_y_l = random.uniform(0, self.pad_percentage_range[1]/2) # y-axis random padding ratio (left)
        pad_y_r = random.uniform(0, self.pad_percentage_range[1]/2) # y-axis random padding ratio (right)
        pad_fill = random.randint(int(self.padding_value_range[0]), int(self.padding_value_range[1])) # random padding fill value
        img = F.pad(img, (int(pad_x_l*img.size[0]), int(pad_y_l*img.size[1]), int(pad_x_r*img.size[0]), int(pad_y_r*img.size[1])), fill=pad_fill, padding_mode='constant')
        img = F.resize(img, original_size, interpolation=InterpolationMode.BILINEAR)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f", pad_percentage_range={self.pad_percentage_range}"
            f", padding_value_range={self.padding_value_range}"
            f")"
        )
        return s

class RandomCutout(T.RandomErasing):
    r"""
    Random cutout with random numbers
    """
    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value_range=(0, 255),
    ):
        super().__init__(p=p, scale=scale, ratio=ratio)
        self.value_range = value_range
    
    def forward(self, img):
        convert_to_pil=False
        if F._is_pil_image(img):
            img = F.pil_to_tensor(img)
            convert_to_pil=True
        if torch.rand(1) < self.p:
            rand_value = random.randint(self.value_range[0], self.value_range[1])
            # cast self.value to script acceptable type
            if isinstance(rand_value, (int, float)):
                rand_value = [float(rand_value)]
            elif isinstance(rand_value, str):
                rand_value = None
            elif isinstance(rand_value, (list, tuple)):
                rand_value = [float(v) for v in rand_value]
            else:
                rand_value = rand_value

            if rand_value is not None and not (len(rand_value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )
            x, y, h, w, v = self.get_params(img, self.scale, self.ratio, rand_value)
            img = F.erase(img, x, y, h, w, v)
        if convert_to_pil:
            img = F.to_pil_image(img)
        return img

class RandomVisualization(torch.nn.Module):
    r"""
    Randomly visualizes the fully augmented images by saving them at a specified directory.
    """
    def __init__(
        self,
        save_dir: str = "/nfs/turbo/coe-ahowens-nobackup/jespark/visualizations/fake_img",
        save_p: float = 0.01,
        max_imgs: int = 500,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_p = save_p
        self.max_imgs = max_imgs
        self.overwrite = overwrite
        self.skip_namecheck=False

    def next_available_filename(self, save_dir, max_imgs):
        # Returns next available filename
        # image format = visualization_{03d}_{i}.png, i=[0, max_imgs)
        # let's not make it overwrite
        imgs = os.listdir(save_dir)
        imgs_list = [int(img.split("_")[-1].split(".")[0]) for img in imgs]
        random_int = random.randint(0, 999)
        if len(imgs_list) >= max_imgs:
            if self.overwrite:
                return random.choice(imgs) # overwrite random file from imgs
            else:
                self.skip_namecheck=True
                return False
        elif len(imgs_list) > 0:
            next_int = max(imgs_list) + 1
            return f"visualization_{next_int}_{random_int:03d}.png"
        elif len(imgs_list) == 0:
            return f"visualization_0_{random_int:03d}.png"
        else: # uncaught, unexpected situation.
            raise ValueError("Error in next_available_filename")
        
    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        if not self.skip_namecheck:
            if random.random() < self.save_p:
                os.makedirs(self.save_dir, exist_ok=True)
                filename = self.next_available_filename(self.save_dir, self.max_imgs)
                if filename:
                    img.save(os.path.join(self.save_dir, filename))
        return img

class RandomStateAugmentation(torch.nn.Module):
    r"""
    Randomly applies augmentations given in the input
    """
    def __init__(
        self,
        resize_size=256,
        crop_size=224,
        auglist="JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding",
        min_augs='0',
        max_augs='5',
    ):
        """
        auglist: augmentation lists to apply. Input comma-separated string of augmentations.
        min_augs: minimum number of augmentations to apply. (can be comma-separated string to denote per-augmentation minimum)
        max_augs: maximum number of augmentations to apply. (can be comma-separated string to denote per-augmentation maximum)
        """
        super().__init__()
        self.resize_size=resize_size
        self.crop_size=crop_size

        self.auglist = self.parse_auglist(auglist)
        # convert min_augs and max_augs to appropriate format
        min_augs = self.parse_augnums(min_augs)
        max_augs = self.parse_augnums(max_augs)
        if type(min_augs) == list:
            assert type(max_augs) == list, "max_augs should be list if min_augs is list."
            assert len(min_augs) == len(auglist), "min_augs length should be equal to auglist length."
            assert len(max_augs) == len(auglist), "max_augs length should be equal to auglist length."
        # convert min_augs and max_augs to list if they are not
        self.min_augs = [min_augs] * len(self.auglist) if type(min_augs) != list else min_augs
        self.max_augs = [max_augs] * len(self.auglist) if type(max_augs) != list else max_augs
    
    def parse_augnums(self, augsnum):
        # parse min_augs or max_augs. They are expected to be a string of integers, optinally separated by commas.
        augsnum_list = augsnum.split(",")
        if len(augsnum_list) == 1:
            return int(augsnum_list[0])
        else:
            return [int(aug) for aug in augsnum_list]


    def parse_auglist(self, auglist):
        # parse str-comma-separated auglist to list of augmentations
        # default augmentation thoughts: "JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding"
        auglist_list = auglist.split(",")
        parsed_list = torch.nn.ModuleList()
        for aug_name in auglist_list:
            if aug_name=='singleJPEG':
                parsed_list.append(ResizeAndJPEGinMemory(size=self.crop_size, quality=95, dtype=torch.float32))
            if aug_name=='StochasticJPEG':
                parsed_list.append(StochasticJPEG(size=self.crop_size, quality=(75, 100), num_jpeg=(1, 5), jpeg_p=0.5, rrc_p=0.5, rrc_scale=(0.75, 1.0), rrc_ratio=(3./4., 4./3.), no_rrc=False, dtype=torch.float32))
            if aug_name=='JPEGinMemory':
                parsed_list.append(JPEGinMemory(quality_range=(75, 100), dtype=torch.float32))
            if aug_name=='RandomResizeWithRandomIntpl':
                parsed_list.append(RandomResizeWithRandomIntpl(size_range=(self.crop_size+1,round(self.crop_size*1.228)))) # should not be smaller; causes issues with Random Crop.
            if aug_name=='RandomCrop':
                parsed_list.append(T.RandomCrop(self.crop_size))
            if aug_name=='RandomHorizontalFlip':
                parsed_list.append(T.RandomHorizontalFlip())
            if aug_name=='RandomVerticalFlip':
                parsed_list.append(T.RandomVerticalFlip())
            if aug_name=='RRCWithRandomIntpl':
                parsed_list.append(RRCWithRandomIntpl(size=self.crop_size, scale=(0.9, 1.0), ratio=(3./4., 4./3.)))
            if aug_name=='RandomRotation':
                parsed_list.append(T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR))
            if aug_name=='RandomTranslate':
                parsed_list.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None, interpolation=InterpolationMode.BILINEAR))
            if aug_name=='RandomShear':
                parsed_list.append(T.RandomAffine(degrees=0, translate=None, scale=None, shear=(-15, 15, -15, 15), interpolation=InterpolationMode.BILINEAR))
            if aug_name=='RandomPadding' or aug_name=='RandomPaddingAndResize':
                parsed_list.append(RandomPaddingAndResize(pad_percentage_range=(0.1, 0.1), padding_value_range=(0, 255)))
            if aug_name=='RandomCutout':
                parsed_list.append(RandomCutout(p=0.5, scale=(0.02, 0.06), ratio=(0.3, 3.3), value_range=(0, 255)))
        
        return parsed_list

    def generate_randAug_counts(self):
        # Generates random required counts per augmentation
        per_aug_counts = [0] * len(self.auglist)
        for i in range(len(per_aug_counts)):
            per_aug_counts[i] = random.randint(self.min_augs[i], self.max_augs[i])
        return per_aug_counts

    def convert_aug_counts_to_idxList(self, per_aug_counts):
        # convert per augmentation count to list of indices. For example, [1,3,2] = [0,1,1,1,2,2]
        idxList = []
        for i in range(len(per_aug_counts)):
            idxList += [i] * per_aug_counts[i]
        return idxList

    def check_if_complete(self, count, min_augs):
        # not needed
        if type(min_augs) == list:
            min_augs_list = min_augs
        else:
            min_augs_list = [min_augs] * len(self.auglist)
        for i in range(len(min_augs_list)):
            if count[i] < min_augs_list[i]:
                return False
        return True

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        assert F._is_pil_image(img), "Input should be a PIL image (RandomStateAugmentation transform)"
        # randomly applies augmentation. Randomly walks through the list of augmentations and applies them. They should be applied at least "min_augs" number of times.
        #count = [0] * len(self.auglist)

        idxList = self.convert_aug_counts_to_idxList(self.generate_randAug_counts())

        while len(idxList) > 0:
            randomIdx = idxList.pop(random.randint(0, len(idxList)-1)) # randomly pop index from idxList
            img = self.auglist[randomIdx](img)
            #count[randomIdx] += 1 # not needed, idxList contains exact amount of augmentations to apply per idx.

        return img
    
class RandomSignRotation(torch.nn.Module):
    r"""
    Randomly rotates the image by given angle. Randomly changes sign.
    """

    def __init__(
        self,
        angle: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> Tensor:
        super().__init__()
        self.angle = angle
        self.interpolation = interpolation

    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        if random.random() < 0.5:
            angle = -self.angle
        else:
            angle = self.angle
        img = F.rotate(img, angle, interpolation=self.interpolation)
        return img
    
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f" angle={self.angle}"
            f", interpolation={self.interpolation}"
            f")"
        )
        return s
    
class RandomResize(torch.nn.Module):
    r"""
    Randomly resizes the input. Either up or downsample and then return it to the original size. Arguments take percentage of resizing (e.g., 0.3 means it can be downsized or upsampled by 30%)
    """
    def __init__(
        self,
        resize_percentage: float,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> Tensor:
        super().__init__()
        self.resize_percentage = resize_percentage
        self.interpolation = interpolation
    
    def forward(self, img) -> Tensor:
        """
        Args:
            img: PIL image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        if random.random() < 0.5:
            resize_percentage = 1.0 - self.resize_percentage
        else:
            resize_percentage = 1.0 + self.resize_percentage
        original_size_1, original_size_0 = img.size # width, height
        img = F.resize(img, (int(original_size_0*resize_percentage), int(original_size_1*resize_percentage)), interpolation=self.interpolation) # resized height, width
        img = F.resize(img, (original_size_0, original_size_1), interpolation=self.interpolation)
        return img
    
