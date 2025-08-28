import dataloader as dl
import torch
import argparse
import PIL.Image as Image
from typing import Union, List

from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import PushToHubMixin

class CommForImageProcessor(BaseImageProcessor, PushToHubMixin):
    """
    Image processor for Community Forensics VIT model. Processes PIL images and returns PyTorch tensors.
    """
    image_processor_type = "commfor_image_processor"
    model_input_names = ["pixel_values"]

    def __init__(self, size=384, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        assert self.size in [224, 384], f"Unsupported size: {self.size}. Supported sizes are 224 and 384."

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        mode: str = "test",
        **kwargs
    ):
        """
        Preprocess the input images to PyTorch tensors.
        """
        assert mode in ["test", "train"], f"Unsupported mode: {mode}. Supported modes are 'test' and 'train'."
        assert isinstance(images, (Image.Image, list)), "Input must be a PIL Image or a list of PIL Images."
        if isinstance(images, Image.Image):
            images = [images]
        
        args = argparse.Namespace()
        args.input_size = self.size
        args.rsa_ops="JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding,RandomCutout"
        args.rsa_min_num_ops='0'
        args.rsa_max_num_ops='2'

        transform = dl.get_transform(args, mode=mode)

        processed_images = [transform(image) for image in images] # the output would be tensors
        if len(processed_images) == 1:
            return {"pixel_values": processed_images[0]}
        else:
            return {"pixel_values": torch.stack(processed_images)}
        



