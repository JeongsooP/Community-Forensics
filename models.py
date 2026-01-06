import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import PIL.Image as Image
    
class ViTClassifier(nn.Module):
    def __init__(self, args, device='cuda', dtype=torch.float32):
        """
        # ViT Classifier based on huggingface timm module
        """
        super(ViTClassifier, self).__init__()
        self.args = args
        self.device=device
        self.dtype=dtype
        if args.input_size==224:
            if args.patch_size==16:
                self.vit = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True).to(device)
            else:
                raise ValueError(f"Unsupported patch size: {args.patch_size}")
        elif args.input_size==384:
            if args.patch_size==16:
                self.vit = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True).to(device)
            else:
                raise ValueError(f"Unsupported patch size: {args.patch_size}")
        if args.freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.vit.head = nn.Linear(in_features=384, out_features=1, bias=True, device=device, dtype=dtype)
        for param in self.vit.head.parameters():
            assert param.requires_grad==True, "Model head should be trainable."
    
    def preprocess_input(self, x):
        """
        # Preprocess input image for ViT model.
        """
        #x = x/255.0
        assert isinstance(x, Image.Image) or isinstance(x, torch.Tensor), "Input should be a PIL image or a PyTorch tensor."
        norm_mean = [0.485, 0.456, 0.406] #imagenet norm
        norm_std = [0.229, 0.224, 0.225]
        augment_list = []
        resize_size=440 # assuming default input size is 384 as it is the best model
        crop_size=384
        if self.args.input_size==224:
            resize_size=256
            crop_size=224
        
        augment_list.extend([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.ConvertImageDtype(self.dtype),
        ])
        preprocess = transforms.Compose(augment_list)
        x = preprocess(x)
        x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self.preprocess_input(x).to(self.device)
        x = self.vit(x)
        x = torch.nn.functional.sigmoid(x)
        return x
        
