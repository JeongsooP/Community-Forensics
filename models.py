## CLIP extractor model with linear classifier on top.

import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    def __init__(self, args, device='cuda', dtype=torch.float32):
        """
        ViT Classifier based on huggingface timm module
        """
        super(ViTClassifier, self).__init__()
        self.args = args
        self.device=device
        self.dtype=dtype
        if args.model_size=="small":
            if args.input_size==224:
                if args.patch_size==32:
                    self.vit = timm.create_model('vit_small_patch32_224.augreg_in21k_ft_in1k', pretrained=True).to(device)
                elif args.patch_size==16:
                    self.vit = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True).to(device)
            elif args.input_size==384:
                if args.patch_size==32:
                    self.vit = timm.create_model('vit_small_patch32_384.augreg_in21k_ft_in1k', pretrained=True).to(device)
                elif args.patch_size==16:
                    self.vit = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True).to(device)
            if args.freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            self.vit.head = nn.Linear(in_features=384, out_features=1, bias=True, device=device, dtype=dtype)
        elif args.model_size=="tiny":
            assert args.patch_size==16, "Only patch size 16 is available for ViT-Ti"
            if args.input_size==224:
                self.vit = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True).to(device)
            elif args.input_size==384:
                self.vit = timm.create_model('vit_tiny_patch16_384.augreg_in21k_ft_in1k', pretrained=True).to(device)
            if args.freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            self.vit.head = nn.Linear(in_features=192, out_features=1, bias=True, device=device, dtype=dtype)
        for param in self.vit.head.parameters():
            assert param.requires_grad==True, "Model head should be trainable."
        
    def forward(self, x):
        return self.vit(x)
        