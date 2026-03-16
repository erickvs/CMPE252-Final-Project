import timm
import torch.nn as nn

def build_dl_model(config):
    model_name = config.name
    num_classes = config.get("num_classes", 10)
    pretrained = config.get("pretrained", False)
    
    if model_name == "resnet18":
        model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
        # Modify the stem for 32x32 CIFAR-10 images to prevent spatial collapse
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == "vit_b16":
        # ViT requires 224x224 input (handled by dataloader transforms)
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")
        
    return model