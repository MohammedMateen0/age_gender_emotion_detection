from torchvision.models import (
    resnet18,
    ResNet18_Weights
)
import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        for param in self.backbone.parameters():
            param.requires_grad=False
        in_features=(
            self.backbone.fc.in_features
        )
        self.backbone.fc=nn.Identity()
        self.emotion_head=nn.Linear(
            in_features,
            7
        )
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad=True
    def forward(self,x):
        features=self.backbone(x)

        emotion_logits=(
            self.emotion_head(features)
        )
        return emotion_logits
    
if __name__ == "__main__":

    model = EmotionNet()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable Parameters: "
        f"{trainable:,}"
    )

    x = torch.randn(
        4,
        3,
        224,
        224
    )

    output = model(x)

    print(output.shape)