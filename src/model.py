from torchvision.models import (
    resnet18,
    ResNet18_Weights
)
import torch
import torch.nn as nn

class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.agegendernet=resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        for param in self.agegendernet.parameters():
            param.requires_grad=False
        in_features=self.agegendernet.fc.in_features
        self.agegendernet.fc=nn.Identity()
        self.age_head=nn.Linear(
            in_features,
            6
        )
        self.gender_head=nn.Linear(
            in_features,
            2
        )
        for param in self.agegendernet.layer4.parameters():
            param.requires_grad=True
    def forward(self,x):
        features=self.agegendernet(x)
        age_logits=self.age_head(features)
        gender_logits=self.gender_head(features)
        return age_logits,gender_logits