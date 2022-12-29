from torch import nn
from torchvision import models


def res_model(num_classes):
    model = models.resnet18()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes))
    return model
