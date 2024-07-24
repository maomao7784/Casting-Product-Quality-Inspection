import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class YOLOModel(nn.Module):
    def __init__(self, num_classes):
        super(YOLOModel, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
