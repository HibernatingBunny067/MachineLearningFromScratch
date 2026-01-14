import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self,S=7,n=7,c=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.n = n
        self.c = c
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self,predictions,target):
        predictions = predictions.reshape(-1,self.S,self.S,self.c + self.n*5)

        iou_b1 = intersection_over_union(predictions[...,self.c+1:self.c+5],target[...,self.c+1:self.c+5])
        iou_b2 = intersection_over_union(predictions[...,self.c+6:self.c+10],target[...,self.c+1:self.c+5])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        best_box = torch.argmax(ious,dim=0).unsqueeze(-1)

        exists_box = target[...,self.c].unsqueeze(3)

        ##BBOX loss
    
        box_predictions = exists_box*(
            best_box*predictions[...,self.c+6:self.c+10] +
            (1-best_box)*predictions[...,self.c+1:self.c+5]
        )

        box_targets = exists_box*target[...,self.c+1:self.c+5]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2)
        )

        ##object loss
        pred_box = (
            best_box*predictions[...,self.c+5:self.c+6] +
            (1-best_box)*predictions[...,self.c:self.c+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[...,self.c:self.c+1])
        )

        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*pred_box),
            torch.flatten((1-exists_box)*target[...,self.c:self.c+1])
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[...,self.c+5:self.c+6]),
            torch.flatten((1-exists_box)*target[...,self.c,self.c+1])
        )

        class_loss = self.mse(
            torch.flatten(exists_box*predictions[...,:self.c],end_dim=-2),
            torch.flatten(exists_box*target[...,:self.c],end_dim=-2)
        )

        loss = (
            self.lambda_coord*box_loss +
            object_loss +
            self.lambda_noobj*no_object_loss +
            class_loss
        )

        return loss