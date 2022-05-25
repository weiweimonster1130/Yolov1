import torch
import torch.nn as nn
from utils import iou


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambdacoord = 5
        self.lambdanoobj = 0.5

    def forward(self, pred, label):
        """
        :param pred : tensor of shape (BATCH_SIZE, 7 * 7 * 30)
        :param label : tensor of shape (BATCH_SIZE, 7, 7, 25)
        :var pred1_boxes: tensor of shape ()
        :return: the loss of the entire batch
        """

        pred = pred.reshape(-1, 7, 7, 30)
        identity_mask = label[..., 20:21]
        # identity_mask tensor of shape (m, 7, 7, 1)
        pred1_boxes, pred2_boxes = pred[..., 21:25], pred[..., 26:30]
        # pred1_boxes: tensor of shape (m, 7, 7, 4)
        iou1 = iou(pred1_boxes, label[..., 21:25]).unsqueeze(0)
        # iou1: tensor of shape (m, 7, 7, 1)
        iou2 = iou(pred2_boxes, label[..., 21:25]).unsqueeze(0)
        # iou2: tensor of shape (m, 7, 7, 1)
        best_boxes_idx = torch.argmax(torch.cat((iou1, iou2), dim=0), dim=0)
        # best_boxes_idx: tensor of shape (m, 7, 7, 1)

        pred1_boxes[..., 2:4], pred2_boxes[..., 2:4] = \
            torch.sqrt(pred1_boxes[..., 2:4]), torch.sqrt(pred2_boxes[..., 2:4])
        best_boxes = identity_mask * ((1 - best_boxes_idx) * pred1_boxes + best_boxes_idx * pred2_boxes)
        # best_boxes: tensor of shape (m, 7, 7, 4)

        filtered_label = identity_mask * label
        # filtered_label: tensor of shape (m, 7, 7, 30)

        box_loss = self.mse(best_boxes.flatten(end_dim=-2), filtered_label[..., 21:25].flatten(end_dim=-2))
        # BOX LOSS ENDS HERE

        best_prob = identity_mask * ((1 - best_boxes_idx) * pred[..., 25:26] + best_boxes_idx * pred[..., 20:21])
        filtered_prob = identity_mask * label[..., 20:21]
        # best_prob, filtered_prob: tensor of shape (m, 7, 7, 1)

        object_loss = self.mse(best_prob.flatten(), filtered_prob.flatten())
        # OBJECT LOSS ENDS HERE

        false_best_prob = (1 - identity_mask) * ((1 - best_boxes_idx) * pred[..., 25:26]
                                                 + best_boxes_idx * pred[..., 20:21])
        no_object_prob = (1 - identity_mask) * label[..., 20:21]

        no_object_loss = self.mse(false_best_prob.flatten(), no_object_prob.flatten())
        # NO OBJECT LOSS ENDS HERE

        label_class = label[..., :20]
        # label_class: tensor of shape (m, 7, 7, 20)

        class_loss = self.mse(label_class.flatten(end_dim=-2), pred[..., :20].flatten(end_dim=-2))

        test_loss = self.mse(torch.flatten(pred[...,:25]), torch.flatten(label))
        return self.mse(torch.flatten(pred[...,:25]), torch.flatten(label))

def test():
    pred = torch.randn((16, 7, 7, 30))
    label = torch.randn((16, 7, 7, 25))
    loss_fn = YoloLoss()
    loss = loss_fn(pred, label)
    print(loss.item())

if __name__ == "__main__":
    test()

