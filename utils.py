import torch
import numpy as np


def iou (box1, box2):
    """
    Parameters:
        box1 (tensor): the 4 outputs of a batch of boxes (BATCH_SIZE, 7, 7, 4)
        box2 (tensor): the 4 outputs of a batch of boxes (BATCH_SIZE, 7, 7, 4)

    Returns:
        The iou of the two boxes given shape: (m, 7, 7, 1)
    """
    box1_x, box1_y, box1_w, box1_h = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
    # box1_x: tensor of shape (m, 7, 7, 1)
    box2_x, box2_y, box2_w, box2_h = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]
    # box2_x: tensor of shape (m, 7, 7, 1)

    box1_x1, box1_x2 = box1_x - 0.5 * box1_w, box1_x + 0.5 * box1_w
    # box1_x1: tensor of shape (m, 7, 7, 1)
    box1_y1, box1_y2 = box1_y - 0.5 * box1_h, box1_y + 0.5 * box1_h

    box2_x1, box2_x2 = box2_x - 0.5 * box2_w, box2_x + 0.5 * box2_w
    # box2_x1: tensor of shape (m, 7, 7, 1)
    box2_y1, box2_y2 = box2_y - 0.5 * box2_h, box2_y + 0.5 * box2_h

    xi1 = torch.max(box1_x1, box2_x1)
    # xi1: tensor of shape (m, 7, 7, 1)
    yi1 = torch.max(box1_y1, box2_y1)
    xi2 = torch.min(box1_x2, box2_x2)
    yi2 = torch.min(box1_y2, box2_y2)

    inter_width = xi2 - xi1
    # inter_width: tensor of shape (m, 7, 7, 1)
    inter_height = yi2 - yi1
    inter_area = torch.max(inter_width, torch.tensor(0)) * torch.max(inter_height, torch.tensor(0))
    # inter_area: tensor of shape (m, 7, 7, 1)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    # box1_area: tensor of shape (m, 7, 7, 1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - inter_area
    return inter_area / (union + 1e-6)

def test_iou():
    box1 = torch.tensor([[0.5,0.5,0.5,0.5], [1,1,1,1]])
    box2 = torch.tensor([[2,2,0.1,0.1], [1, 1, 0.5, 0.5]])
    result = iou(box1, box2)
    assert result[0].item() == 0, "the iou should be 0"
    assert np.isclose(result[1].item(), 0.25), "the iou should be close to 1"

# test_iou()