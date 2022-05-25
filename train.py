import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim
from model import Yolov1
from loss import YoloLoss
from Dataset import VOCDataset
from torch.utils.data import DataLoader
from alladinyolo import Yolov1 as yv1

seed = 123
torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "images"
LABEL_DIR = "labels"


class Compose:
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img), boxes
        return img, boxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for image, label in loop:
        torch.autograd.set_detect_anomaly(True)
        image, label = image.to(DEVICE), label.to(DEVICE)
        pred = model(image)
        loss = loss_fn(pred, label)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = yv1(split_size=7, num_boxes=2, num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    train_data = VOCDataset("8examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_data = VOCDataset("test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(train_data, batch_size=4,  shuffle=True,
                              drop_last=True, pin_memory=PIN_MEMORY,
                              )
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,  shuffle=True,
                              drop_last=True, pin_memory=PIN_MEMORY)

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()