import torch
import torch.utils.data
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[item, 0])
        image = Image.open(img_path)

        label_path = os.path.join(self.label_dir, self.annotations.iloc[item, 1])
        boxes = []
        with open(label_path) as f:
            for lines in f.readlines():
                class_idx, x, y, w, h = [float(num) if float(num) != int(float(num))
                                         else int(num) for num in lines.replace("\n", "").split()]
                boxes.append([class_idx, x, y, w, h])
        boxes = torch.tensor(boxes)
        if self.transform:
            image, boxes = self.transform(image, boxes)

        label = torch.zeros((7, 7, 25))
        for box in boxes:
            class_idx, x, y, w, h = box.tolist()
            x_idx, y_idx = int(7 * x), int(7 * y)
            x, y = 7 * x - x_idx, 7 * y - y_idx
            w, h = w * 7, h * 7
            if label[x_idx, y_idx, 20] == 0:
                label[x_idx, y_idx, 20:25] = torch.tensor([1, x, y, w, h])
                label[x_idx, y_idx, int(class_idx)] = 1

        return image, label

def test():
    data = VOCDataset("8examples.csv", "images", "labels")
    image, label = data[1]
    image = transforms.Resize((448,448))(image)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    test()