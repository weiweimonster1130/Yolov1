"""
this is the implementation of Yolov1
Author: Jacob Hsiung  2020/05/23
"""

import torch
import torch.nn as nn
from alladinyolo import Yolov1 as yv1

layer_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=k, stride=s, padding=p,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.lrelu = nn.LeakyReLU(0.1)
        self.k = k
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        # if any(torch.isnan(out).tolist()):
        #     print("the kernel size is {}, and the out channel is {}".format(self.k, self.out_channel))
        return out


class Yolov1(nn.Module):
    def __init__(self, in_channel=3, c=20):
        super().__init__()
        self.config = layer_config
        self.in_channel = in_channel
        self.conv_layers = self._construct_conv_layers()
        self.fcs = self._construct_fc_layers()

    def forward(self, x):
        out = self.fcs(self.conv_layers(x))
        return out

    def _construct_conv_layers(self):
        layers = []
        in_channel = self.in_channel

        for layer in layer_config:
            if isinstance(layer, tuple):
                k, out_channel, s, p = layer
                layers.append(conv_block(in_channel, out_channel, k, s, p))
                in_channel = out_channel
            elif isinstance(layer, list):
                cb1, cb2, num = layer
                for _ in range(num):
                    layers.append(
                        conv_block(k=cb1[0], out_channel=cb1[1], s=cb1[2], p=cb1[3], in_channel=in_channel)
                    )
                    in_channel = cb1[1]
                    layers.append(
                        conv_block(k=cb2[0], out_channel=cb2[1], s=cb2[2], p=cb2[3], in_channel=in_channel)
                    )
                    in_channel = cb2[1]
            elif isinstance(layer, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                print("You should not reach here there is an error in layer_config")
        return nn.Sequential(*layers)

    def _construct_fc_layers(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 512),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 7 * 7 * 30)
        )


def main():
    test = torch.randn(16, 3, 448, 448)
    model = yv1(split_size=7, num_boxes=2, num_classes=20)
    out = model(test)
    if any(torch.isnan(out).tolist()):
        print("error")
    print(out.shape)


if __name__ == "__main__":
    main()
