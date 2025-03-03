import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  #Convert to 1-channel grayscale
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  #Convert test images to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        """
        in_channels: number of input channels (for FER2013, this is 1 for grayscale images)
        out_channels: number of output channels (tune this to change feature map depth)
        kernel_size: spatial dimensions of the convolution filter
        stride: step size of the filter
        padding: zero-padding added to both sides of the input
        bias: whether to include a bias term (usually False when using BatchNorm)
        """
        super(SeparableConv2d, self).__init__()

        #Depthwise convolution applies a spatial filter independently for each channel.
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels, bias=bias)

        #Pointwise convolution: 1x1 convolution to mix information across channels.
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        """
        Xception Residual Block
        Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - reps: Number of separable convolutions in the block.
        - stride: Controls downsampling (use stride=2 for reducing spatial size).
        - start_with_relu: Whether to apply ReLU before first convolution.
        - grow_first: Whether to increase channels at the start or later.
        """
        super(XceptionBlock, self).__init__()
        layers = []
        current_in = in_channels

        #First layer: Decide whether to increase the number of filters
        if grow_first:
            if start_with_relu:
                layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(current_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            current_in = out_channels  #Update current input size

        #Add remaining separable conv layers
        for i in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(current_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

        #Apply MaxPooling if downsampling is required
        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        #Combine layers into a sequential block
        self.block = nn.Sequential(*layers)

        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(x)  # Adjust dimensions if necessary
        return out + identity  # Residual connection

class Xception(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.5):  # Added dropout_prob parameter
        super(Xception, self).__init__()

        # **ENTRY FLOW**
        self.entry_flow = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False, grow_first=True),
            XceptionBlock(128, 256, reps=2, stride=2, grow_first=True),
            XceptionBlock(256, 512, reps=2, stride=2, grow_first=True)
        )

        # **MIDDLE FLOW** (with Dropout)
        self.middle_flow = nn.Sequential(
            *[XceptionBlock(512, 512, reps=3, stride=1, grow_first=True) for _ in range(8)],
            nn.Dropout(dropout_prob)  # Dropout added after the middle flow
        )

        # **EXIT FLOW** (with Dropout)
        self.exit_flow = nn.Sequential(
            XceptionBlock(512, 1024, reps=2, stride=2, grow_first=False),

            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout added

            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )

        # **Fully Connected Layer**
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout before final classification
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = torch.flatten(x, 1)  # Flatten before FC layer
        x = self.fc(x)
        return x
