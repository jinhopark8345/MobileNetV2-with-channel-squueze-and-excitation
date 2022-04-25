import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision.models as models
import torch.autograd.profiler as profiler


class InvertedResidualBlock(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride, bias=False):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride

        expanded_planes = expansion * in_planes
        # conv1: pointwise convolution
        # 1x1 expansion layer
        self.conv1 = nn.Conv2d(
            in_planes,
            expanded_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(expanded_planes)

        self.conv2 = nn.Conv2d(
            expanded_planes,
            expanded_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=expanded_planes,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(expanded_planes)

        # 1x1 Projection Layer
        # conv3: pointwise conv again
        self.conv3 = nn.Conv2d(
            expanded_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class ChannelWiseAvgPool(nn.Module):
    def __init__(
        self, channel, width, reduction=16, min_cap=8
    ):  # number of nodes for scaling
        super(ChannelWiseAvgPool, self).__init__()
        self.reduction = reduction
        self.avg_pool = nn.AvgPool3d(kernel_size=(channel, 1, 1))
        self.hw = width * width
        self.mid_num = self.hw // self.reduction
        self.mid_num = min(
            min_cap, self.mid_num
        )  # minimum cap so that "between nodes" won't be too small
        self.fc1 = nn.Linear(self.hw, self.mid_num, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.mid_num, self.hw, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        t, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.view(t, 1, self.hw)
        y = y.view(t, self.hw)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)

        #########################
        y = y.view(t, 1, h, w)
        return x * y.expand_as(x)


class ChannelWiseSqueezeAndExcitation(nn.Module):  # CSE block
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, stride, width, bias=False):
        super(ChannelWiseSqueezeAndExcitation, self).__init__()
        self.temp_stride = stride
        self.width = width
        if stride == 2:
            stride = 1

        # planes = expansion * in_planes
        # conv1: pointwise convolution
        # 1x1 expansion layer
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_planes,
            bias=bias,
        )
        self.act1 = nn.PReLU()

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.channelwise_avgpool = ChannelWiseAvgPool(out_planes, width)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.act2 = nn.PReLU()
        self.avgpool_conv1 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            groups=out_planes,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.channelwise_avgpool(out)
        out = out + self.shortcut(x)  # always shortcut connect
        out = self.avgpool_conv1(out) if self.temp_stride == 2 else out
        return out


class MobileNetV2_with_CSE(nn.Module):
    width = 32

    def __init__(
        self,
        bottle_neck_configuration,
        channel_squeeze_and_excitation_configuration,
        num_classes=10,
        show_configuation=True
    ):
        super(MobileNetV2_with_CSE, self).__init__()

        self.show_configuration = show_configuation
        if self.show_configuration:
            indent = '\t'
            print("bottleneck blocks configuration: ")
            [print(indent, layer) for layer in bottle_neck_configuration]
            print("channel squeeze and excitation configuration: ")
            [print(indent, layer) for layer in channel_squeeze_and_excitation_configuration]

        self.cfg_btn = bottle_neck_configuration
        self.cfg_cse = channel_squeeze_and_excitation_configuration

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()

        self.btn_blocks = self._btn_blocks(in_planes=32)
        self.cse_blocks = self._cse_blocks(in_planes=32)
        # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, groups=320, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            320,
            1280,
            kernel_size=1,
            stride=1,
            groups=320,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.act2 = nn.ReLU()
        self.avgpool_conv1 = nn.Conv2d(
            1280,
            1280,
            kernel_size=2,
            stride=2,
            groups=1280,
            padding=0,
            bias=False,
        )

        self.linear = nn.Linear(1280, num_classes)

    def _btn_blocks(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg_btn:
            strides = [stride] + [1] * (num_blocks - 1)
            # print(strides)
            for stride in strides:
                if self.show_configuration:
                    print(f'{in_planes, out_planes, expansion, stride, self.width = }')
                layers.append(
                    InvertedResidualBlock(
                        in_planes, out_planes, expansion, stride, bias=False
                    )
                )
                if stride == 2:
                    self.width = self.width // 2
                in_planes = out_planes
        return nn.Sequential(*layers)

    def _cse_blocks(self, in_planes):
        layers = []
        for out_planes, num_blocks, stride in self.cfg_cse:

            # if stride == 2 and num block is bigger than 1, rest of the blocks should have stride == 1
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                if self.show_configuration:
                    expansion = None
                    print(f'{in_planes, out_planes, expansion, stride, self.width = }')
                layers.append(
                    ChannelWiseSqueezeAndExcitation(
                        in_planes, out_planes, stride, self.width, bias=False
                    )
                )
                if stride == 2:
                    self.width = self.width // 2
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.btn_blocks(out)
        out = self.cse_blocks(out)
        out = self.act2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def main():

    # bottleneck blocks configurations:
    #   (expansion, out_planes, num_blocks, stride)
    cfg_btn = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 1),
    ]

    # channel-wise squeeze and excitation configurations:
    #   (out_planes, num_blocks, stride)
    cfg_cse = [
        (64, 4, 2),
        (96, 3, 2),
        (160, 3, 1),
        (320, 1, 2),
    ]

    model = MobileNetV2_with_CSE(
        bottle_neck_configuration=cfg_btn,
        channel_squeeze_and_excitation_configuration=cfg_cse,
    )

    # printing summary of the model
    from torchsummary import summary
    summary(model, (3, 32, 32), device="cpu") # CIFAR10 dataset image size : (3, 32, 32)

    # sample_data = torch.randn(1, 3, 32, 32)
    # sample_output = model(sample_data)


if __name__ == "__main__":
    main()
