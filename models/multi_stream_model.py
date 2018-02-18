from torch import nn
from torchvision.models import resnet50
import torch


class MultiStreamNet(nn.Module):
    def __init__(self):
        super().__init__()

        # resnet = resnet50(pretrained=True)
        # layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # self.resnet_layers = nn.Sequential(layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        #
        # # This can be a Conv2d, instead of AvgPool2d, if we want to reduce the number of channels.
        # self.post_resnet_layer = nn.AvgPool2d(5)
        # image_output_layers = 2048

        image_output_layers = 64
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 4),
            nn.Conv2d(32, 32, 5, 4),
            nn.Conv2d(32, 32, 5, 2),
            nn.Conv2d(32, image_output_layers, 3, 1)
        )

        point_net_output_channels = 64
        self.point_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.Conv2d(16, 32, 3, 2),
            nn.Conv2d(32, 64, 3, 2),
            nn.Conv2d(64, point_net_output_channels, 7, 1)
        )

        # not sure of the input features
        self.fcn_layers = nn.Sequential(
            nn.Linear(image_output_layers + point_net_output_channels + point_net_output_channels, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3)
        )

    def forward(self, *input):
        image, point_1_img, point_2_img = input

        # Have to figure out the image size here.
        # image = self.resnet_layers(image)
        # image = self.post_resnet_layer(image)
        image = self.conv_net(image)

        point_1_img = self.point_net(point_1_img)
        point_2_img = self.point_net(point_2_img)

        image = torch.cat((image, point_1_img, point_2_img), 1)
        image = image.view(image.size(0), -1)

        labels = self.fcn_layers(image)

        return labels
