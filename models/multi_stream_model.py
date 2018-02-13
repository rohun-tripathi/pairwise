from torch import nn
from torchvision.models import resnet50


class MultiStreamNet(nn.Module):
    def __init__(self):
        super.__init__()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        resnet = resnet50(pretrained=True)
        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.resnet_layers = nn.Sequential(layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.point_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.Conv2d(16, 32, 3, 2),
            nn.Conv2d(32, 64, 3, 2),
            nn.Conv2d(64, 64, 7, 1)
        )

        # not sure of the input features
        self.fcn_layers = nn.Sequential(
            nn.Linear(192, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3)
        )

    def forward(self, *input):
        image, point_1_img, point2_img = input

        # Have to figure out the image size here.
        image = self.resnet_layers(image)



        pass
