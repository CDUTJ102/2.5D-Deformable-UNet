from model.utils import *
import torch as t


class D_UNet(nn.Module):
    def __init__(self, inc=3, n_classes=1, base_chns=12, norm='in', depth=False, dilation=1):
        super(D_UNet, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv1 = SETriResSeparateConv3D(inc, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv2 = SETriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                            dilat=dilation)
        self.conv3 = SETriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same',
                                            dilat=dilation)
        self.conv4 = SEBasicBlock3D(4 * base_chns, 8 * base_chns)
        self.conv5 = SEBasicBlock3D(8 * base_chns, 4 * base_chns)
        self.conv6_1 = SETriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.conv6_2 = SETriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.conv7_1 = SETriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.conv7_2 = SETriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.conv8_1 = SETriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.conv8_2 = SETriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',
                                              dilat=dilation)
        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels=2 * base_chns, out_channels=n_classes, kernel_size=1),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)
        conv2 = self.conv2(out)
        out = self.downsample(conv2)
        conv3 = self.conv3(out)
        out = self.downsample(conv3)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.upsample(out)
        out = t.cat((out, conv3), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)
        out = self.upsample(out)
        out = t.cat((out, conv2), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.upsample(out)
        out = t.cat((out, conv1), 1)
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        out = self.classification(out)
        out = t.sigmoid(out)
        return out


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256, 80))
    net = D_UNet()
    y = net(x)
    print(y.shape)
