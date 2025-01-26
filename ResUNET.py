import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super(SingleConv3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


# class DoubleConv3DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout_rate=0.1):
#         super(DoubleConv3DBlock, self).__init__()
#         self.block = nn.Sequential(
#             SingleConv3DBlock(in_channels, out_channels // 2, kernel_size=3, dropout_rate=dropout_rate),
#             SingleConv3DBlock(out_channels // 2, out_channels, kernel_size=3, dropout_rate=dropout_rate)
#         )

#     def forward(self, x):
#         return self.block(x)

class DoubleResidualConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.2):
        super(DoubleResidualConv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout3d(dropout_rate)  # Dropout added here
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)  # Residual connection

    def forward(self, x):
        residual = self.residual(x)  # 1x1 Conv for residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)  # Apply dropout before adding residual
        return self.relu2(out + residual)



class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleDeconv3DBlock, self).__init__()
        self.block = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.block(x)


class UpConv3DBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = SingleDeconv3DBlock(in_channels, in_channels // 2)
        self.conv1 = SingleConv3DBlock(in_channels, in_channels // 2, kernel_size=3)
        self.conv2 = SingleConv3DBlock(in_channels // 2, in_channels // 2, kernel_size=3)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual is not None:
            # Interpolate residual to match the size of upsampled output
            if residual.shape[2:] != out.shape[2:]:
                residual = F.interpolate(residual, size=out.shape[2:], mode='trilinear', align_corners=True)
            out = torch.cat((out, residual), 1)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.last_layer:
            out = self.conv3(out)
        return out


# class UNet3D(nn.Module):
#     def __init__(self, in_channels, num_classes, level_channels=[16, 32, 64, 128], bottleneck_channel=256):
#         super(UNet3D, self).__init__()
#         self.encoder1 = DoubleConv3DBlock(in_channels=in_channels, out_channels=level_channels[0], dropout_rate=0.1)
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder2 = DoubleConv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1], dropout_rate=0.1)
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder3 = DoubleConv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2], dropout_rate=0.2)
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder4 = DoubleConv3DBlock(in_channels=level_channels[2], out_channels=level_channels[3], dropout_rate=0.2)
#         self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.bottleneck = DoubleConv3DBlock(in_channels=level_channels[3], out_channels=bottleneck_channel, dropout_rate=0.3)
#         self.upconv3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[3])
#         self.upconv2 = UpConv3DBlock(in_channels=level_channels[3], res_channels=level_channels[2])
#         self.upconv1 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
#         self.upconv0 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0], num_classes=num_classes, last_layer=True)


#     def forward(self, input):
#         # Encoder path
#         out1 = self.encoder1(input)
#         pool1 = self.pool1(out1)
#         out2 = self.encoder2(pool1)
#         pool2 = self.pool2(out2)
#         out3 = self.encoder3(pool2)
#         pool3 = self.pool3(out3)
#         out4 = self.encoder4(pool3)
#         pool4 = self.pool4(out4)
#         bottleneck = self.bottleneck(pool4)

#  # Decoder path
#         up3 = self.upconv3(bottleneck, out4)
#         up2 = self.upconv2(up3, out3)
#         up1 = self.upconv1(up2, out2)
#         up0 = self.upconv0(up1, out1)
#         return up0

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[16, 32, 64, 128], bottleneck_channel=256, dropout_rate=0.2):
        super(UNet3D, self).__init__()
        self.encoder1 = DoubleResidualConv3DBlock(in_channels=in_channels, out_channels=level_channels[0], dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = DoubleResidualConv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1], dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = DoubleResidualConv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2], dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = DoubleResidualConv3DBlock(in_channels=level_channels[2], out_channels=level_channels[3], dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = DoubleResidualConv3DBlock(in_channels=level_channels[3], out_channels=bottleneck_channel, dropout_rate=dropout_rate)
        self.upconv3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[3])
        self.upconv2 = UpConv3DBlock(in_channels=level_channels[3], res_channels=level_channels[2])
        self.upconv1 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
        self.upconv0 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0], num_classes=num_classes, last_layer=True)

    def forward(self, input):
        # Encoder path
        out1 = self.encoder1(input)
        pool1 = self.pool1(out1)
        out2 = self.encoder2(pool1)
        pool2 = self.pool2(out2)
        out3 = self.encoder3(pool2)
        pool3 = self.pool3(out3)
        out4 = self.encoder4(pool3)
        pool4 = self.pool4(out4)
        bottleneck = self.bottleneck(pool4)

        # Decoder path
        up3 = self.upconv3(bottleneck, out4)
        up2 = self.upconv2(up3, out3)
        up1 = self.upconv1(up2, out2)
        up0 = self.upconv0(up1, out1)
        return up0


# Testing
if __name__ == '__main__':
    #
    model = UNet3D(in_channels=2, num_classes=5)
    input = torch.rand(1, 2, 128, 128, 128)
    out = model(input)
    print(out.shape)