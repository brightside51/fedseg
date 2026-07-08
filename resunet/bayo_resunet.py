import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    intersection = torch.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred, smooth=1.0):
    return 1.0 - dice_coef(y_true, y_pred, smooth)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)
        
    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x

class BayoResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stage 0
        self.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Stage 1 (64 channels)
        self.res1_1 = ResidualBlock(64)
        self.res1_2 = ResidualBlock(64)
        self.res1_3 = ResidualBlock(64)
        
        # Stage 2 (128 channels)
        self.down2 = DownBlock(64, 128)
        self.res2_1 = ResidualBlock(128)
        self.res2_2 = ResidualBlock(128)
        self.res2_3 = ResidualBlock(128)
        self.res2_4 = ResidualBlock(128)
        
        # Stage 3 (256 channels)
        self.down3 = DownBlock(128, 256)
        self.res3_1 = ResidualBlock(256)
        self.res3_2 = ResidualBlock(256)
        self.res3_3 = ResidualBlock(256)
        self.res3_4 = ResidualBlock(256)
        self.res3_5 = ResidualBlock(256)
        self.res3_6 = ResidualBlock(256)
        
        # Stage 4 (512 channels)
        self.down4 = DownBlock(256, 512)
        self.res4_1 = ResidualBlock(512)
        self.res4_2 = ResidualBlock(512)
        self.res4_3 = ResidualBlock(512)
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.up_bn1_1 = nn.BatchNorm2d(256)
        self.up_conv1_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.up_bn1_2 = nn.BatchNorm2d(256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up_bn2_1 = nn.BatchNorm2d(128)
        self.up_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up_bn2_2 = nn.BatchNorm2d(128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up_bn3_1 = nn.BatchNorm2d(64)
        self.up_conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.up_bn3_2 = nn.BatchNorm2d(64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_conv4_1 = nn.Conv2d(128, 64, 7, padding=3)
        self.up_bn4_1 = nn.BatchNorm2d(64)
        self.up_conv4_2 = nn.Conv2d(64, 64, 7, padding=3)
        self.up_bn4_2 = nn.BatchNorm2d(64)
        
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_conv5_1 = nn.Conv2d(32, 16, 3, padding=1)
        self.up_bn5_1 = nn.BatchNorm2d(16)
        self.up_conv5_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.up_bn5_2 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        # Stage 0
        x0 = self.conv0(x)
        x0 = F.relu(self.bn0(x0))
        x0_pool = self.pool0(x0)
        
        # Stage 1
        x1 = self.res1_1(x0_pool)
        x1 = self.res1_2(x1)
        x1 = self.res1_3(x1)
        
        # Stage 2
        x2 = self.down2(x1)
        x2 = self.res2_1(x2)
        x2 = self.res2_2(x2)
        x2 = self.res2_3(x2)
        x2 = self.res2_4(x2)
        
        # Stage 3
        x3 = self.down3(x2)
        x3 = self.res3_1(x3)
        x3 = self.res3_2(x3)
        x3 = self.res3_3(x3)
        x3 = self.res3_4(x3)
        x3 = self.res3_5(x3)
        x3 = self.res3_6(x3)
        
        # Stage 4
        x4 = self.down4(x3)
        x4 = self.res4_1(x4)
        x4 = self.res4_2(x4)
        x4 = self.res4_3(x4)
        
        # Upsample
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up_conv1_1(u1)
        u1 = F.relu(self.up_bn1_1(u1))
        u1 = self.up_conv1_2(u1)
        u1 = F.relu(self.up_bn1_2(u1))
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up_conv2_1(u2)
        u2 = F.relu(self.up_bn2_1(u2))
        u2 = self.up_conv2_2(u2)
        u2 = F.relu(self.up_bn2_2(u2))
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.up_conv3_1(u3)
        u3 = F.relu(self.up_bn3_1(u3))
        u3 = self.up_conv3_2(u3)
        u3 = F.relu(self.up_bn3_2(u3))
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x0], dim=1)
        u4 = self.up_conv4_1(u4)
        u4 = F.relu(self.up_bn4_1(u4))
        u4 = self.up_conv4_2(u4)
        u4 = F.relu(self.up_bn4_2(u4))
        
        u5 = self.up5(u4)
        u5 = self.up_conv5_1(u5)
        u5 = F.relu(self.up_bn5_1(u5))
        u5 = self.up_conv5_2(u5)
        u5 = F.relu(self.up_bn5_2(u5))
        
        out = torch.sigmoid(self.final(u5))
        
        return out