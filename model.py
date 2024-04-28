import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from up import UpProject

class EnhancedUpSample(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        """Enhanced upsampling layer utilizing up-projection function"""
        super(EnhancedUpSample, self).__init__()
        self.up_proj = UpProject(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.up_proj(x, BN=True)

class EnhancedDecoder(nn.Sequential):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        """Enhanced decoder utilizing up-projection layers"""
        super(EnhancedDecoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = EnhancedUpSample(in_channels=features//1, out_channels=features//2)
        self.up2 = EnhancedUpSample(in_channels=features//2, out_channels=features//4)
        self.up3 = EnhancedUpSample(in_channels=features//4, out_channels=features//8)
        self.up4 = EnhancedUpSample(in_channels=features//8, out_channels=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_d0 = self.conv2(F.relu(features[12]))

        x_d1 = self.up1(x_d0)
        x_d2 = self.up2(x_d1)
        x_d3 = self.up3(x_d2)
        x_d4 = self.up4(x_d3)
        return self.conv3(x_d4)

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        """Standard upsampling layer utilizing bilinear upsampling"""
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        """Standard decoder utilizing bilinear upsampling layers and skip connections"""
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    """Encoder from DenseNet-169"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169( pretrained=True ) #TODO: changed from False

    def forward(self, x):
        features = [x]
        for _, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    """Depth-estimation model with encoder-decoder architecture"""
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = EnhancedDecoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

