'''
Whole UNet is divided in 3 parts: Endoder -> BottleNeck -> Decoder. There are skip connections between 'Nth' level of Encoder with Nth level of Decoder.

There is 1 basic entity called "Convolution" Block which has 3*3 Convolution (or Transposed Convolution during Upsampling) -> ReLu -> BatchNorm
Then there is Maxpooling
'''

import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    '''
    The basic Convolution Block Which Will have Convolution -> RelU -> Convolution -> RelU
    '''
    def __init__(self, blockwise_features, out_features, upsample:bool = False,):
        '''
        args:
            upsample: If True, then use TransposedConv2D (Means it being used in the decoder part) instead MaxPooling 
            batch_norm was introduced after UNET so they did not know if it existed. Might be useful
        '''
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(blockwise_features, out_features, kernel_size = 3, padding= 1), # padding is 0 by default, 1 means the input width, height == out width, height
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)  if not upsample else nn.ConvTranspose2d(out_features, out_features//2, kernel_size = 2, )  # As it is said in the paper that it TransPose2D halves the features 
        )

    def forward(self, feature_map_x):
        '''
        feature_map_x could be the image itself or the
        '''
        return self.network(feature_map_x)


class Encoder(nn.Module):
    '''
    '''
    def __init__(self, image_channels:int = 3, blockwise_features = [64, 128, 256, 512]):
        '''
        In UNET, the features start at 64 and keeps getting twice the size of the previous one till it reached BottleNeck
        args:
            image_channels: Channels in the Input Image. Typically it is any of the 1 or 3 (rarely 4)
            blockwise_features = Each block has it's own input and output features. it means first ConV block will output 64 features, second 128 and so on
        '''
        super().__init__()
        # blockwise_features = [image_channels, 64, 128, 256, 512]
        # out_features = [64, 128, 256, 512, 1024] 

        # Below code is just a logic to create N layers dynamically for the above 2 commented lines. Can be done in multiple different ways. Can be hardcoded
        out_features = blockwise_features.copy() # because we need to pop() the element and without copy, it is just aliasing in OOP
        blockwise_features.insert(0, image_channels) # Add the first element as the channels for the image
        out_features.append(blockwise_features[-1]*2)

        repeat = len(blockwise_features) # how many layers we need to add len of blockwise_features == len of out_features


        self.layers = nn.ModuleList(
            [ConvolutionBlock(blockwise_features = blockwise_features[i], out_features = out_features[i]) for i in range(repeat)]
        )
    
    def forward(self, feature_map_x):
        for layer in self.layers:
            feature_map_x = layer(feature_map_x)
        return feature_map_x


class Decoder(nn.Module):
    '''
    '''
    def __init__(self, output_classes:int = 2, blockwise_features = [1024, 512, 256, 128, 64]):
        '''
        Do exactly opposite of Encoder
        args:
            output_classes: If you want to segment B&W (foreground vs background), it'll be 2, if [man, car, background]: it'll be 3 and so on
        '''
        super().__init__()

        out_features = blockwise_features[1:] # everything except the first value
        out_features.append(output_classes)

        repeat = len(blockwise_features)

        self.layers = nn.ModuleList(
            [ConvolutionBlock(blockwise_features = blockwise_features[i], out_features = out_features[i], upsample = True) for i in range(repeat)]
        )
    
    def forward(self, feature_map_x):
        for layer in self.layers:
            feature_map_x = layer(feature_map_x)
        return feature_map_x

