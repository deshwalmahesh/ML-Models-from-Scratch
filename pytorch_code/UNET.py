'''
Whole UNet is divided in 3 parts: Encoder -> BottleNeck -> Decoder. There are skip connections between 'Nth' level of Encoder with Nth level of Decoder.

There is 1 basic entity called "Convolution" Block which has 3*3 Convolution (or Transposed Convolution during Upsampling) -> ReLu -> BatchNorm
Then there is Maxpooling
'''

import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    '''
    The basic Convolution Block Which Will have Convolution -> RelU -> Convolution -> RelU
    '''
    def __init__(self, input_features, out_features):
        '''
        args:
            batch_norm was introduced after UNET so they did not know if it existed. Might be useful
        '''
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_features, out_features, kernel_size = 3, padding= 0), # padding is 0 by default, 1 means the input width, height == out width, height
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 0),
            nn.ReLU(),
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
        repeat = len(blockwise_features) # how many layers we need to add len of blockwise_features == len of out_features

        self.layers = nn.ModuleList()
        
        for i in range(repeat):
            if i == 0:
                in_filters = image_channels
                out_filters = blockwise_features[0]
            else:
                in_filters = blockwise_features[i-1]
                out_filters = blockwise_features[i]
            
            self.layers.append(ConvolutionBlock(in_filters, out_filters))

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)  # Since There is No gradient for Maxpooling, You can instantiate a single layer for the whole operation
        # https://datascience.stackexchange.com/questions/11699/backprop-through-max-pooling-layers
        
    
    def forward(self, feature_map_x):
        skip_connections = [] # i_th level of features from Encoder will be conatenated with i_th level of decoder before applying CNN
        
        for layer in self.layers:
            feature_map_x = layer(feature_map_x)
            skip_connections.append(feature_map_x)
            feature_map_x = self.maxpool(feature_map_x) # Use Max Pooling AFTER storing the Skip connections

        return feature_map_x, skip_connections

    
class BottleNeck(nn.Module):
    '''
    ConvolutionBlock without Max Pooling
    '''
    def __init__(self, input_features = 512, output_features = 1024):
        super().__init__()
        self.layer = ConvolutionBlock(input_features, output_features)

        
    def forward(self, feature_map_x):
        return self.layer(feature_map_x)
        

class Decoder(nn.Module):
    '''
    '''
    def __init__(self, blockwise_features = [512, 256, 128, 64]):
        '''
        Do exactly opposite of Encoder
        '''
        super().__init__()

        self.upsample_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        for i, feature in enumerate(blockwise_features):

            self.upsample_layers.append(nn.ConvTranspose2d(in_channels = feature*2, out_channels = feature, kernel_size = 2, stride = 2))  # Takes in 1024-> 512, takes 512->254 ......
            self.conv_layers.append(ConvolutionBlock(feature*2, feature)) # After Concatinating (512 + 512-> 1024), Use double Conv block

    

    def pad_or_crop(self, larger_feature, smaller_feature, kind = 'pad'):
        '''
        args:
            larger_feature: In UNET, it is the Skip Connection Feature
            smaller_feature: It is upsampled feature

        Either 'pad' the smaller feature or 'crop' larger feature
        Features MUST be square in size (width == height)
        '''
        assert kind in ['pad','crop'], "kind must be one of ['pad','crop']"
        delta = (larger_feature.shape[-1] - smaller_feature.shape[-1])// 2 # how much to pad or crop on EACH side (there are 4 sides)
        
        if kind == 'pad':
            return larger_feature, nn.functional.pad(smaller_feature, pad = (delta, delta, delta, delta, 0,0, 0,0)) #pad (height,height,width,width, channel,channel, batch,batch)
        
        else: # in case or cropping
            size = larger_feature.shape[-1] # because width == height
            return larger_feature[:,:,delta:size-delta, delta:size-delta], smaller_feature # crop the pixels from the boundries from each side of width and height
        

    def forward(self, feature_map_x, skip_connections, kind = 'crop'):
        '''
        args:
            kind: When Concatinating, whether to perform Cropping or Padding

        Repeat Steps as:
        1. Upsample
        2. Make the two features (skip and current) compatible in height and width
        3. Concat Skip Connection
        4. Apply ConvolutionBlock
        '''

        for i, layer in enumerate(self.conv_layers): # 4 levels, 4 skip connections, 4 upsampling, 4 Double Conv Block
            skip_feature = skip_connections[-i-1]
            feature_map_x = self.upsample_layers[i](feature_map_x) # step 1

            if skip_feature.shape[-1] != feature_map_x.shape[-1]: # step 2 The part of CROP / PAD to make the dimensions equal
                skip_feature, feature_map_x = self.pad_or_crop(skip_feature, feature_map_x, kind = kind)

            feature_map_x = torch.cat((skip_feature, feature_map_x), dim = 1) # step 3, Concatinating along Channels or Features dimensions given [N,C,W,H]
            feature_map_x = self.conv_layers[i](feature_map_x) # Step 4

        return feature_map_x


class UNET(nn.Module):
    '''
    Complete UNET Module
    '''
    def __init__(self, image_channels, num_classes, features_list = [64, 128, 256, 512]):
        '''
        Perform the whole 4 part step
        1. Down Scaling
        2. Bottleneck
        3. Up Scaling
        4. Prediction using 1x1 Convolution

        args:
            image_channels: any of 1 or 3 for Grayscale or RGB images
            num_classes: Num of classes you want to predict. In Original UNET, 2 (black and white) were predicted
        '''
        super().__init__()

        self.encoder = Encoder(image_channels, features_list)
        self.bottleneck = BottleNeck(input_features = features_list[-1], output_features = features_list[-1]*2) # In-> 512, out-> 1024
        self.decoder = Decoder(features_list[::-1]) # Reverse of Encoder features
        # Apply 1x1 convolution
        self.final_layer = nn.Conv2d(in_channels = features_list[0], out_channels = num_classes, kernel_size = 1) # Decoder will output 64 features  which will serve as input to this


    def forward(self, image_batch):
        '''
        '''
        features, skip_connections = self.encoder(image_batch)
        features = self.bottleneck(features)
        features = self.decoder(features, skip_connections)
        return self.final_layer(features)


def test_unet_code():
    image = torch.randn(1,1, 572,572)
    out = UNET(image_channels=1, num_classes=2)(image)
    print(f"Input Shape: {image.shape} || Output Shape: {out.shape}")
    print("Code Running Perfectly")