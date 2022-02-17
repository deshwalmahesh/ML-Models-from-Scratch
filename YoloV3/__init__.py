import torch
import torch.nn as nn


class MishActivation(nn.module):
    '''
    Old versions don't have nn.Mish(). Users of Yolov3 did not use Mish as it did not exist back then
    '''
    def __init__(self):
        super(MishActivation, self).__init__()

    def forward(self, x):
        return x * torch.tanh((torch.nn.functional.softplus(x))) # nn.Tanh is a class also torch.nn.functional.tanh has been deprecated


class ConvolutionBlock(nn.module):
    '''
    Implement a block of Convolution consistes of Convolution Layer(s), BatchNormalization(optional)
    https://res.cloudinary.com/practicaldev/image/fetch/s--5kVLEyT3--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zdmk2adlckbnm8k9n0p8.png
    '''
    def __init__(self, in_channels:int, out_channels:int, use_batchnorm_activation:bool = True, activation = nn.LeakyRelu(0.1), **kwargs):
        super().__init__()
        self.use_batchnorm_activation = use_batchnorm_activation
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not use_batchnorm_activation **kwargs) # if using Batch Normalization and Activation Function, don't use Bias
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        

    def forward(self, x):
        '''
        Forward Pass Logic on input X
        '''
        return self.activation(self.batchnorm(self.conv(x))) if self.use_batchnorm_activation else self.conv(x)


class ResidualBlock(nn.module):
    '''
    Use the residual Connections. It might repeat 1 to N number of times. In YoloV3, N is usually 1, 2,4,8 at different times. 
    Also the auther did not use the residual connection at all
    https://res.cloudinary.com/practicaldev/image/fetch/s--5kVLEyT3--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zdmk2adlckbnm8k9n0p8.png
    '''
    def __init__(self, input_channels:int, repeat:int = 1, use_residual_connection:bool = True):
        super(ResidualBlock,self).__init__()
        self.use_residual = use_residual_connection
        self.all_layers = nn.ModuleList() # Contains the N number of layers inside this dynamically where each layer consists of 2 Convolution Blocks

        for _ in range(repeat):
            self.all_layers += [nn.Sequential(
                ConvolutionBlock(input_channels, input_channels//2, kernel_size = 1, ), # 1st Convolution Layer has 0 padding with 1*1 Kernal. output size is according to what the original paper has used
                ConvolutionBlock(input_channels//2, input_channels, kernel_size = 3, padding = 1), # Second Convolution Layer has 3*3 Kernal and padding of 1
            )]   

    def forward(self, x):
        '''
        'all_layers' is a list (length 1-8 typically) of layer where each 'layer' in turn is a group of 2 Convolution blocks. 
        if the residual is activated, whatever input that group of 2 Convolution got, it'll be added to their output
        '''
        for layer in self.all_layers: 
            x = layer(x) + x if self.use_residual else layer(x) # Either pass the input directly from those 2 layers || do the same + add the input too
        
        return x

    
class BranchScaledPredictions(nn.module):
    '''
    There are 3 branches where the prediction is done for small, medium and large size objects. Image comes to this Layers will be passed to
    1. A Convolution layer with kernel size 3*3, a padding of 1 and output channels as twice the number of channels
    2. Input from the above step will be taken and will be passed to a kernel of 1*1, without Batch Normalization and Activation to get an output of size (No of classes + 5) *3
        Each prediction has 5 elements (x,y,w,h,class) and 3 anchor Boxes for each prediction

    https://miro.medium.com/max/1838/1*vDfofpXSxcJ_zVITxCQDlA.png

    So output will be one of the 3
        1. Images in batch, 3, 13 (width), 13(height), Number of classes + 5
        2. Images in batch, 3, 26 (width), 26(height), Number of classes + 5
        3. Images in batch, 3, 52 (width), 52(height), Number of classes + 5

    '''
    def __init__(self, input_channels:int, classes:int):
        super().__init__()
        self.classes = classes
        self.first_layer = ConvolutionBlock(input_channels, input_channels*2, kernel_size = 3, padding  = 1) # it's output will be input to the next layer
        self.second_layer = ConvolutionBlock(input_channels*2, 3 * (classes +5) , use_batchnorm_activation = False, kernel_size = 1)


    def forward(self,x):
        '''
        Pass the input feature in sequential manner from 2 layers and then get output. Output is based on different scales of [width,height] as 13*3, 26*26, 52*2
        Reshape the output:
            1. Reshape the output as [No of image in the batch, 3 anchor BOXES per cell, 5 * classes numbers for each BOX, width, height]
            2. Shift / PErmute / roll the axis as: [No of image in the batch, 3 anchor boxes, width, height, 5 predictions per class]
        
        https://www.atlantis-press.com/assets/articles/IJCIS-13-1-1153/IJCIS-13-1-1153-g001.png
        '''
        output = self.first_layer(x)
        output = self.second_layer(output)
        output = output.reshape(output.shape[0], 3, 5*self.classes, output.shape[2], output.shape[3])
        output = output.permute(0,1,3,4,2) # change the axis or shifting so it has shape of 
        return output





        


