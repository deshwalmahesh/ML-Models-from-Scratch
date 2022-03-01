from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50


def DilatedConvolutionBlock(input_features, kernel_size = 3, dilation_rate = 1, filters = 256, padding = 'same', use_bias = False):
    '''
    Takes in some features and perform the folloring operations: Dilated (Atrous) Contolution -> BatchNorm -> ReLu
    To read more about "SAME" Padding: read https://stackoverflow.com/questions/68035443/what-does-padding-same-exactly-mean-in-tensorflow-conv2d-is-it-minimum-paddin
    '''
    x = Convolution2D(filters, kernel_size, dilation_rate = dilation_rate, use_bias = use_bias, padding = padding)(input_features)
    x = BatchNormalization()(x)
    return ReLU()(x) # tf.nn.relu(x)


def Perform_ASPP(input_features):
    '''
    Get features From Backbone and Create Atrous Spatial Pyramid Pooling. 
    The features are first bilinearly upsampled by a factor 4, and then concatenated with the corresponding low-level features from the network backbone that have the same spatial resolution.

    Pay attention to the "ENCODER" part here: https://miro.medium.com/max/1037/1*2mYfKnsX1IqCCSItxpXSGA.png In this image, DCNN is any pre-trained architecture features
    '''
    width, height = input_features.shape[1], input_features.shape[2] # given Channels Last format

    one_cross_one = DilatedConvolutionBlock(input_features = input_features, kernel_size = 1, dilation_rate = 1,)
    rate_6 = DilatedConvolutionBlock(input_features = input_features, dilation_rate = 6,) # (default) 3x3 convolution with dilation rate of 6
    rate_12 = DilatedConvolutionBlock(input_features = input_features, dilation_rate = 12,)
    rate_18 = DilatedConvolutionBlock(input_features = input_features, dilation_rate = 18,)

    pooling_part = AveragePooling2D(pool_size = (width, height))(input_features)
    pooling_part = DilatedConvolutionBlock(pooling_part, kernel_size = 1, use_bias = True)
    pooling_part = UpSampling2D(size=(width, height), interpolation="bilinear")(pooling_part)
    # We could use Transposed Convolution for this, for difference, see: https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker

    concatenated_fetures = Concatenate(axis = -1)([pooling_part, one_cross_one, rate_6, rate_12, rate_18]) # Concatenate all features
    return DilatedConvolutionBlock(concatenated_fetures, kernel_size = 1, )
    

def EncoderPart(backbone, right_layer_name, down_layer_name):
    '''
    The dotted ENCODER part in the image: https://miro.medium.com/max/1037/1*2mYfKnsX1IqCCSItxpXSGA.png
    Takes an input as Image and generates two outputs:
    1. Features extracted from down_layer_name is directly sent to the Decoder (where 1x1 convolution will be performed later)
    2. Right Path: Feature which are computed from the ASPP

    args:
        backbone: Pre Trained Model
        right_layer_name: Layer name whose output will be used for ASPP. You can try with different layers and their outputs
        down_layer_name: Layer name whose output will be sent directly to the Decoder
    '''
    features = backbone.get_layer(right_layer_name).output # this is the layer whose output they have used as Features extraction. You could also use: backbone.layers[142].output
    right_path = Perform_ASPP(features)
    return backbone.get_layer(down_layer_name).output, right_path


def DecoderPart(down_path_features, right_path_features, image_shape):
    '''
    The dotted DECODER part in the image: https://miro.medium.com/max/1037/1*2mYfKnsX1IqCCSItxpXSGA.png
    Takes two inputs (down_path_features, right_path_features: See the DocString of Encoder to see what they are) and return an output
    '''
    first_upsampling_by_4 = UpSampling2D(size = (4, 4), interpolation = 'bilinear')(right_path_features)

    one_cross_one = DilatedConvolutionBlock(down_path_features, kernel_size = 1, dilation_rate = 1, filters = 48)

    concat = Concatenate(axis = -1)([first_upsampling_by_4, one_cross_one])

    x = DilatedConvolutionBlock(concat, kernel_size = 3, dilation_rate = 1) # default parameters are used
    x = DilatedConvolutionBlock(x, kernel_size = 3, dilation_rate = 1)

    second_upsampling_by_4 = UpSampling2D(size = (4,4), interpolation = 'bilinear')(x)
    return second_upsampling_by_4


def DeepLabV3Plus(num_classes, backbone = ResNet50, image_shape = (512,512,3), right_layer_name = "conv4_block6_2_relu", down_layer_name = "conv2_block3_2_relu"):
    '''
    Build DeepLabV3+ Architecture Model. Following Steps are Done:
    1. Fetch a pre trained Model and instantiate it
    2. Extract 2 different features (direct feature from one layer, features from another layer where ASPP is performed) from the encoder part
    3. Pass those features to Decoder to get final features
    4. Pass those final features to 1x1 convolution so that makss can be generated from the images

    args:
        num_classes: Total number of classes that have to be predicted
    '''
    assert (not image_shape[0] % 4) and (not image_shape[1] % 4), "Image Height and Width must be divisible by 4" 
    backbone = backbone(input_tensor = Input(image_shape),include_top=False, weights="imagenet") # Step 1

    down_path_features, right_path_features = EncoderPart(backbone, right_layer_name, down_layer_name) # step 2

    decoder_features = DecoderPart(down_path_features, right_path_features, image_shape) # step 3

    output = Conv2D(num_classes, kernel_size = 1, padding = 'same')(decoder_features) # step 4
    model = Model(inputs = backbone.input, outputs = output)
    return model


def test_tensorflow_DeepLabv3_code():
    print("Building Model with 20 final classes. Find the summary below....")
    print(DeepLabV3Plus(20).summary())