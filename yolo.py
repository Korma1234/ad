from functools import wraps
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from utils.utils import compose
from tensorflow.keras.layers import (Activation, BatchNormalization, Multiply, Add, GlobalAveragePooling2D, Layer,
                                     Concatenate, Conv2D, DepthwiseConv2D, LeakyReLU,Reshape, ZeroPadding2D,
                                     Input, Lambda, MaxPooling2D, UpSampling2D, Conv1D, multiply)
from nets.CSPdarknet import (CSPLayer, DarknetConv2D_BN_SiLU, darknet_body)
from nets.yolo_training import get_yolo_loss
from nets.attention import cbam_block, eca_block, se_block, CoordAtt, ECBAM
import math
from tensorflow.keras import backend


##########################################################################
def relu6(x):
    return K.relu(x, max_value=6)
def hard_sigmoid(x):
    return K.relu(x + 3.0, max_value=6.0) / 6.0
def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
###Ghost
#################################################################################
def slices(dw, n):
    return dw[:,:,:,:n]

def _ghost_module(inputs, exp, ratio, kernel_size=1, dw_size=3, stride=1):
    #--------------------------------------------------------------------------#
    #   ratio一般会指定成2
    #   这样才可以保证输出特征层的通道数，等于exp
    #--------------------------------------------------------------------------#
    output_channels = math.ceil(exp * 1.0 / ratio)

    #--------------------------------------------------------------------------#
    #   利用1x1卷积对我们输入进来的特征图进行一个通道的缩减，获得一个特征浓缩
    #   跨通道的特征提取
    #--------------------------------------------------------------------------#
    # x = res2net_bottleneck_block(inputs, output_channels, s=4, expansion=4, use_se_block=False)
    x = Conv2D(output_channels, kernel_size, strides=stride, padding="same", use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(inputs)
    x = BatchNormalization()(x)
    # x = Activation(hard_swish)(x)  #Activation(hard_swish)  relu
    x = SiLU()(x)
    #--------------------------------------------------------------------------#
    #   在获得特征浓缩之后，使用逐层卷积，获得额外的特征图
    #   跨特征点的特征提取
    #--------------------------------------------------------------------------#
    dw = DepthwiseConv2D(dw_size, 1, padding="same", depth_multiplier=ratio-1, use_bias=False, depthwise_initializer=RandomNormal(stddev=0.02))(x)
    dw = BatchNormalization()(dw)
    # dw = Activation(hard_swish)(dw)
    x = SiLU()(x)

    #--------------------------------------------------------------------------#
    #   将1x1卷积后的结果，和逐层卷积后的结果进行堆叠
    #--------------------------------------------------------------------------#
    x = Concatenate(axis=-1)([x, dw])
    x = Lambda(slices, arguments={'n':exp})(x)
    
    n,h,w,c = x.get_shape()
    x_reshape = tf.reshape(x,[-1, h, w, 2, c//2])
    x_transpose = tf.transpose(x_reshape,[0,1,2,4,3])
    x = tf.reshape(x_transpose,[-1,h,w,c])
    return x

def _ghost_bottleneck(inputs, output_channel, hidden_channel, kernel, ratio, strides, squeeze):
    input_shape = backend.int_shape(inputs)

    #--------------------------------------------------------------------------#
    #   首先利用一个ghost模块进行特征提取
    #   此时指定的通道数会比较大，可以看作是逆残差结构
    #--------------------------------------------------------------------------#
    x = _ghost_module(inputs, hidden_channel, ratio)

    if strides > 1:
        #--------------------------------------------------------------------------#
        #   如果想要进行特征图的高宽压缩，则进行一个逐层卷积
        #--------------------------------------------------------------------------#
        x = DepthwiseConv2D(kernel, strides, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization()(x)

    if squeeze:
        x = ECBAM(x)

    #--------------------------------------------------------------------------#
    #   再次利用一个ghost模块进行特征提取
    #--------------------------------------------------------------------------#
    x = _ghost_module(x, output_channel, ratio)

    if strides == 1 and input_shape[-1] == output_channel:
        res = inputs
    else:
        res = DepthwiseConv2D(kernel, strides=strides, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=RandomNormal(stddev=0.02))(inputs)
        res = BatchNormalization()(res)
        res = Conv2D(output_channel, (1, 1), padding='same', strides=(1, 1), use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(res)
        res = BatchNormalization()(res)
    x = Add()([res, x])
    return x
#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, num_classes, weight_decay=5e-4):
   
    input_shape = [416,416,3]
    
    inputs      = Input(input_shape)
    #---------------------------------------------------#
    #   feat1 80, 80, 256
    #   feat2 40, 40, 512
    #   feat3 20, 20, 1024
    #---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)

    x = _ghost_module(feat3, exp=64, ratio=2)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    P5 = _ghost_module(x, exp=64, ratio=2)
    
    P5_upsample = UpSampling2D(2)(P5)
    
    P4 = _ghost_module(feat2, exp=64, ratio=2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = _ghost_bottleneck(P4, 48, 48, (3, 3), ratio=2, strides=1, squeeze=True)
  
    P4_upsample = UpSampling2D(2)(P4)

    P3 = _ghost_module(feat1, exp=64, ratio=2)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = _ghost_bottleneck(P3, 48, 48, (3, 3), ratio=2, strides=1, squeeze=True)
    
    # #---------------------------------------------------#
    # #   第三个特征层
    # #   y3=(batch_size,52,52,3,85)
    # #---------------------------------------------------#
    P3_out = _ghost_module(P3, exp=64, ratio=2)
    
    P3_downsample = Conv2D(64, (1,1), strides=(2,2))(P3_out)
    P4 = Concatenate()([P3_downsample, P4])

    # #---------------------------------------------------#
    # #   第二个特征层
    # #   y2=(batch_size,26,26,3,85)
    # #---------------------------------------------------#
    P4_out = _ghost_module(P4, exp=64, ratio=2)

    P4_downsample = Conv2D(64, (1,1), strides=(2,2))(P4_out)
    P5 = Concatenate()([P4_downsample, P5])

    # #---------------------------------------------------#
    # #   第一个特征层
    # #   y1=(batch_size,13,13,3,85)
    # #---------------------------------------------------#
    P5_out = _ghost_module(P5, exp=64, ratio=2)
    
    
    fpn_outs    = [P3_out, P4_out, P5_out]
    yolo_outs   = []
    for i, out in enumerate(fpn_outs):
        # 利用1x1卷积进行通道整合
        stem = _ghost_module(out, exp=96,ratio=2)
        cls_conv = _ghost_bottleneck(stem, 96, 96, (3, 3), ratio=2, strides=1, squeeze=False)
        
        
        # stem    = DarknetConv2D_BN_SiLU(int(64), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.stems.' + str(i))(out)
        # 利用3x3卷积进行特征提取
        # cls_conv = DarknetConv2D_BN_SiLU(int(64), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.0')(stem)
        # cls_conv = DarknetConv2D_BN_SiLU(int(64), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.1')(cls_conv)
        #---------------------------------------------------#
        #   判断特征点所属的种类
        #   80, 80, num_classes
        #   40, 40, num_classes
        #   20, 20, num_classes
        #---------------------------------------------------#
        cls_pred = DarknetConv2D(num_classes, (1, 1), strides = (1, 1), name = 'head.cls_preds.' + str(i))(cls_conv)

        reg_conv = _ghost_bottleneck(stem, 96, 96, (3, 3), ratio=2, strides=1, squeeze=False)
        

        # 利用3x3卷积进行特征提取
        # reg_conv = DarknetConv2D_BN_SiLU(int(64), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.0')(stem)
        # reg_conv = DarknetConv2D_BN_SiLU(int(64), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.1')(reg_conv)
        #---------------------------------------------------#
        #   特征点的回归系数
        #   reg_pred 80, 80, 4
        #   reg_pred 40, 40, 4
        #   reg_pred 20, 20, 4
        #---------------------------------------------------#
        reg_pred = DarknetConv2D(4, (1, 1), strides = (1, 1), name = 'head.reg_preds.' + str(i))(reg_conv)
        #---------------------------------------------------#
        #   判断特征点是否有对应的物体
        #   obj_pred 80, 80, 1
        #   obj_pred 40, 40, 1
        #   obj_pred 20, 20, 1
        #---------------------------------------------------#
        obj_pred = DarknetConv2D(1, (1, 1), strides = (1, 1), name = 'head.obj_preds.' + str(i))(reg_conv)
        output   = Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)
    return Model(inputs, yolo_outs)

def get_train_model(model_body, input_shape, num_classes):
    y_true = [Input(shape = (None, 5))]
    model_loss  = Lambda(
        get_yolo_loss(input_shape, len(model_body.output), num_classes), 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
    )([*model_body.output, *y_true])
    
    model       = Model([model_body.input, *y_true], model_loss)
    return model
