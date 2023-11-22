import tensorflow as tf
import math
from tensorflow.keras import backend
from keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, Multiply,
                          Reshape, multiply, AveragePooling2D, Permute, Lambda, DepthwiseConv2D,
                          Conv1D, Conv2D, BatchNormalization, ReLU)
from tensorflow.keras.initializers import RandomNormal

def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def ECBAM(ecbam_feature):
#########################################################
###Channel

    channel = ecbam_feature.shape[-1]
    kernel_size = int(abs((math.log(channel, 2) + 1) / 2))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
     
    avg_pool = GlobalAveragePooling2D()(ecbam_feature)    
    max_pool = GlobalMaxPooling2D()(ecbam_feature)
    
    x = Reshape((-1,1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False,)(x)
    x = Activation(hard_sigmoid)(x)  #hard_sigmoid   'sigmoid'
    x = Reshape((1, 1, -1))(x)
    output1 = multiply([ecbam_feature,x])

    y = Reshape((-1,1))(max_pool)
    y = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False,)(y)
    y = Activation(hard_sigmoid)(x)   #hard_sigmoid   'sigmoid'
    y = Reshape((1, 1, -1))(x)
    output2 = multiply([ecbam_feature,y])
    
    out_feature = Add()([output1,output2])
    out_feature = Activation(hard_sigmoid)(out_feature)  
    out_feature = multiply([ecbam_feature, out_feature])
###########################################################
###Spatial
    kernel_size = 7
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(out_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(out_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    ecbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					)(concat)	
    ecbam_feature = Activation(hard_sigmoid)(ecbam_feature)
		
    return multiply([out_feature, ecbam_feature])
    
