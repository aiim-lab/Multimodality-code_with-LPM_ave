import numpy as np
import theano
import theano.tensor as T

from keras.layers.core import Layer
import tensorflow as tf

floatX = theano.config.floatX
from keras import backend as K
from keras.layers import Input, merge, Lambda, LeakyReLU, MaxPooling2D,concatenate, Concatenate, maximum,average,add,Reshape, Multiply, Add

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    """
    def __init__(self, localization_net, downsample_factor=1, return_theta=False, **kwargs):
        self.downsample_factor = downsample_factor
        self.locnet = localization_net
        self.return_theta = return_theta
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, something):
        print something
        print self.locnet.input.type()
        if hasattr(self, 'previous'):
            self.locnet.set_previous(self.previous)
        self.locnet.build()
        self.trainable_weights = self.locnet.trainable_weights
        self.regularizers = self.locnet.regularizers
        self.constraints = self.locnet.constraints
        self.input = self.locnet.input  # This must be T.tensor4()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (None, input_shape[1],
                int(input_shape[2] / self.downsample_factor),
                int(input_shape[3] / self.downsample_factor))

    def get_output(self, train=False):
        X = self.get_input(train)
        theta = apply_model(self.locnet, X)
        theta = theta.reshape((X.shape[0], 2, 3))
        output = self._transform(theta, X, self.downsample_factor)

        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output

    @staticmethod
    def _repeat(x, n_repeats):
        # rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
        # rep_t = K.ones((n_repeats,), dtype='int32')
        rep_t = K.ones((n_repeats.eval(session= K.get_session()),))
        # rep_t1= K.expand_dims(x,0)
        rep_t1= K.expand_dims(rep_t,0)
        #casting rep to int32 so that next line can be run
        rep= K.cast(rep_t1, dtype= 'int32')
        
        x = K.dot(K.reshape(x,(-1, 1)), rep)
        return K.flatten(x)


    @staticmethod
    def _interpolate(im, x, y, downsample_factor):
        # constants
        num_batch, height, width, channels = im.shape
        
        # height_f = T.cast(height, floatX)
        height_f = K.cast_to_floatx(height.value)
        # width_f = T.cast(width, floatX)
        width_f= K.cast_to_floatx(width.value)
        # out_height = T.cast(height_f // downsample_factor, 'int64')
        out_height = K.cast(height_f // downsample_factor, 'int32')
        out_width = K.cast(width_f // downsample_factor, 'int32')
        # zero = K.zeros([], dtype='int64')
        zero = K.zeros([])
        #casting to int64 so that zero can be used further in the function
        zero_t= K.cast(zero, 'int32')

 
        max_y = K.cast(im.shape[1] - 1, 'int32')
        max_x = K.cast(im.shape[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        # x0 = K.cast(T.floor(x), 'int64')
        x0 = K.cast(K.round(x), 'int32')
        x1 = x0 + 1
        y0 = K.cast(K.round(y), 'int32')
        y1 = y0 + 1

        #Converting zero,max_x, max_y to float/int(pythonic, not tensor) so that it can be used in  K.clip
        zero= zero_t.eval(session=K.get_session())
        max_xt= max_x.eval(session=K.get_session())
        max_yt= max_y.eval(session=K.get_session())

        x0 = K.clip(x0, zero, max_xt)
        x1 = K.clip(x1, zero, max_xt)
        y0 = K.clip(y0, zero, max_yt)
        y1 = K.clip(y1, zero, max_yt)
        dim2 = width
        dim1 = width*height
        # base = SpatialTransformer._repeat(
            # K.arange(num_batch, dtype='int32')*dim1, out_height*out_width)
        base = SpatialTransformer._repeat(
            K.arange(-1, dtype='int32')*dim1, out_height*out_width)

        #from https://github.com/oarriaga/STN.keras/blob/master/src/models/layers.py
        # batch_size = K.shape(im)[0]
        # height = K.shape(im)[1]
        # width = K.shape(im)[2]
        # num_channels = K.shape(im)[3]

        # pixels_batch= K.arange(0,batch_size)*(height*width)
        # pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        # flat_output_size = y[0] * y[1]
        # # flat_output_size = 30 * 30
        # base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        # base= K.flatten(base)

        #base= int 32 and y0= int64, hence casting base to int 64
        base= K.cast(base, 'int32')
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat
        #  image and restore channels dim
        # im_flat = im.reshape((-1, channels))
        im_flat = K.reshape(im,(-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finanly calculate interpolated values
        x0_f = T.cast(x0, floatX)
        x1_f = T.cast(x1, floatX)
        y0_f = T.cast(y0, floatX)
        y1_f = T.cast(y1, floatX)
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        return output

    @staticmethod
    def _linspace(start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        start = K.cast_to_floatx(start)
        stop = K.cast_to_floatx(stop)
        #converting num to numpy array because K.cast_to_floatx needs numpy array as an input
        num_array= num.eval(session=K.get_session())

        num = K.cast_to_floatx(num_array)
        step = (stop-start)/(num-1)
        # return K.arange(num, dtype=floatx)*step+start
        return K.arange(num,dtype= 'float32')*step+start

    @staticmethod
    def _meshgrid(height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        # x_t = K.dot(K.ones((height, 1)),
        #             K.permute_dimensions(SpatialTransformer._linspace(-1.0, 1.0, width),('x', 0)))
        temp = SpatialTransformer._linspace(-1.0, 1.0, width)
        # temp = K.expand_dims(temp,1)
        temp0= K.expand_dims(temp,0 )
        temp0= K.cast(temp0, dtype='float32')
        # x_t = K.dot(temp,K.ones((1,K.get_value(height))))
        # K.set_floatx('float64')
        ones= K.ones((K.get_value(height),1))
        x_t = K.dot(ones, temp0)

        temp1= SpatialTransformer._linspace(-1.0, 1.0, height)
        temp1= K.cast(temp1, dtype='float32')
        temp2 = K.expand_dims(temp1,1)
        temp2= K.cast(temp2, dtype='float32')
        oness= K.ones((1,K.get_value(width)))
        y_t = K.dot(temp2,oness)

        


        # x_t_flat = x_t.reshape((1, -1))
        x_t_flat = K.reshape(x_t,(1, -1))
        # y_t_flat = y_t.reshape((1, -1))
        y_t_flat = K.reshape(y_t,(1, -1))
        ones = K.ones_like(x_t_flat)
        grid = K.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    @staticmethod
    def _transform(theta, input, downsample_factor):
        num_batch, num_channels, height, width = input.shape
        # num_batch= K.shape(input)[0]
        # num_channels= K.shape(input)[1]
        # height= K.shape(input)[2]
        # width= K.shape(input)[3]

        # K.set_floatx('float64')
        theta = K.reshape(theta, (-1, 2, 3))  # T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        # height_f = T.cast(height, floatX)
        #reshaping height to scalar as i was getting an error
        # tf_session= K.get_session()
        # height=K.shape(height).eval(session= tf_session)
        # height=K.eval(session= tf_session)
        height_f = K.cast_to_floatx(height.value)
        # width_f = T.cast(width, floatX)
        width_f = K.cast_to_floatx(width.value)
        # out_height = T.cast(height_f // downsample_factor, 'int64')
        out_height = K.cast(height_f//downsample_factor, "int32")
        print "this is sofjsljs {}".format(out_height)
        out_width = K.cast(width_f // downsample_factor, 'int32')
        grid = SpatialTransformer._meshgrid(out_height, out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        theta= K.cast(theta,'float32')
        T_g = K.dot(theta, grid)
        x_s, y_s = T_g[:, 0], T_g[:, 1]
        # x_s_flat = x_s.flatten()
        x_s_flat = K.flatten(x_s)
        # y_s_flat = y_s.flatten()
        y_s_flat = K.flatten(y_s)

        # dimshuffle input to  (bs, height, width, channels)
        # input_dim = input.dimshuffle(0, 2, 3, 1)
        input_dim = K.permute_dimensions(input,(0, 2, 3, 1))

        input_transformed = SpatialTransformer._interpolate(
            input_dim, x_s_flat, y_s_flat,
            downsample_factor)

        output = T.reshape(input_transformed,
                           (num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)
        return output