import tensorflow as tf
import tensorflow.contrib.slim as slim

# Necessary bulding blocks for constructing U-Net ---------------
# The design only uses low-level tf.slim operation, e.g. conv2d, max_pool to simplify the code.

KEEPPROB = 0.5

def doulbe_conv2d_bn_relu(input, num_out, filter_size):
  '''
  Two convolution layers with batch norm, relu activation and dropout
  :param input: 4D image tensor
  :param num_out: number of output filters (usually double the input number of filter)
  :param filter_size: filter size, normally set as (3,3)
  :return: image tensor after operation
  '''
  input = slim.conv2d(input, num_out, (filter_size, filter_size),
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True})
  input = slim.dropout(input, KEEPPROB)

  input = slim.conv2d(input, num_out, (filter_size, filter_size),
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True})
  input = slim.dropout(input, KEEPPROB)
  return input

def up_conv(input, num_out, filter_size):
  '''
  First upsample the image (double the image size) and then apply convolution
  :param input: 4D image tensor
  :param num_out: number of output filters (usually half of input number of filter)
  :param filter_size: filter size, normally set as (2,2)
  :return: image tensor after operation
  '''
  [_, h, w, _] = input.shape.as_list()
  input = tf.image.resize_images(input, [h*2, w*2])
  input = slim.conv2d(input, num_out, (filter_size, filter_size),
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True})
  input = slim.dropout(input, KEEPPROB)
  return input

# U-Net Defintion -------------------------------------------------------------
def unet(img_tensor):
  '''
  Modified U-Net graph construction.
  :param img_tensor: batch of image data
  :return: output prediction, packed intermediate layers
  '''
  # 320
  down0 = doulbe_conv2d_bn_relu(img_tensor, 32, 3)
  down0_pool = slim.max_pool2d(down0, [2, 2], scope='down0_pool')

  # 160
  down1 = doulbe_conv2d_bn_relu(down0_pool, 64, 3)
  down1_pool = slim.max_pool2d(down1, [2, 2], scope='down1_pool')

  # 80
  down2 = doulbe_conv2d_bn_relu(down1_pool, 128, 3)
  down2_pool = slim.max_pool2d(down2, [2, 2], scope='down2_pool')

  # 40
  down3 = doulbe_conv2d_bn_relu(down2_pool, 256, 3)
  down3_pool = slim.max_pool2d(down3, [2, 2], scope='down3_pool')

  # 20
  down4 = doulbe_conv2d_bn_relu(down3_pool, 512, 3)
  down4_pool = slim.max_pool2d(down4, [2, 2], scope='down4_pool')

  # 10
  center = doulbe_conv2d_bn_relu(down4_pool, 1024, 3)

  # 20
  up4 = up_conv(center, 512, 2)
  up4 = tf.concat([down4, up4], axis=3)
  up4 = doulbe_conv2d_bn_relu(up4, 512, 3)

  # 40
  up3 = up_conv(up4, 256, 2)
  up3 = tf.concat([down3, up3], axis=3)
  up3 = doulbe_conv2d_bn_relu(up3, 256, 3)

  # 80
  up2 = up_conv(up3, 128, 2)
  up2 = tf.concat([down2, up2], axis=3)
  up2 = doulbe_conv2d_bn_relu(up2, 128, 3)

  # 160
  up1 = up_conv(up2, 64, 2)
  up1 = tf.concat([down1, up1], axis=3)
  up1 = doulbe_conv2d_bn_relu(up1, 64, 3)

  # 320
  up0 = up_conv(up1, 32, 2)
  up0 = tf.concat([down0, up0], axis=3)
  up0 = doulbe_conv2d_bn_relu(up0, 32, 3)

  # final output
  output = slim.conv2d(up0, 3, (1,1),
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True},
                      scope='output')
  output = tf.nn.softmax(output)

  # pack intermediate variables for future debugging
  packed_interlayer = [down0, down1, down2, down3, up3, up2, up1, up0]

  return output, packed_interlayer
