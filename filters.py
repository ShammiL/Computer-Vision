import cv2
import os
import numpy as np

def wrap_image(image):
  x_shape = image.shape[1]
  y_shape = image.shape[0]
  channels = image.shape[2]

  input = np.zeros((y_shape + 2, x_shape +2, channels))  
  for i in range(channels):
    channel = image[:, :, i]
    channel_input = np.zeros((y_shape + 2, x_shape +2))
    channel_input[1:-1, 1:-1] = channel
    channel_input[0, 1:-1] = channel[-1]
    channel_input[-1, 1:-1] = channel[0]
    channel_input[1:-1, 0] = channel[:, -1]
    channel_input[1:-1, -1] = channel[:, 0]
    input[:, :, i] = channel_input
  return input


def mean_filter(input, x_shape, y_shape, channels, flt_size = 3):

  filter = np.ones(shape=(flt_size, flt_size)) / (flt_size**2)
  mean_out = np.zeros((y_shape, x_shape, channels))
  for c in range(channels):
    for x in range(x_shape):
      for y in range(y_shape):
        mean_out[y, x, c] = (filter * input[y: y+flt_size, x: x+flt_size, c]).sum()
  return mean_out

def median_filter(input, x_shape, y_shape, channels, flt_size = 3):

  filter = np.ones(shape=(flt_size, flt_size))
  median_out = np.zeros((y_shape, x_shape, channels))
  for c in range(channels):
    for x in range(x_shape):
      for y in range(y_shape):
        median_out[y, x, c] = np.median(filter * input[y: y+flt_size, x: x+flt_size, c])
  return median_out

def mid_point_filter(input, x_shape, y_shape, channels, flt_size = 3):

  filter = np.ones(shape=(flt_size, flt_size))
  mid_out = np.zeros((y_shape, x_shape, channels))

  for c in range(channels):
    for x in range(x_shape):
      for y in range(y_shape):
        mid_out[y, x, c] = (np.max(filter * input[y: y+flt_size, x: x+flt_size, c]) + 
        np.min((filter * input[y: y+flt_size, x: x+flt_size, c]))) / 2
  return mid_out


for f in os.listdir():
  if f.endswith(".jpeg") or f.endswith(".jpg"):
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    file_name = os.path.splitext(f)[0]
    x_shape = image.shape[1]
    y_shape = image.shape[0]
    channels = image.shape[2]

    wrapped_image = wrap_image(image)
    mean_img = mean_filter(wrapped_image, x_shape, y_shape, channels, 3)
    cv2.imwrite(file_name + '_mean' + '.jpg', mean_img)
   
    median_img = median_filter(wrapped_image, x_shape, y_shape, channels, 3)
    cv2.imwrite(file_name + '_median' + '.jpg', median_img)
   
    mid_img = mid_point_filter(wrapped_image, x_shape, y_shape, channels, 3)
    cv2.imwrite(file_name + '_mid_point' + '.jpg', mid_img)
    
"""references

https://www.codegrepper.com/code-examples/python/remove+extension+filename+python
https://www.geeksforgeeks.org/python-os-listdir-method/

"""