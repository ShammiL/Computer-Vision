import cv2
import os
import numpy as np
import math

def filled_array(x_shape, y_shape, num):
  return [[num for i in range(y_shape)] for j in range(x_shape)]

def assign_slice(input, image, xs, xe, ys, ye):
  xi = 0
  for x in range(xs, xe):
    yi = 0
    for y in range(ys, ye):
      input[x][y] = image[xi][yi]
      yi += 1
    xi += 1
  return input

def get_slice(image, xs, xe, ys, ye):
  return [x[ys:ye] for x in image[xs:xe]]

def convolve(filter, input):
  out = 0
  for i in range(len(filter)):
    for j in range(len(input)):
      out += filter[i][j] * input[i][j]
  return out

def flip(filter):
  for f in filter:
    filter.reverse()
  filter.reverse()
  return filter

def convert_greyscale(image, x_shape, y_shape):
  grey_out = filled_array(x_shape, y_shape, 0)
  for i in range(x_shape):
    for j in range(y_shape):
      grey_out[i][j] =  image[i][j][0] * 0.2989 + image[i][j][1] * 0.5870 + image[i][j][2] * 0.1140 
  return grey_out

def wrap_image(image, x_shape, y_shape, flt_size):

  input = filled_array(x_shape + (flt_size-2) * 2, y_shape + (flt_size-2) * 2, 0)
  input_x = len(input) - (flt_size-2)
  input_y = len(input[0]) - (flt_size-2)
  image_y = len(image[0]) - (flt_size-2)

  input = assign_slice(input, image, (flt_size - 2), input_x, (flt_size - 2), input_y)
  input = assign_slice(input, image[-(flt_size - 2):], 0, (flt_size - 2), (flt_size - 2), input_y)
  input = assign_slice(input, image[0: (flt_size - 2)], input_x, len(input), (flt_size - 2), input_y)
  input = assign_slice(input, get_slice(image, 0, len(image), image_y, len(image[0])), (flt_size - 2), input_x, 0, (flt_size - 2))
  input = assign_slice(input, get_slice(image, 0, len(image), 0, (flt_size - 2)), (flt_size - 2), input_x, input_y, len(input))
  return input

def gaussian_filter(input, x_shape, y_shape, flt_size = 3):

  filter = filled_array(flt_size, flt_size, 1)
  filter = [[y / (flt_size ** 2) for y in x] for x in filter]
  wrapped = wrap_image(input, x_shape, y_shape, flt_size)
  gaus_out = filled_array(x_shape, y_shape, 0)
  for x in range(x_shape):
    for y in range(y_shape):
      gaus_out[x][y] = convolve(filter, get_slice(wrapped, x, x+flt_size, y, y+flt_size))
  return gaus_out

def sobel_filter(image, x_shape, y_shape):
  S_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  S_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  S_x = flip(S_x)
  S_y = flip(S_y)
  wrapped = wrap_image(image, x_shape, y_shape, 3)
  sobel_x = filled_array(x_shape, y_shape, 0)
  sobel_y = filled_array(x_shape, y_shape, 0)
  sobel_out = filled_array(x_shape, y_shape, 0)


  for x in range(x_shape):
    for y in range(y_shape):
      sobel_x[x][y] = convolve(S_x, get_slice(wrapped, x, x+3, y, y+3))
      sobel_y[x][y] = convolve(S_y, get_slice(wrapped, x, x+3, y, y+3))
      sobel_out[x][y] = (sobel_x[x][y] ** 2 + sobel_y[x][y] ** 2) ** 0.5
  return (sobel_x, sobel_y, sobel_out)

def non_max_supression(sobel_x, sobel_y, sobel_out, x_shape, y_shape):
  supressed = filled_array(x_shape, y_shape, 0)
  for x in range(x_shape):
    for y in range(y_shape):
      next = 255
      prev = 255
      if sobel_x[x][y] == 0 and sobel_y[x][y] >= 0:
        angle = 90
      elif sobel_x[x][y] == 0 and sobel_y[x][y] < 0:
        angle = -90
      else:
        angle = math.atan(sobel_y[x][y]/sobel_x[x][y])*180/np.pi

      if angle < 0:
        angle += 180
        
      try:
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
          next = sobel_out[x][y+1]
          prev = sobel_out[x][y-1]
        elif (22.5 <= angle < 67.5):
            next = sobel_out[x+1][y-1]
            prev = sobel_out[x-1][y+1]
        elif (67.5 <= angle < 112.5):
            next = sobel_out[x+1][y]
            prev = sobel_out[x-1][y]
        elif (112.5 <= angle < 157.5):
            next = sobel_out[x-1][y-1]
            prev = sobel_out[x+1][y+1]
        if (sobel_out[x][y] >= next) and (sobel_out[x][y] >= prev):
          supressed[x][y] = sobel_out[x][y]
        else:
          supressed[x][y] = 0
      except IndexError as e:
        pass
  return supressed

def double_threshold(image, x_shape, y_shape, high_t = 0.09, low_t = 0.05):
  highT = max(max(image)) * high_t
  lowT = highT * low_t

  out = filled_array(x_shape, y_shape, 0)

  low = 25  
  high = 255  

  for x in range(x_shape):
      for y in range(y_shape):
          if image[x][y] >= highT:
              out[x][y] = high
          elif highT >= image[x][y] >= lowT:
              out[x][y] = low
          else:
              out[x][y] = 0
  return out

for f in os.listdir():
  file_name = os.path.splitext(f)[0]
  if f.lower().endswith(".jpeg") or f.lower().endswith(".jpg") or f.lower().endswith(".png"):
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    image = image.tolist()
    x_shape = len(image)
    y_shape = len(image[0])
    grey_out = convert_greyscale(image, x_shape, y_shape)
    gaus_out = gaussian_filter(grey_out, x_shape, y_shape, 3)
    sobel_x, sobel_y, sobel_out = sobel_filter(gaus_out, x_shape, y_shape)
    supressed = non_max_supression(sobel_x, sobel_y, sobel_out, x_shape, y_shape)
    final = double_threshold(supressed, x_shape, y_shape)
    cv2.imwrite(file_name + '_canny' + '.jpg', np.array(final))