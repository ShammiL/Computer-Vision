import cv2
import numpy as np
import os


def read_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return img.tolist()


def image_write(img, file_name):
    cv2.imwrite(file_name + '.jpg', np.array(img))


def zero_array(x, y):
    """
    :param x: x dim
    :param y: y dim
    :return: 2d list of 0s
    """
    return [[0 for j in range(y)] for i in range(x)]


def intensity_image(img):
    """
    intensity is the sum of the RGB values normalized to 1
    :param img:
    :return:
    """
    x_shape = len(img)
    y_shape = len(img[0])
    sum_img = 0
    output = zero_array(x_shape, y_shape)

    for i in range(x_shape):
        for j in range(y_shape):
            output[i][j] = (img[i][j][0] + img[i][j][1] + img[i][j][2]) / 3
            sum_img += output[i][j]
    return output, sum_img / (x_shape * y_shape)


def partition(img, threshold):
    x_shape = len(img)
    y_shape = len(img[0])
    output = zero_array(x_shape, y_shape)
    sum1 = 0
    count1 = 0
    sum2 = 0
    count2 = 0

    for i in range(x_shape):
        for j in range(y_shape):
            if img[i][j] > threshold:
                output[i][j] = 0
                count1 += 1
                sum1 += img[i][j]
            else:
                output[i][j] = 255
                count2 += 1
                sum2 += img[i][j]
    return output, sum1 / count1, sum2 / count2


def inter_means_algorithm(img):
    intensity_img, new_threshold = intensity_image(img)

    while True:
        threshold = new_threshold
        partition_img, mean1, mean2 = partition(intensity_img, threshold)
        new_threshold = (mean1 + mean2) / 2

        if abs(new_threshold - threshold) < 0.0000000000001:
            break

    return partition_img


if __name__ == '__main__':
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        n = f.split('.')
        # check image file
        if n[-1] == 'jpg' or n[-1] == 'jpeg' or \
                n[-1] == 'JPEG' or n[-1] == 'JPG' or \
                n[-1] == "png" or n[-1] == "PNG":
            image = read_image(f)
            out = inter_means_algorithm(image)
            image_write(out, n[0] + "_segment")
