import math

def calculate_maxpool_output(input_size, number_of_maxpools, pooling_kernel_size, pooling_stride):
    H, W = input_size
    Hp, Wp = pooling_kernel_size
    s = pooling_stride
    for i in range(number_of_maxpools):
        H, W = math.floor((H - Hp)/s + 1), math.floor((W - Wp)/s + 1)
    return H, W

def min_maxer(input_image):
    min = input_image.min()
    max = input_image.max()
    min_maxed_image = (input_image - min)/(max - min)
    return min_maxed_image