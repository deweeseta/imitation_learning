import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(image_file.strip())


def crop(image):
    """
    Crop the image, resize input to 280x120
    """
    #       (original - target)
    # rows:      (height - 120) / 2 == crop_height
    # columns:   (width - 280) / 2 == crop_width
    height, width, _ = image.shape
    crop_height = int((height - 120) / 2)
    crop_width = int((width - 280) / 2)
    return image[crop_height:-crop_height, crop_width:-crop_width]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess_multiple(center, left, right, steering_angle):
    """
    Combine all preprocess functions into one
    """
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image = preprocess(image)
    return image, steering_angle


def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2
    elif choice == 1:
        return load_image(right), steering_angle - 0.2
    return load_image(center), steering_angle
