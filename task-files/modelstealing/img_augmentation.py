import random
from PIL import Image, ImageEnhance
import numpy as np


def crop_img(image, size=(16, 16)):
    width, height = image.size
    left = random.randint(0, width - size[0])
    top = random.randint(0, height - size[1])
    right = left + size[0]
    bottom = top + size[1]
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((width, height))

    return resized_image


def rotate_img(image, angel=90):
    rotated_image = image.rotate(angel)

    return rotated_image


def color_distort(image, jitter=0.4):
    # Convert image to HSV color space
    image_hsv = image.convert('HSV')

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image_hsv)
    brightness_factor = 1 + random.uniform(-jitter, jitter)
    image_hsv = enhancer.enhance(brightness_factor)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image_hsv)
    contrast_factor = 1 + random.uniform(-jitter, jitter)
    image_hsv = enhancer.enhance(contrast_factor)

    # Adjust saturation
    enhancer = ImageEnhance.Color(image_hsv)
    saturation_factor = 1 + random.uniform(-jitter, jitter)
    image_hsv = enhancer.enhance(saturation_factor)

    # Adjust hue
    hue_factor = random.uniform(-jitter, jitter)
    image_hsv = image_hsv.convert('RGB')
    image_hsv = Image.merge('HSV', (
        image_hsv.split()[0].point(lambda x: (x + hue_factor) % 256),
        image_hsv.split()[1],
        image_hsv.split()[2]
    ))
    image_rgb = image_hsv.convert('RGB')

    return image_rgb


def add_gaussian_noise(image, mean=2, std=0.1):
    # Convert image to numpy array
    img_array = np.array(image)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, img_array.shape)

    # Add noise to the image
    noisy_img_array = img_array + noise

    # Clip values to ensure they are within the valid range [0, 255]
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    # Convert back to PIL image
    noisy_image = Image.fromarray(np.uint8(noisy_img_array))

    return noisy_image
