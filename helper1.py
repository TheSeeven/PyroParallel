import numpy as np
from scipy.signal import convolve
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
picture_path = "C:\\Users\\peria\\Videos\\Mass Effect Andromeda\\25600x14400_1_NO_HDR.bmp"


def edge_detection(input_image):
    # Define the Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel operators
    gradient_x = np.abs(convolve(input_image, sobel_x, mode='same'))
    gradient_y = np.abs(convolve(input_image, sobel_y, mode='same'))

    # Calculate the gradient magnitude
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient


# Example usage
input_image = Image.open(picture_path)
image_array = np.array(input_image)
grayscale_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
grayscale_image = np.clip(grayscale_image, 0, 255)
grayscale_image = grayscale_image.astype(np.uint8)
output_image = edge_detection(grayscale_image)
output_image_pil = Image.fromarray(output_image.astype(np.uint8), mode='L')
output_image_pil.save('./output_test/edge_detection_no_kernel_test.bmp')