"""code for moire "SIM" demo

This will generate PNGs suitable for printout for the moire SIM
demo. It's important to make sure that however you choose to print the
image you maintain the intended DPI. I did this using GIMP and
explicitly setting the image dpi back to 600 before printing (by
default it tries to scale the image to fill the full page, resulting
in somewhat lower DPI)

BKC, 7/12/19
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

# dots per inch of your figure; I choose 600 to match the printer
DPI = 600
# paper dimensions chosen to give .5 margins on letter in landscape orientation
X_INCHES = 10
Y_INCHES = 7.5  # these numbers chosen to give uniform .5 margins on letter
X_PIXELS = int(X_INCHES*DPI)
Y_PIXELS = int(Y_INCHES*DPI)
STRIPES_PER_INCH = 40
DOTS_PER_CYCLE = DPI / STRIPES_PER_INCH
# reduce visibility of stripe distortions in the encoded image using
# smaller modulations and blurring sharp features
MODULATION_AMPLITUDE = .5
GAUSSIAN_SIGMA = 15
OUTPUT_FOLDER = '.'


def prepare_image_for_encoding(raw_image):
    gray_image = rgb2gray(raw_image)
    return resize(gray_image, (Y_PIXELS, X_PIXELS))


def generate_encoded_image(raw_image, output_dir, preprocess_image=True,
                           output_name='encoded_image.png'):
    if preprocess_image:
        raw_image = prepare_image_for_encoding(raw_image)
    base_phase = ((np.pi / DOTS_PER_CYCLE) *
                  np.arange(DPI*X_INCHES).reshape(1, -1))
    modulation_phase = (MODULATION_AMPLITUDE *
                        gaussian_filter(raw_image, GAUSSIAN_SIGMA))
    pattern = np.sin(base_phase + modulation_phase)**2
    plt.imsave(os.path.join(output_dir, output_name),
               pattern, format='png', dpi=DPI, cmap='gray')


def generate_stripe_mask(output_dir, stripe_angle=0):
    k_x, k_y = np.cos(stripe_angle), np.sin(stripe_angle)
    y_ind, x_ind = np.ogrid[0:Y_PIXELS,
                            0:X_PIXELS]
    raw_phase = (np.pi / DOTS_PER_CYCLE)*(k_x*x_ind + k_y*y_ind)
    stripe_mask = np.sin(raw_phase)**2
    plt.imsave(os.path.join(output_dir, 'stripe_mask.png'),
               stripe_mask, format='png', dpi=DPI, cmap='gray')


if __name__ == '__main__':
    generate_stripe_mask(OUTPUT_FOLDER)
    horse = imread(os.path.join(OUTPUT_FOLDER, 'horse.png'))
    generate_encoded_image(horse, OUTPUT_FOLDER, preprocess_image=True,
                           output_name='encoded_horse.png')
