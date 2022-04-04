import math
import os
import shutil

import cv2
import numpy as np

import scipy.stats as st

image = cv2.imread("src.jpg")


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    print(x)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


kernel_size = 15

lowpass_kernel_gaussian = gkern(kernel_size)
lowpass_kernel_gaussian = lowpass_kernel_gaussian / lowpass_kernel_gaussian.sum()

lowpass_kernel_box = np.ones((kernel_size, kernel_size))
lowpass_kernel_box = lowpass_kernel_box / (kernel_size * kernel_size)


# lowpass_image_gaussian = cv2.filter2D(image, -1, lowpass_kernel_gaussian)
# lowpass_image_box = cv2.filter2D(image, -1, lowpass_kernel_box)
#
# cv2.imwrite('1.jpg', lowpass_image_box)
#
# highpass_image_gaussian = image - lowpass_image_gaussian
# highpass_image_gaussian = np.absolute(highpass_image_gaussian)
#
# highpass_image_box = image - lowpass_image_box
# highpass_image_box = np.absolute(highpass_image_box)
#
# cv2.imwrite('2.jpg', highpass_image_box)
#
# bandreject_image = lowpass_image_gaussian + highpass_image_box
#
# bandpass_image = image - bandreject_image
# bandpass_image = np.absolute(bandpass_image)
#
# cv2.imwrite('3.jpg', bandpass_image)


def choose_file(file):
    f_name, f_ext = os.path.splitext(file)
    print(f"./data/{file}")
    image_src = cv2.imread(f"./data/{file}")
    cv2.imwrite(f"results/{f_name}/src.jpg", image_src)
    os.mkdir(f"results/{f_name}")

    lowpass_image_gaussian = cv2.filter2D(image_src, -1, lowpass_kernel_gaussian)
    lowpass_image_box = cv2.filter2D(image_src, -1, lowpass_kernel_box)

    cv2.imwrite(f"results/{f_name}/lpf.jpg", lowpass_image_box)

    highpass_image_gaussian = image_src - lowpass_image_gaussian
    highpass_image_gaussian = np.absolute(highpass_image_gaussian)

    highpass_image_box = image_src - lowpass_image_box
    highpass_image_box = np.absolute(highpass_image_box)

    cv2.imwrite(f"results/{f_name}/hpf.jpg", highpass_image_box)

    bandreject_image = lowpass_image_gaussian + highpass_image_box

    bandpass_image = image_src - bandreject_image
    bandpass_image = np.absolute(bandpass_image)

    cv2.imwrite(f"results/{f_name}/bpf.jpg", bandpass_image)


if __name__ == "__main__":
    shutil.rmtree("./results/")
    os.mkdir("./results")
    for root, dirs, files in os.walk("./data"):
        print(files)
        for file in files:
            choose_file(file)
