import os
from PIL import Image
import numpy as np


def read_and_tranfor_imag(directory):
    imags = []
    for archive in os.listdir(directory):
        ruta_archivo = os.path.join(directory, archive)
        imagen = Image.open(ruta_archivo).convert('L') 
        imagen = imagen.resize((256, 256)) 
        imags.append(imagen)
    return imags


def average_imagen(images):
    size = images[0].size
    mode = images[0].mode
    images = [image.resize(size, resample=Image.BICUBIC).convert(mode) for image in images]
    image_arrays = [np.array(image) for image in images]
    mean_array = np.mean(image_arrays, axis=0)
    mean_image = Image.fromarray(np.uint8(mean_array))
    return mean_image


def my_face_distant(imag1, imag2):
    imag1_array = np.array(imag1)
    imag2_array = np.array(imag2)
    return np.sqrt(np.sum((imag1_array - imag2_array) ** 2))


def trasnform_one_image(directory):
    imag = Image.open(directory).convert('L')
    imag = imag.resize((256, 256))
    return imag
