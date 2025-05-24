#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import numpy
import argparse
import SimpleITK
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import skimage
import skimage.io
import skimage.morphology

class CXRSingle:
    def __init__(self, img):
        self.img = img
        self.edge = 10

    def lung(self):
        """ isolate the lungs """
        return(self.img)

    def bbox(self):
        """ find the bounding box """
        a = numpy.full(self.img.shape, 0, dtype = numpy.uint8)

        try:
            [r, c] = numpy.where(self.img)
            a[min(r):max(r), min(c):max(c)] = 255
        except:
            y, x = a.shape      # x-axis; column and y-axis; row
            r, c = skimage.draw.polygon(
                numpy.array([self.edge, self.edge, y - self.edge, y - self.edge, self.edge]),
                numpy.array([self.edge, x - self.edge, x - self.edge, self.edge, self.edge]))
            a[r, c] = 255

        return(a)

    def block(self):
        """ find the boundaries and reset the area """
        a = numpy.full(self.img.shape, 0, dtype = numpy.uint8)
        y, x = a.shape          # x-axis; column and y-axis; row
        r, c = skimage.draw.polygon(
            numpy.array([self.edge, self.edge, y - self.edge, y - self.edge, self.edge]),
            numpy.array([self.edge, x - self.edge, x - self.edge, self.edge, self.edge]))
        a[r, c] = 255
        return(a)

class CXRPaired:
    def __init__(self, img):
        """ identify largest objects in the image """
        self.dsi = SimpleITK.RelabelComponent(
            SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(img)), sortByObjectSize = True)
        self.shape = img.shape

    def lung(self):
        """ get separate segmented lungs """
        try:    # left lung
            a = SimpleITK.GetArrayFromImage(self.dsi == 1)
            a[a == 1] = 255
        except:
            a = numpy.full(self.shape, 0, dtype = numpy.uint8)
            a[:, 0:int(a.shape[1] / 2.0)] = 255

        try:    # right lung
            b = SimpleITK.GetArrayFromImage(self.dsi == 2)
            b[b == 1] = 255
        except:
            b = numpy.full(self.shape, 0, dtype = numpy.uint8)
            b[:, int(b.shape[1] / 2.0):] = 255

        return(a, b)

    def bbox(self):
        """ get separate bounding boxes for left and right lungs """
        a = numpy.full(self.shape, 0, dtype = numpy.uint8)
        b = numpy.full(self.shape, 0, dtype = numpy.uint8)

        try:    # left lung
            [r, c] = numpy.where(SimpleITK.GetArrayFromImage(self.dsi == 1))
            a[min(r):max(r), min(c):max(c)] = 255
        except:
            a[:, 0:int(a.shape[1] / 2.0)] = 255

        try:    # right lung
            [r, c] = numpy.where(SimpleITK.GetArrayFromImage(self.dsi == 2))
            b[min(r):max(r), min(c):max(c)] = 255
        except:
            b[:, int(b.shape[1] / 2.0):] = 255

        return(a, b)

    def block(self):
        """ get separate blocks for left and right lungs """
        a = numpy.full(self.shape, 0, dtype = numpy.uint8)
        b = numpy.full(self.shape, 0, dtype = numpy.uint8)

        try:    # left lung
            [r, c] = numpy.where(SimpleITK.GetArrayFromImage(self.dsi == 1))
            a[:, 0:max(c)] = 255
        except:
            a[:, 0:int(a.shape[1] / 2.0)] = 255

        try:    # right lung
            [r, c] = numpy.where(SimpleITK.GetArrayFromImage(self.dsi == 2))
            b[:, min(c):] = 255
        except:
            b[:, int(b.shape[1] / 2.0):] = 255

        return(a, b)

def segment(image, mask, unet):
    """ lung segmentation algorithm """
    IM_SHAPE = (256, 256)
    x = numpy.expand_dims(
        skimage.exposure.equalize_hist(skimage.transform.resize(image, IM_SHAPE)), -1)
    y = numpy.expand_dims(
        skimage.transform.resize(mask, IM_SHAPE), -1)
    x = numpy.array([x])        # numpy array of images
    y = numpy.array([y])        # numpy array of images
    x -= x.mean()
    x /= x.std()
    img_data = ImageDataGenerator(rescale = 1.)

    for xx, yy in img_data.flow(x, y, batch_size = 1):
        """ flow runs an infinite loop; the output is the same """
        pred = unet.predict(xx)[..., 0].reshape(x[0].shape[:2]) > 0.5
        pred =skimage.morphology.remove_small_holes(
            skimage.morphology.remove_small_objects(
                pred, 0.02 * numpy.prod(IM_SHAPE)), 0.02 * numpy.prod(IM_SHAPE))
        pred = skimage.transform.resize(pred, image.shape) * 255
        return(pred.astype(numpy.uint8))

def main(argv):
    """ main procedure """
    try:
        # load lung image, mask, and segmentation model
        b = CXRPaired(segment(
            skimage.img_as_float(skimage.io.imread(argv.input, as_gray = True)),
            skimage.img_as_float(skimage.io.imread(argv.mask, as_gray = True)),
            load_model("trained_model.hdf5")))
        x, y = b.lung()     # separate lung segmentation
        #x, y = b.bbox()     # separate bounding boxes for left and right lungs
        #x, y = b.block()    # separate blocks for left and right lungs
        skimage.io.imsave("lung.left.png", x)   # left image
        skimage.io.imsave("lung.right.png", y)  # right image
    except Exception as e:
        print("ERROR: {}, file: {}".format(e, argv.input))

    return(True)

if __name__ == "__main__":
    argv = argparse.ArgumentParser(description = """
        Lung segmentation refers to the process of accurately identifying regions and boundaries of the
        lung field from surrounding thoracic tissue. It is an essential first step in pulmonary image
        analysis of many clinical decision support systems. This implementation utilizes the U-Net
        (https://github.com/imlab-uiip/lung-segmentation-2d) for lung segmentation.""")
    argv.add_argument("--input", "-i", type = str, dest = "input", required = True,
        help = "[Required] Lung image to be segmented.")
    argv.add_argument("--mask", "-m", type = str, dest = "mask", required = False, default = "4-mask.png",
        help = "[Optional] Lung segmentation mask. The default value is 4-mask.png.")

    sys.exit(not main(argv.parse_args()))
