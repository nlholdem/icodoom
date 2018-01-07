#!/usr/bin/python3
import deep_feedback_learning
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import itertools
from PIL import Image


def buildFilters():
    ksize = 35
    sigma = 5.
    gamma = 1.
    theta_vals = np.linspace(0., np.pi, 4, endpoint=False)
#    lambd_vals = (3,7)
#    sigma_vals = (1.5,3)

    lambd_vals = (1.5, 3, 7, 17)
    sigma_vals = (0.5, 1.5, 3, 7)
    """
    theta: orientation
    lambda: wavelength
    sigma: standard deviation
    gamma: aspect ratio
    """
    coeffs = ((theta, lambd, sigma) for lambd, sigma in zip(lambd_vals, sigma_vals) for theta in theta_vals)

    filters = [(cv2.getGaborKernel((ksize,ksize), coeff[2], coeff[0], coeff[1], gamma)/(0.01*ksize*ksize*sigma), coeff[2])
               for coeff in coeffs]

    for f in filters:
        print("Filter: Min: ", np.amin(f[0]), " Max: ", np.amax(f[0]))
        img = Image.fromarray((f[0] * 128 / np.amax(f[0])))
        img.show()

    return filters



def main(argv):

    static = Image.open("/home/paul/Pictures/PBR.png")
    staticArray = np.array(static)
#    gray = 0.3*staticArray[:,:,2] + 0.59*staticArray[:,:,1] + 0.11*staticArray[:,:,0]
    gray = cv2.cvtColor(staticArray, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('/home/paul/tmp/Images/gray.jpg',gray)

#    cv2.imshow('thing', gray)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    gray = np.sum(staticArray, axis=2)
    print("shape: ", staticArray.shape)
    gray = gray - np.mean(gray)
    width = gray.shape[1]
    height = gray.shape[0]
    thresh = 0.
    print("width: ", width, " height: ", height)
    img = Image.fromarray(gray)
#    img.show()

    filters = buildFilters()
    print("there are ", len(filters), " filters")
    imgArray = []
    i=0
    inputImage = np.zeros((height, width))
    posImage = np.zeros((height, width))
    negImage = np.zeros((height, width))
#    inputImage.fill(0.)
#    posImage.fill(0.)
#    negImage.fill(0.)
    for f in filters:
        print("sigma: ", f[1])
        gray1 = cv2.filter2D(gray, -1, f[0])
        print("Filtered: Min: ", np.amin(gray1), " Max: ", np.amax(gray1))
#        print("max: ", np.amax(gray1), " min: ", np.amin(gray1))
#        print("** max: ", np.amax(gray), " min: ", np.amin(gray))

#        gray1 = cv2.resize(gray1, (int(width / f[1]), int(height / f[1])))
        print("x: ", int(width / f[1]), " y: ", int(height / f[1]))

        negImage[np.where(gray1<thresh)] += gray1[np.where(gray1<thresh)] / float(len(filters))
        posImage[np.where(gray1>thresh)] += gray1[np.where(gray1>thresh)] / float(len(filters))


        inputImage = np.append(inputImage, gray1)
        print("inputImage size: ", inputImage.shape)

#        imgArray.append(Image.fromarray(gray1))
#        imgArray[i].show()
#        i += 1
    print("Pos Max: ", np.amax(posImage), " Min: ", np.amin(posImage))
    print("Neg Max: ", np.amax(negImage), " Min: ", np.amin(negImage))
#    posImage = 255. * (posImage - np.amin(posImage)) / (np.amax(posImage) - np.amin(posImage))
#    negImage = 255. * (negImage - np.amin(negImage)) / (np.amax(negImage) - np.amin(negImage))
    negImage = -negImage
    imgArray.append(Image.fromarray(posImage))
    imgArray.append(Image.fromarray(negImage))
#    cv2.imshow('Pos', posImage)
#    cv2.imshow('Neg', negImage)
    cv2.imwrite('/home/paul/tmp/Images/pos.jpg',posImage)
    cv2.imwrite('/home/paul/tmp/Images/neg.jpg',negImage)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#    imgArray[0].show()
#    imgArray[1].show()

#    input('press a key')
    print("**inputImage size: ", inputImage.shape)


if __name__ == '__main__':
    if len(sys.argv) < 0:
        print ("usage: loadWeights epoch n_filters width height")
    else:
        main(sys.argv[1:])

