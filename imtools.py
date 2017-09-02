import os
import mglearn

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from pylab import *
from numpy import *

def get_imlist(path,str):
    '''
    return a list of all files in 'path' ending with 'str'
    '''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(str)]

def imresize(im,sz):
    '''
    use the value 'sz' to reassign the size of 'im'
    '''
    # use the function uint8() to convert to uint8 format
    pil_im=Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im,nbr_bins=256):
    '''
    Histogram equalization
    '''
    # get the histogram of 'im'
    imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)
    # get the cumulative distribution function of 'im'
    cdf=imhist.cumsum()
    # normalize cdf
    cdf=255*cdf/cdf[-1]
    # linear interpolation
    im2=interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

def compute_average(imlist):
    '''
    generate the average image of all images in 'imlist'
    '''
    # open the first image and store the value in an array
    averageim=array(Image.open(imlist[0]),'f')
    for imname in imlist[1:]:
        try:
            averageim+=array(Image.open(imname))
        except:
            print(imname+'...skipped')
    averageim/=len(imlist)
    # return the average image in uint8 format
    return array(averageim,'uint8')
