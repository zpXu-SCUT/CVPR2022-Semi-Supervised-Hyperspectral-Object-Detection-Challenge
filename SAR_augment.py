import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops , ImageFilter
import PIL
import numpy as np



def Raw(img ,size = 0):
    return img

def MedianFilter(img, size = 3):
    return img.filter(ImageFilter.MedianFilter(size ,))

def MinFilter(img, size = 3):
    return img.filter(ImageFilter.MinFilter(size ,))

def MaxFilter(img, size = 3):
    return img.filter(ImageFilter.MaxFilter(size ,))

def GaussianBlur(img, radius = 2):
    return img.filter(ImageFilter.GaussianBlur(radius ,))

def EDGE_ENHANCE(img, radius = 2):
    return img.filter(ImageFilter.EDGE_ENHANCE())



def SAR_augment_list(m):  # 16 oeprations and their ranges
    l = [
        [
            (Raw, 0 ),
            (MedianFilter, 3),
        ],

        [
            (Raw, 0 ),
            (MedianFilter, 5),

        ],

        [
            (Raw, 0 ),
            (GaussianBlur, 2),

        ],

        [
            (Raw, 0),
            (MedianFilter, 3),
            (MedianFilter, 5),
        ],
        [
            (Raw, 0),
            (EDGE_ENHANCE, 2),
        ],

        [
            (Raw, 0),
        ]

    ]


    return l[m]

def SAR_test_augment_list():  # 16 oeprations and their ranges
    l = [
            (Raw, 0 ),
            (MedianFilter, 3),
            (MedianFilter, 5),
            (GaussianBlur, 2),
            (EDGE_ENHANCE, 2),
        ]

    return l

class SAR_RandAugment:
    def __init__(self, n = 1  , m = 1 ):  # n指定选择1种,m指定第几种策略
        self.n = n
        self.augment_list = SAR_augment_list(m)
        self.choice_weights = [1 / len(self.augment_list) for i in range(len(self.augment_list))]
        self.opera = [i for i in range(len(self.augment_list))]

    def __call__(self, img):
        ops_num = np.random.choice(self.opera ,self.n, p=self.choice_weights)
        for i in ops_num:
            op, val = self.augment_list[i]
            img = op(img, val)

        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.n}, ops='
        for op in self.augment_list:
            fs += f'\n\t{op}'
        fs += ')'
        return fs

class SAR_test_RandAugment:
    def __init__(self, n = 0):  #
        self.n = n
        self.augment_list = SAR_test_augment_list()
        self.ops = [self.augment_list[self.n]]
        print(self.ops)

    def __call__(self, img):
        for op, val in self.ops:
            #print(op, val)
            img = op(img, val)

        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.n}, ops='
        for op in self.augment_list:
            fs += f'\n\t{op}'
        fs += ')'
        return fs