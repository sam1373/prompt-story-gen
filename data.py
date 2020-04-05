import sys
import torch.utils.data as data
from os import listdir
#from utils.tools import default_loader, is_image_file, normalize
import os

import torch
import torchvision.transforms as transforms

from pycocotools.coco import COCO

import random

import numpy as np
import skimage.io as io
from skimage.draw import polygon

import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils import pil_loader

import json

import re

class Dataset(data.Dataset):
    def __init__(self, source, target):
        super(Dataset, self).__init__()

        with open(source, errors='ignore') as fs:
            with open(target, errors='ignore') as ft:
                self.samples = list(zip(fs.readlines(), ft.readlines()))


    def wp_preprocess(self, text):

        # Standardize some symbols
        text = text.replace('<newline>', '\n')
        text = text.replace('``', '"')
        text = text.replace("''", '"')
        # Detokenize
        text = re.sub(' +', ' ', text)
        text = re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', text)
        text = re.sub('" (.*) "', '"\g<1>"', text)
        text = text.replace(" n't", "n't")

        return text

    def __getitem__(self, index):

        prompt, story = self.samples[index]

        prompt = prompt[prompt.find("]") + 1:]

        prompt = re.sub('\[ (.*) \]', '', prompt)

        prompt = self.wp_preprocess(prompt.strip())

        story = self.wp_preprocess(story.strip())
        
        return prompt, story


    def __len__(self):
        return len(self.samples)
