import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform


#get the pretrained net
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights_local/VOC.pth')

#get the images
testset = VOCDetection(VOC_ROOT, [('2012', 'val')], None, VOCAnnotationTransform())

outputs = []
for i in [39, 40, 41, 42]:
    #get an image
    img_id = i
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #pre-process the image
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    #pass through the network
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    outputs.append(y.cpu())
    
#my custom image
image = cv2.imread('everything.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#pre-process the image
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)

#pass through the network
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)
outputs.append(y.cpu())

outputs = np.array(outputs)
np.save('predictions.npy', outputs)

