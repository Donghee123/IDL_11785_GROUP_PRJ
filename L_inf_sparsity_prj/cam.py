
import cv2
import numpy as np
import torch.nn.functional as F
import torch
from torch import topk
from torchvision import models

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        
        # put class label text on the result
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)

def getPosition(listofPositions, image_shape):
    widthdivide = listofPositions.shape[0]
    heightdivide = listofPositions.shape[1]

    nWithpartitial = int(image_shape[0] / widthdivide)
    nHeightpartitial = int(image_shape[1] / heightdivide)

    listofROI = []

    nPosY = 0
    for y in range(0, image_shape[0], nHeightpartitial):
        nPosX = 0

        for x in range(0, image_shape[1], nWithpartitial):

            if listofPositions[nPosY][nPosX] == 1:
                listofROI.append([(yy,xx) for yy in range(y, y + nHeightpartitial) for xx in range(x, x+nWithpartitial)])

            nPosX += 1
        nPosY += 1
    
    listofans = []
    for ROIs in listofROI:
        for ROI in ROIs:
            listofans.append(ROI)

    return listofans

def fillValue(image, listOfActivationPos):
    for pos in listOfActivationPos:
        image[pos[0],pos[1]][0] = 0
        image[pos[0],pos[1]][1] = 0
        image[pos[0],pos[1]][2] = 0
    return image

def getActivationPosition(model, originImage,transoformimage, testCount):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model = model
    model.module.layer4.register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    # forward pass through model
    outputs = model(torch.unsqueeze(transoformimage, 0))
    # get the softmax probabilities
    probs = F.softmax(outputs).data.squeeze()
    # get the class indices of top k probabilities
    class_idx = topk(probs, 1)[1].int()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    save_name = f"test{testCount}"
    show_cam(CAMs, 224, 224, cv2.resize(originImage,(224, 224)), class_idx, save_name)
    
    CAMs[0][CAMs[0] <= 150] = 0 
    CAMs[0][CAMs[0] > 0] = 1

    listOfActivationPos = getPosition(CAMs[0], originImage.shape)

    #originImage = fillValue(originImage, listOfActivationPos)

    # file name to save the resulting CAM image with
    
    # show and save the results


    return listOfActivationPos
    
