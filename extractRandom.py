import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from scipy.cluster.vq import  vq
from sklearn.decomposition import PCA
import os
from sklearn.svm import SVC

def extractDescriptors(img,descriptorTechnique,colour_bins,texture_bins):
    
    if(descriptorTechnique=='global_colour_histogram'):     
        return global_colour_histogram(img,int(colour_bins))

    elif(descriptorTechnique=='grid_based_descriptor'):
        return grid_based_descriptor(img,int(colour_bins),int(texture_bins))
    
    elif(descriptorTechnique=='sift_feature_descriptor' or descriptorTechnique=='BoVW'):
        return sift_detector(img)
    elif(descriptorTechnique=='CNN'):
        return extract_cnn_features(img)




# Function to split the image into a grid and concatenate features from each grid cell
def grid_based_descriptor(img,  colour_bins=32, texture_bins=8,grid_size=(4, 4)):

    h, w, _ = img.shape
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    descriptor = []

    # Loop over grid cells
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            cell = img[row * cell_h: (row + 1) * cell_h, col * cell_w: (col + 1) * cell_w]

            # Extract color and texture histograms for the cell
            color_hist = global_colour_histogram(cell, colour_bins)
            texture_hist = edge_orientation_histogram(cell, texture_bins)

            # Concatenate the histograms for the cell
            cell_descriptor = np.concatenate((color_hist, texture_hist))
            descriptor.extend(cell_descriptor)

    return np.array(descriptor)



# Function to compute global color histogram for an image
def global_colour_histogram(img, bins_per_channel):
    bins = [bins_per_channel, bins_per_channel, bins_per_channel] 

    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()

    return hist

# Function to compute Edge Orientation Histogram (EOH) for texture in a grid cell
def edge_orientation_histogram(img, bins=8):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle_bins = np.int32(angle * (bins / 360.0))  

    angle_bins = angle_bins.flatten()
    mag = mag.flatten()

    hist = np.bincount(angle_bins, weights=mag, minlength=bins)

    hist = hist / (hist.sum() + 1e-7)  

    return hist

def extract_cnn_features(img):
    from cvpr_computedescriptors import loadModel
    model,preprocess,device =loadModel()
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img_tensor = preprocess(img).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.cpu().numpy().flatten()

def sift_detector(img):
    if len(img.shape) == 3:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    extractor = cv2.SIFT_create()
    _, img_descriptors = extractor.detectAndCompute(img, None)

    return img_descriptors

def compute_image_histogram(descriptors, codebook):
    img_visual_words, _ = vq(descriptors, codebook)
    histogram, _ = np.histogram(img_visual_words, bins=np.arange(codebook.shape[0] + 1))
    return histogram

def create_histograms_for_all_images(images, codebook):
    finalImageSet=[]  
    histograms = []
    finalImageSetLabels=[]
    sift = cv2.SIFT_create()
    
    for img in images:
        finalImageSet.append(img)
        finalImageSetLabels.append(os.path.basename(img).split('_')[0])

        gray = cv2.cvtColor(cv2.imread(img).astype('uint8'), cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None:
            histogram = compute_image_histogram(descriptors, codebook)
          
        else:
            histogram = np.zeros(len(codebook))  
            

        histograms.append(histogram)
    
    return np.array(histograms),finalImageSet,finalImageSetLabels

def apply_pca(descriptors, variance=0.95):
    pca = PCA(n_components=variance)
    reduced_descriptors = pca.fit_transform(descriptors)
    return reduced_descriptors, pca
  
def train_svm_classifier(histograms, labels):
    svm = SVC(kernel='linear')
    svm.fit(histograms, labels)
    return svm

  