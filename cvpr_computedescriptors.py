import os 
import numpy as np 
import cv2 
import scipy.io as sio 
from extractRandom import *
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights
from extractRandom import extractDescriptors, create_histograms_for_all_images

class ImportantVariables:

    descriptorDict={}
    feature_database={}
    PCA_descriptors=[]
    bovw_finalImageSet=[]
    bovw_top_indices=[]
    bovw_histograms=[]
    bovw_finalImageSetLabels=[]
    classIndex={}

def loadModel():
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    model = nn.Sequential(*list(model.children())[:-1])  
    model.eval() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model,preprocess,device

def computeDescriptors(descriptorTechnique,isPCA,isSVM,colour_bins=None,texture_bins=None):

    # Initializing Directories
    DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
    OUT_FOLDER = 'newDescriptors'
    OUT_SUBFOLDER = descriptorTechnique
    os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True) # Ensure the output directory exists
 

    # Initializing Variables 
    bovDescriptor=[]
    bw_images=[]
    imgIndex=0
   

    if(descriptorTechnique=="CNN"):
        loadModel()
        print("Model Loaded")
    
    # Iterate through all BMP files in the dataset folder
    for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
        if filename.endswith(".bmp"):
            imgIndex+=1
            img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
            img = cv2.imread(img_path).astype('uint8')  
            fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))   
            F= extractDescriptors(img,descriptorTechnique,colour_bins,texture_bins) 
            if(isPCA):
                ImportantVariables.PCA_descriptors.append(F)

            if(descriptorTechnique=='CNN'):
                ImportantVariables.feature_database[filename] = F
            
            if(descriptorTechnique=='BoVW'):
                bw_images.append(img_path) 
                bovDescriptor.append(F)

            ImportantVariables.descriptorDict[os.path.basename(img_path)] = F
            if(os.path.basename(img_path).split('_')[0] not in ImportantVariables.classIndex):
                ImportantVariables.classIndex[os.path.basename(img_path).split('_')[0]]=imgIndex

            # Save the descriptor to a .mat file
            sio.savemat(fout, {'F': F})  

    if(descriptorTechnique=='BoVW'):
        all_descriptors = []
        for descriptor in bovDescriptor:
                all_descriptors.append(np.array(descriptor))
        all_descriptors = np.vstack(all_descriptors)
    
  
        from scipy.cluster.vq import kmeans 
        k = 200
        iters = 1
        codebook, _ = kmeans(all_descriptors, k,iters)


        ImportantVariables.histograms,ImportantVariables.bovw_finalImageSet,ImportantVariables.bovw_finalImageSetLabels= create_histograms_for_all_images(bw_images, codebook)
