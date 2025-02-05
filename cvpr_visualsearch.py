import os 
import numpy as np 
import scipy.io as sio
import cv2 
from random import randint
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from cvpr_compare import cvpr_compare
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import seaborn as sns   
from confusionMatrix import confusion_matrix
from Evaluation import compute_precision_recall, precisions,recalls, compute_ap
from cvpr_computedescriptors import ImportantVariables
from extractRandom import apply_pca
from extractRandom import extract_cnn_features,train_svm_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cfm, accuracy_score




def visualSearch(descriptorTechnique,isPCA,isSVM,metric):
   

    #Initializing Directories
    DESCRIPTOR_FOLDER = 'newDescriptors'
    DESCRIPTOR_SUBFOLDER = descriptorTechnique
    IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
    Images_subfolder ='Images'
    Images = os.listdir(os.path.join(IMAGE_FOLDER,Images_subfolder))

    #Initializing Variables
    true_labels={}
    total_relevant_images=defaultdict(int)
    ALLFEAT = []
    ALLFILES = []

    # Assigning Class Names
    for i in range(len(Images)):
        Class = Images[i].split('_')[0]
        true_labels[Images[i]]=Class
        total_relevant_images[Class]+=1
    total_relevant_images = dict(total_relevant_images)  
    del total_relevant_images['Thumbs.db']
    del true_labels['Thumbs.db']

    # Looping through descriptors and Images
    for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
        if filename.endswith('.mat'):
            img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            img_actual_path = os.path.join(IMAGE_FOLDER,'Images',filename).replace(".mat",".bmp")
            img_data = sio.loadmat(img_path)
            ALLFILES.append(img_actual_path) 
            ALLFEAT.append(img_data['F'][0]) 

    ALLFEAT = np.array(ALLFEAT, dtype=np.float32)

  
 
    NIMG = ALLFEAT.shape[0] 

    queryimg = randint(0, NIMG - 1)  # randomly selecting a descriptor..
   
    # uncomment the below comments to put query image manually. And comment the above line. Ident accordingly

    # for allfilesIndex in range(len(ALLFILES)): 
        #if(os.path.basename(ALLFILES[allfilesIndex]).split('_')[0]!='13'):
        #           continue
        #queryimg= allfilesIndex


    metric =metric

    if(descriptorTechnique=='BoVW'):
        query_image_path=ImportantVariables.bovw_finalImageSet[queryimg]
    elif(descriptorTechnique=='CNN'):
        query_image_path = ALLFILES[queryimg]
        queryFeature=extract_cnn_features(cv2.imread(query_image_path))
    else:
        query_image_path = ALLFILES[queryimg]
        query = ALLFEAT[queryimg]


    query_image = cv2.imread(query_image_path)


    fig, axs = plt.subplots(1, 1) 

    plt.imshow(cv2.cvtColor(query_image,cv2.COLOR_BGR2RGB))
    axs.set_title(f"Query Image\n(Class: {true_labels.get(os.path.basename(query_image_path))})")
    axs.axis('off')

    SHOW = 15 
    
    if(isSVM):
        X_train, X_test, y_train, y_test = train_test_split(ImportantVariables.histograms, ImportantVariables.bovw_finalImageSetLabels, test_size=0.2, random_state=42)
        svm = train_svm_classifier(X_train, y_train)
        


    elif(descriptorTechnique=='BoVW'):
        query_histogram = ImportantVariables.histograms[queryimg]
        similarities = cosine_similarity([query_histogram], ImportantVariables.histograms)[0]
        bovw_top_indices = np.argsort(similarities)[::-1]

    elif(descriptorTechnique=='CNN'):
        similarities = []
        for filename, features in ImportantVariables.feature_database.items():
            sim = cosine_similarity([queryFeature], [features])[0][0]
            similarities.append((filename, sim))                
            similarities.sort(key=lambda x: x[1], reverse=True)
            cnn_top_results=similarities

    elif(isPCA):
        if(descriptorTechnique=="sift_feature_descriptor"):
            ImportantVariables.PCA_descriptors=ImportantVariables.PCA_descriptors[0]

        reduced_descriptors,pca=apply_pca(ImportantVariables.PCA_descriptors, variance=0.95)
        query_descriptor_pca = pca.transform([query])  # Reduce dimensions
        pca_distances = cvpr_compare(query_descriptor_pca[0],reduced_descriptors,metric)
        pca_sorted_indices = np.argsort(pca_distances)  # Sort by distance


    else:
        dst = []
        for i in range(NIMG):
            candidate = ALLFEAT[i]
            distance = cvpr_compare(query, candidate,metric)
            dst.append((distance, i))
        dst.sort(key=lambda x: x[0])

    if(isSVM):
            # Predict and evaluate
            y_pred = svm.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            conf_matrix = cfm(y_test, y_pred)
            # Plot confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svm.classes_, yticklabels=svm.classes_)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()
            return
    
    num_cols = 5
    num_rows = math.ceil(SHOW / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    if(descriptorTechnique=='BoVW'):
        i=0
        for imgIndex in bovw_top_indices[:SHOW]:
            img = cv2.imread(ImportantVariables.bovw_finalImageSet[imgIndex]) 
            row, col = divmod(i, num_cols)
            if row < num_rows and col < num_cols:  

                axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            axs[row, col].set_title(f"Result {i+1}\n(Class: {true_labels.get(os.path.basename(ImportantVariables.bovw_finalImageSet[imgIndex]))})")#, Dist: {formatted_value})")
            axs[row, col].axis('off')
            i+=1
    
    elif(descriptorTechnique=='CNN'):
        j=0
        for i, (filename, sim_score) in enumerate(cnn_top_results[:SHOW], start=2):
            img=Image.open(os.path.join(os.path.join(IMAGE_FOLDER,Images_subfolder), filename)).convert("RGB")
            row, col = divmod(j, num_cols)
            if row < num_rows and col < num_cols:  
                axs[row, col].imshow(img)
            axs[row, col].imshow(img)
            axs[row, col].set_title(f"Result {j+1}")
            axs[row, col].axis('off')
            j+=1
    elif(isPCA):
            for i in range(SHOW):
                img = cv2.imread(ALLFILES[pca_sorted_indices[i]])                  
                row, col = divmod(i, num_cols)
                if row < num_rows and col < num_cols: 
                    axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                axs[row, col].set_title(f"Result {i+1}\n(Class: {true_labels[os.path.basename(ALLFILES[pca_sorted_indices[i]])]}), ")#Dist: {formatted_value})
                axs[row, col].axis('off')
    
    else:
        for i in range(SHOW):
            img = cv2.imread(ALLFILES[dst[i][1]])
            row, col = divmod(i, num_cols)
            if row < num_rows and col < num_cols: 
                axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            formatted_value = f"{dst[i][0]:.2f}"
            axs[row, col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            axs[row, col].set_title(f"Result {i+1}\n(Class: {true_labels[os.path.basename(ALLFILES[dst[i][1]])]}), Dist: {formatted_value})")
            axs[row, col].axis('off')
    



    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    ################### Calling Compute Precision and Recall fn  #########################

    aps = []
    top_images=[]
    dataset=ALLFILES
    for i in range(1, NIMG + 1):  
        if(descriptorTechnique=='BoVW'):
            top_images=bovw_top_indices[:i]
            dataset=ImportantVariables.bovw_finalImageSet
        elif(descriptorTechnique=='CNN'):
            top_images=cnn_top_results[:i]
        elif(descriptorTechnique=='global_colour_histogram' and isPCA==True):
            top_images=pca_sorted_indices[:i]
        elif(descriptorTechnique=='grid_based_descriptor' and isPCA==True):
            top_images=pca_sorted_indices[:i]
        elif(descriptorTechnique=='sift_feature_descriptor' and isPCA==True):
            top_images=pca_sorted_indices[:i]
            
        else:
            top_images = dst[:i]
        precision, recall = compute_precision_recall(top_images, true_labels.get(os.path.basename(query_image_path)), total_relevant_images.get(true_labels.get(os.path.basename(query_image_path)), 0),dataset,true_labels,descriptorTechnique,isPCA)
        precisions.append(precision)
        recalls.append(recall)
        ap = compute_ap(precisions, recalls)
        aps.append(ap)

    # Compute Mean Average Precision (mAP)
    mAP = np.mean(aps)
    print(f"Mean Average Precision (mAP) of {mAP}:")



    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, marker='o')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    #break

    if(not isPCA):
        confusion_matrix(true_labels,ALLFEAT,ALLFILES,ImportantVariables.descriptorDict,metric,descriptorTechnique)