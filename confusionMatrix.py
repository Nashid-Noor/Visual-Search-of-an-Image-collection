import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict
import os
import seaborn as sns
from cvpr_compare import cvpr_compare
from extractRandom import extract_cnn_features
import cv2
from cvpr_computedescriptors import ImportantVariables
from sklearn.metrics.pairwise import cosine_similarity


############################ Compute confusion matrix #####################################



def confusion_matrix(true_labels,ALLFEAT,ALLFILES,descriptorDict,metric,descriptorTechnique):

  class_labels = sorted(set(true_labels.values()))
  conf_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
  
  class_labels = list(f"Class {i}" for i in range(1,21))
  conf_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
  class_to_index = {label: index for index, label in enumerate(class_labels)}

  completedClass=[]
  NIMG = ALLFEAT.shape[0]

  Ponce=True
  for imgPath, classValue in true_labels.items():
      if classValue in completedClass:
          continue
      completedClass.append(classValue) 
        
      query_label = f"Class {classValue}"
      query_class_index = class_to_index[query_label]

      top_results=[]
      if(descriptorTechnique=='CNN'):
        queryFeature=extract_cnn_features(cv2.imread(os.path.join('MSRC_ObjCategImageDatabase_v2','Images',imgPath)))
        similarities = []
        for filename, features in ImportantVariables.feature_database.items():
          sim = cosine_similarity([queryFeature], [features])[0][0]
          similarities.append((filename, sim))                
          similarities.sort(key=lambda x: x[1], reverse=True)
          top_results=similarities[:20]

      elif(descriptorTechnique=='BoVW'):
          query_histogram = ImportantVariables.histograms[ImportantVariables.classIndex[classValue]]
          similarities = cosine_similarity([query_histogram], ImportantVariables.histograms)[0]
          bovw_top_indices = np.argsort(similarities)[::-1]
          top_results=bovw_top_indices[:20]

      else:
        queryDescriptor = descriptorDict[imgPath]
        distance = []
        if(descriptorTechnique=='sift_feature_descriptor'):
           queryDescriptor=queryDescriptor[0]
        for i in range(NIMG):
            candidate = ALLFEAT[i]
            d = cvpr_compare(queryDescriptor, candidate, metric)
            distance.append((d, i))
          
        distance.sort(key=lambda x: x[0])
        
        top_results = distance[:20] 

      predicted_counts = defaultdict(int)

      if(descriptorTechnique=='CNN'):
          for img_index,_ in top_results:
             result_label = f"Class {true_labels.get(os.path.basename(img_index))}"
             predicted_counts[result_label] += 1 
      elif(descriptorTechnique=='BoVW'):
         for img_index in top_results:
            result_label = f"Class {true_labels.get(os.path.basename(ALLFILES[img_index]))}"
            predicted_counts[result_label] += 1 
      else:
        for _, img_index in top_results:
            result_label = f"Class {true_labels.get(os.path.basename(ALLFILES[img_index]))}"
            predicted_counts[result_label] += 1 
      
      predicted_counts = dict(predicted_counts)
      for classLabel, labelValue in predicted_counts.items():
          predicted_class_index = class_to_index[classLabel]
          conf_matrix[query_class_index][predicted_class_index] =labelValue
    
  

  #plot confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
  xticklabels=class_labels, yticklabels=class_labels)
  plt.xlabel("Predicted Class")
  plt.ylabel("True Class")
  plt.title("Confusion Matrix of Image Retrieval")
  plt.show()

