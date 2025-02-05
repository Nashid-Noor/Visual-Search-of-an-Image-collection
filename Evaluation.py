import os
# Initialize lists for precision and recall
precisions = []
recalls = []


def compute_precision_recall(similar_images, true_label, total_relevant,ALLFILES,true_labels,descriptorTechnique,isPCA):
    relevant_count = 0
    retrieved_count = len(similar_images) 
    if(descriptorTechnique=='BoVW'):
        for img_index in similar_images:
            img_path = ALLFILES[img_index]
            if true_labels.get(os.path.basename(img_path)) == true_label:
                relevant_count += 1
    elif(descriptorTechnique=='CNN'):
        for img_path,_ in similar_images:
             if true_labels.get(os.path.basename(img_path)) == true_label:
                relevant_count += 1
    elif(isPCA):
         for img_index in similar_images:
            img_path = ALLFILES[img_index]
            if true_labels.get(os.path.basename(img_path)) == true_label:
                relevant_count += 1
    else:
        for _, img_index in similar_images:
            img_path = ALLFILES[int(img_index)]
            if true_labels.get(os.path.basename(img_path)) == true_label:
                relevant_count += 1
            
    precision = relevant_count / retrieved_count if retrieved_count > 0 else 0
    recall = relevant_count / total_relevant if total_relevant > 0 else 0
    
    return precision, recall


def compute_ap(precisions, recalls):
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap