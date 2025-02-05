
from cvpr_computedescriptors import computeDescriptors
from cvpr_visualsearch import visualSearch

def selectMetric(descriptorTechnique,isPCA):
      
      if(isPCA):
         return 'mahalanobis'
      
      if(descriptorTechnique=='CNN' or descriptorTechnique=='BoVW' or descriptorTechnique=='SVM'):
         return ''
                
         
      metric=input("Enter the Serial number assosciated with metric you want to use: \n 1.) Euclidean \n 2.) L1 \n 3.) cosine \n 4.) Hamming\n").strip().lower()
      while metric not in ('1', '2', '3','4'):
         print("Invalid choice. Please try again.")
         metric = input("Enter your choice (1/2/3/4): ").strip().lower()
      if(metric=='1'):
          return 'euclidean'
      if(metric=='2'):
          return 'L1'
      if(metric=='3'):
          return 'cosine'
      if(metric=='4'):
          return 'hamming'
 
    

def selectDescriptorTechnique_PCA_SVM():

   choice= input("Enter the Serial number assosciated with descriptor technique you want to implement: \n 1.) Global Colour Histogram \n 2.) Spatial Grid \n 3.) SIFT \n 4.) Bag of Visual Words \n 5.) CNN\n 6.) SVM \n").strip().lower()
   while choice not in ('1', '2', '3','4','5','6'):
      print("Invalid choice. Please try again.")
      choice = input("Enter your choice (1/2/3/4/5): ").strip().lower()
   
   if choice == '1':
      choiceOfDescriptor='global_colour_histogram'
   elif choice=='2':
      choiceOfDescriptor='grid_based_descriptor'
   elif choice=='3':
      choiceOfDescriptor='sift_feature_descriptor'
   elif choice=='4' or choice=='6':
      choiceOfDescriptor='BoVW'
   elif choice=='5':
         choiceOfDescriptor='CNN'

   isPCA=False
   if(choice!='4' and choice!='6' and choice!='5'):
      pca_apply=input("Enter Y if you want to apply PCA else Enter N \n")
      while  pca_apply not in ('Y', 'N', 'y','n'):
         print("Invalid choice. Please try again.")
         pca_apply = input("Enter your choice (Y/N/y/n): ").strip().lower()
      
      if(pca_apply=='Y' or pca_apply=='y'):
            isPCA=True


   isSVM=False
   if(choice=='6'):
         isSVM=True
   return choiceOfDescriptor,isPCA,isSVM

def getQuantizationValues(descriptorTechnique):
   
   if(descriptorTechnique=='global_colour_histogram'):
      return input("Enter the number of colour bins:\n"),None
   
   if(descriptorTechnique=='grid_based_descriptor'):
      colour_bins =input("Enter the number of colour bins:\n")
      texture_bins  =input("Enter the number of texture bins:\n")
      return colour_bins,texture_bins

   return None,None

if __name__ == "__main__": 
  descriptorTechnique,isPCA,isSVM=selectDescriptorTechnique_PCA_SVM()
  colour_bins,texture_bins=getQuantizationValues(descriptorTechnique)
  print("Starting Descriptors extraction")
  computeDescriptors(descriptorTechnique,isPCA,isSVM,colour_bins,texture_bins)
  print("Starting Visual Search")
  metric = selectMetric(descriptorTechnique,isPCA)
  visualSearch(descriptorTechnique,isPCA,isSVM,metric)

