# Melanoma Image Classifier
## Installation
This is an repository contains a notebook to train, and test an Image Classifier using a dataset of ~11k images of benign/malignant melanomas  
Contents:  
  * Notebook.ipynb: A jupyter notebook containting a step by step guide of the preprocessing, data handling, tarining, testing and using existing models  
  * models/ :folder with pretrained models  
  * train/ : folder with the train dataset classified in two folders based on class: Benign and Malignant  
  * test/ : folder with the test dataset classified in two folders based on class: Benign and Malignant  
  * figures/ : folder with graphs of training  
  * environment.yml : file with list of required python and conda libraries to install  

After downloading all the contents of this repository and navigating to it via your terminal, use this terminal command to create a new conda environment using the evironment.yml file:  
```
    conda create --file environment.yml

```
Make sure the correct versions of the following libraries are installed, if you intend to use a GPU:  
  * CUDA® Toolkit 12.3.  
  * cuDNN SDK 8.9.7.

Once the  installations are complete, your ready to use the notebook, of course selecting the proper kernel (tf) from conda



## Model
For our classifier we chose EfficientNet-B6 as our backbone model because it balances well between efficiency and accuracy.   
We didn't use the top (fully connected) layer of EfficientNet as we want a binary classifier.  
Our new added toplayers of our binary classifier includes:  
 * a global average pooling 2D layer for feature extraction from EfficientNet
 * a dense layer with ReLu activation (because it is less susceptible to vanishing gradients)
 * a dense layer with softmax activation for classification.

## Preprocessing  
We included multiple processing and data augmentation techinques using tensorflow's built-in functions.  
Image Processing:  
   * Rescaling pixel values to [0,1] interval
   * Samplewise centtering, centering mean of each image at 0
   * Featurewise centering, subtracting the mean image out of each image
  
Augmmentation:
   * Rotation
   * Shear
   * Horizontal/Vertical flips
  
## Training  
Parameters for training used:   
   * 85/15 train/val split
   * learning rate = 0.0001 with Adam optimizer to adjust lr as training goes
   * categorical crossentropy loss
   * batch size = 16 for training and 8 for validation
   * using accuracy and AUC for metrics
  
We trained multiple versions of our model with multiple preprocessing techniques:  
   * Using rescaling and featurewise centering
   * Using rescaling and samplewise centering
  
We used different initializations for EfficientNet-B6:  
   * Using model weights when pretrained on ImageNet dataset for which we trained over 10 epochs
   * Random initial weights for which we trained over 20 epochs  
  
## Evaluation

| Model                                  | Accuracy | AUC   |
|----------------------------------------|----------|-------|
| efficient_net_b6_samplewise.h5         | 0.935    | 0.958 |
| efficient_net_b6_featurewise.h5        | 0.900    | 0.938 |
| efficient_net_b6_random_samplewise.h5  | 0.831    | 0.931 |
| efficient_net_b6_random_featurewise.h5 | 0.871    | 0.945 |
