# Melanoma Image Classifier
## Installation
This is an repository contains a notebook to train, and test an Image Classifier using a dataset of ~11k images of benign/malignant melanomas  
Contents:  
  * Notebook.ipynb: A jupyter notebook containting a step by step guide of the preprocessing, data handling, tarining, testing and using existing models   
  * figures/ : folder with graphs of training  
  * environment.yml: file with list of required python and conda libraries to install  

After downloading all the contents of this repository and navigating to it via your terminal, use this terminal command to create a new conda environment using the environment.yml file:  
```
    conda create --file environment.yml

```
Make sure the correct versions of the following libraries are installed, if you intend to use a GPU:  
  * CUDA® Toolkit 12.3.  
  * cuDNN SDK 8.9.7.

Download and unzip dataset into folder from this link: 
https://d4h9zj04.na1.hs-sales-engage.com/Ctc/L1+23284/d4H9ZJ04/JlF2-6qcW8wLKSR6lZ3pVW3CK5mk3bjl24W70Xbxm8_1Y9MW6DCkTC1cVjJJW1nQDmn4W1-TTW7W6pn65kWVZ8W43hVc92_H4BkW5S6BxJ3LLhkTW3RF7Vj4jZcrGW3Cv3L95xBTlsW6ZCv5-6ddCTNW5PJZ3-97qt3KW1Zm-yY4GZjtqN7mYNnxVWjjlW3nHyGv7QQ1cmW8b9nVy86tZ30W2D87sB1KRMMKW94QyqF9cLKJkW7X5Qct7r-RQsW1SmpQb6bp308W6btVTt71kj13W58V0HG1SdRlqW41N2NG1z7qk7VtdPbn8ylLGxW6rP9T994qYlDW1w7sQV68qFfjW5K5GKB2-jntHW2dKMqs89LXx-W3p2rlT5zY1Qgf49ntLl04

Download models and save them in folder named models:
https://drive.google.com/drive/folders/14oKibld7ge7JNxShWAD_YuynIkJ99OGk?usp=sharing


Folder summary after installs:  
  * Notebook.ipynb 
  * models/ :folder with pretrained models  
  * train/ : folder with the train dataset classified in two folders based on class: Benign and Malignant  
  * test/ : folder with the test dataset classified in two folders based on class: Benign and Malignant  
  * figures/  
  * requirements.txt
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
   * Weights pretrained on ImageNet dataset
   * Random initial weights
  
## Evaluation

| Model                                  | Accuracy  | AUC    |
|----------------------------------------|---------- | -------|
| efficient_net_b6_samplewise.h5         | 0.9386    | 0.9833 |
| efficient_net_b6_featurewise.h5        | 0.9327    | 0.9800 |
| efficient_net_b6_samplewise_40.h5      | 0.9523    | 0.9864 |
| efficient_net_b6_random_featurewise.h5 | 0.8335    | 0.9434 |
