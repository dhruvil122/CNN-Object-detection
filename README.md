# CNN-Object-detection
This is the repository of object detection with CNN. The repo includes several types of models with different epochs and in some cases different data preprocessing files.


So far, the maximum of around 60 percent of accuracy is achieved with a validation set containing 160 unlabelled images spanning across 8 different categories. Each catergory contains around 20 images.


## Preprocessing
The training set images were reduces with 128x128 resolution. 

## models
Most of the model has 3 convolution layers with both relu and maxpool layers. At first, The input will come into the convolution 1 layer. Relu and maxpool is applied. In most models, A dropout function is also used to prevent overfitting. Till now (at 60 percent validation accuracy), the model is only trained with 10 epochs with the latest update. 

## Files

1. All the folders have their own individual model.py, model.pkl and data.py files. 
2. The data.py file is for preprocessing of the labelled images.
3. Some of the models also include .csv files which can be helful to validate the data results directly on kaggle.
4. The repo includes both training and vallidation snippet which you can use to train your own model to see the results. The validation data can be found from [here.](url).
