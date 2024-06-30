# Sports Image Classifier

## Introduction

This project is an image classifier that can classify sports images into different categories such as football, basketball, baseball, golf, volleyball, hockey, and sky diving. It uses a pre-trained ResNet50 model with custom classification layers to perform the classification.

## Dataset

The dataset used in this project contains sports images divided into three sets: train, test, and validation. The dataset is available in CSV format, containing file paths, labels, and data set information. The target sports categories were selected as football, basketball, baseball, golf, volleyball, hockey, and sky diving.

## Preprocessing

The images in the dataset are preprocessed before being used for training, testing, and validation. Preprocessing involves resizing the images to the required input size of the ResNet50 model (224x224 pixels) and converting them to NumPy arrays.

## Model Architecture

The classifier uses a pre-trained ResNet50 model without the final classification layers. Custom classification layers are added on top of the base model. The model is then compiled using the Adam optimizer with a learning rate of 0.001 and sparse categorical cross-entropy loss.

## Training

The model is trained on the training data using an image data generator for data augmentation. The training data is augmented by applying random rotations, horizontal flips, and shifts. The model is trained for 5 epochs with a batch size of 32.

## Evaluation

The model is evaluated on the test data to measure its performance. The evaluation includes calculating the test accuracy of the model.

## Prediction

The trained model is used to make predictions on new images. A function is provided to preprocess a single image, make a prediction using the trained model, and display the image along with the predicted class label.

## Demo

A live demo of the image classifier is provided using Gradio. The Gradio interface allows users to upload an image, and the classifier will make a prediction and display the image along with the predicted class label.

## Docker
1.Build docker image : Open a terminal/command prompt, navigate to the directory containing your files, and execute the following command to build the Docker image:

`docker build -t fastapi-app .`

2.Run the container :

`docker run -d -p 8000:80 fastapi-app`

3.Access your app using postman

## Postman

![330156-1652257077_EditgolfsportArtboard1](https://github.com/aybstain/Image_classifier_ResNet/assets/103702856/144d6ccb-488e-455d-8537-b4cbc97c42bd)

<img width="596" alt="Capture" src="https://github.com/aybstain/Image_classifier_ResNet/assets/103702856/02e0581e-fb0d-4e1c-9373-31bee3e66b10">

![2f3dda4b-ca81-4d65-956d-ce85d7a3f592](https://github.com/aybstain/Image_classifier_ResNet/assets/103702856/5c28f55a-85ff-4a70-a4fa-3de214a1ca2f)

<img width="627" alt="Capture1" src="https://github.com/aybstain/Image_classifier_ResNet/assets/103702856/2be0a75c-4272-4492-819c-d52fd7f3343d">

## Requirements

- pandas
- numpy
- keras
- tensorflow
- matplotlib
- gradio

## How to Use

1. Clone the repository: `git clone https://github.com/aybstain/Image_classifier_ResNet.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the live demo: `py -m gradio_app.py`
4. Run fastapi app : `py -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8080 --reload`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
