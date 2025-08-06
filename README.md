# Microplastics Segmentation and Classification with U-Net, VGG16, and Logistic Regression

This repository contains the experimental code developed for the segmentation and classification of microplastics using U-Net, VGG16, and Logistic Regression. The project was developed as part of a Software Engineering Bachelor's thesis at Vilnius University.

## Overview

The goal of this project is to segment and classify microplastics in images using deep learning techniques. The pipeline includes:

1. **Segmentation**: Using a U-Net model to segment microplastics in images.
2. **Feature Extraction**: Leveraging a pre-trained VGG16 model to extract features from segmented regions.
3. **Classification**: Using Logistic Regression to classify the extracted features into predefined categories.

## Features

- **U-Net Model**: A convolutional neural network architecture for image segmentation.
- **VGG16 Feature Extraction**: Pre-trained VGG16 model for extracting meaningful features from segmented regions.
- **Logistic Regression Classifier**: A lightweight and efficient classifier for microplastic classification.
- **Data Augmentation**: Techniques to improve model generalization.
- **Evaluation Metrics**: Includes accuracy, precision, recall, F1-score, and Jaccard index.

## Dataset

The dataset consists of images and corresponding masks for microplastic segmentation.

## Results

The project achieves high accuracy in segmenting and classifying microplastics. Detailed evaluation metrics and visualizations are provided in the notebook.

## Acknowledgments

This project was developed as part of a Software Engineering Bachelor's thesis at Vilnius University. Special thanks to the university and the advisors for their guidance and support.

## Author

Kristupas Jon (https://github.com/KristupasJon)
