# Microplastics Segmentation and Classification with U-Net, VGG16, and Logistic Regression

This repository contains the experimental code developed for the segmentation and classification of microplastics using U-Net, VGG16, and Logistic Regression. The project was developed as part of a Software Engineering Bachelor's thesis at Vilnius University.

<img width="1824" height="440" alt="image" src="https://github.com/user-attachments/assets/368e727f-b417-408e-a1e7-a94801557ce7" />
<img width="1826" height="441" alt="image" src="https://github.com/user-attachments/assets/eb1ec69a-5f30-41eb-90ac-6436862f1209" />
<img width="1826" height="437" alt="image" src="https://github.com/user-attachments/assets/60ccc459-1eda-4eff-9f8c-25625f5de41b" />


## Overview

This project addresses the challenging problem of microplastic identification in environmental samples through an innovative computer vision pipeline. Unlike traditional supervised learning approaches, this work tackles the complex scenario of having only unlabeled images with poor quality conditions, requiring the development of novel unsupervised and semisupervised techniques.

**The Challenge**
Working with a dataset containing only raw microscopic images and corresponding segmentation masks, without any classification labels. The images exhibited poor quality conditions typical of environmental sampling, making standard deep learning approaches ineffective.

**The Solution**
A sophisticated three stage pipeline that combines segmentation, feature extraction, and unsupervised classification:

1. **Semantic Segmentation**: A U-Net architecture identifies and segments microplastic particles from complex environmental backgrounds, handling the binary segmentation task of separating particles from debris and organic matter.

2. **Deep Feature Extraction**: A pre trained VGG16 network extracts high dimensional feature representations from segmented microplastic regions, leveraging transfer learning to capture meaningful visual patterns despite the limited and unlabeled dataset.

3. **Unsupervised Classification**: KMeans clustering applied to extracted features automatically discovers natural groupings in the data, with pseudo labels assigned to create three distinct microplastic categories: oval, string, and other morphological types. A Logistic Regression classifier then learns to map features to these discovered categories.

**Innovation**: The key innovation lies in the pseudo labeling strategy that transforms an unsupervised problem into a supervised one, combined with extensive data augmentation to overcome dataset limitations. 
**Practical Impact for Scientists**
This approach demonstrates that minimal manual labeling effort (only a small CSV file with basic annotations) can achieve robust classification results, significantly reducing the time scientists need to spend on tedious data labeling tasks. Even as a proof of concept, this pipeline shows that advanced machine learning can provide effective workarounds for the traditional grunt work of extensive dataset annotation, allowing researchers to focus on analysis rather than data preparation.

## Dataset Challenges and Solutions

### **Major Challenge: Unlabeled Dataset with Poor Image Quality**

This project faced significant challenges that required innovative solutions:

- **Unlabeled Data**: The dataset contained only raw images and segmentation masks without classification labels
- **Poor Image Quality**: Images were of suboptimal quality, making traditional supervised learning approaches difficult
- **Limited Data**: Small dataset size required careful handling to prevent overfitting
- **Manual Labeling Burden**: Creating labels manually would have been extremely timeconsuming and prone to human error

### **Solutions Implemented**

#### **1. KMeans Clustering for Pseudo Labeling**
Since the dataset lacked classification labels, I implemented an unsupervised approach:
- Extracted features from segmented regions using pretrained VGG16
- Applied KMeans clustering (k=3) to group similar microplastic features
- Assigned pseudolabels based on dominant patterns in each cluster
- This approach automatically categorized microplastics into three classes: oval, string, and other
- Only a small CSV file with limited manual labels was used for validation, proving that extensive manual labeling by scientists is unnecessary for effective results

#### **2. Data Augmentation for Small Dataset**
To address the limited dataset size, I implemented several augmentation techniques:
- **Horizontal Flipping**: 50% probability to flip images left or right
- **Random Rotation**: 0°, 90°, 180°, or 270° rotations
- **Brightness Adjustment**: Random brightness changes (±0.1) to simulate varying lighting conditions
- These techniques effectively expanded the training data while maintaining realistic variations

## Model Architecture Justification

### **U-Net for Segmentation**
- **Chosen because**: Excellent performance on biomedical image segmentation tasks
- **Skip connections**: Preserve details essential for accurate microplastic boundaries
- **Encoder decoder structure**: Captures both local and global context effectively
- **Binary output**: Perfect for microplastic vs. background segmentation

### **VGG16 for Feature Extraction**
- **Pretrained on ImageNet**: Leverages learned features from millions of images
- **Robust feature extraction**: Captures texture, shape, and color patterns effectively
- **Transfer learning**: Reduces training time and improves performance on limited data
- **Global average pooling**: Provides fixed size feature vectors regardless of input patch size

### **Logistic Regression for Classification**
- **Computational efficiency**: Fast training and inference on extracted features
- **Interpretability**: Clear understanding of feature importance and decision boundaries
- **Class balancing**: Built in support for handling imbalanced datasets
- **Regularization**: L2 regularization prevents overfitting on limited data

## Technical Implementation Details

### **Hyperparameter Optimization**
- Grid search with 3-fold cross-validation for optimal parameters
- Tested solvers: 'lbfgs', 'sag', 'saga', 'newton-cg'
- Regularization strength (C): [0.01, 0.1, 1, 10]
- Class weighting: 'balanced' to handle class imbalances

### **Evaluation Metrics**
- **Segmentation**: Accuracy, Precision, Recall, F1 score, Jaccard Index
- **Classification**: Macro averaged precision and recall, confusion matrices
- **Cross-validation**: 3-fold CV for robust performance estimation

## Features

- **U-Net Model**: A convolutional neural network architecture for image segmentation.
- **VGG16 Feature Extraction**: Pretrained VGG16 model for extracting meaningful features from segmented regions.
- **Logistic Regression Classifier**: A lightweight and efficient classifier for microplastic classification.
- **Data Augmentation**: Techniques to improve model generalization.
- **Evaluation Metrics**: Includes accuracy, precision, recall, F1 score, and Jaccard index.
- **Pseudo Labeling**: KMeans clustering approach to handle unlabeled data.
- **Class Balancing**: Weighted loss functions to handle imbalanced datasets.

## Results

The project achieves high accuracy in segmenting and classifying microplastics. Detailed evaluation metrics and visualizations are provided in the notebook.

## Acknowledgments

This project was developed as part of a Software Engineering Bachelor's thesis at Vilnius University. Special thanks to the university and the advisors for their guidance and support.
