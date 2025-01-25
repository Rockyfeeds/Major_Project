

Project Overview

The project focuses on **Facial Expression Recognition (FER)**, which involves using computer vision techniques to classify a person’s expression (or mood) using a camera. The aim is to identify emotional states based on facial features. 

This is a **classification problem** and has various practical applications in fields such as:
- **Surveillance**
- **Biometrics**
- **Law Enforcement**
- **Marketing**
  
The project compares two popular deep learning models: **DenseNet161** and **ResNet152**, using **PyTorch** and **Transfer Learning**.

---

## **Dataset**
The dataset used for training and testing can be found here:
[Kaggle Dataset](https://www.kaggle.com/apollo2506/facial-recognition-dataset).

---

## **Models Used**
### **1. DenseNet161**
DenseNet161 is a Convolutional Neural Network (CNN) that uses:
- **Dense Connectivity**: Every layer is connected to every other layer.
- **Advantages**:
  - Alleviates the vanishing gradient problem.
  - Strengthens feature propagation and reuse.
  - Requires fewer parameters compared to other architectures.
- **Applications**: Image classification tasks on datasets like CIFAR and ImageNet.
  
**Key Features**:
- Efficient training of deeper networks.
- Utilizes the outputs of all preceding layers.

### **2. ResNet152**
ResNet152 is a Residual Network that introduces:
- **Residual Learning**: Instead of learning the output directly, it learns the residual (difference) with reference to the input.
- **Advantages**:
  - Simplifies optimization in very deep networks.
  - Reduces overfitting and improves performance on unseen data.
- **Applications**: Tasks such as object detection and segmentation in competitions like ImageNet and COCO.

---

## **Transfer Learning**
- **Concept**: Use a pre-trained model (trained on a large dataset) and fine-tune it for a smaller dataset.
- **Steps**:
  1. Freeze the earlier convolutional layers (general feature extraction).
  2. Train only the later layers to adapt to the specific task (e.g., facial expression recognition).
- **Reason**: Early layers extract basic features like edges and gradients, while later layers focus on task-specific features.

---

## **Results**
### **ResNet152**
- **Loss Graph Observation**:
  - Training and validation losses decrease until epoch 15.
  - After epoch 15, training loss continues to decrease, but validation loss plateaus or increases, indicating **overfitting**.
  - Overfitting occurs because the model starts memorizing training data instead of generalizing.
  
### **DenseNet161**
- **Loss Graph Observation**:
  - Training and validation losses decrease until epoch 5.
  - After epoch 5, training loss fluctuates while validation loss remains constant.

---

## **Challenges**
1. Expressions like happiness and surprise may appear visually similar, leading to misclassifications.
2. Resource-intensive process:
   - Higher-quality images and better augmentation (e.g., resizing, cropping) might improve performance but require high-end GPUs to avoid memory issues.

---

## **Conclusion**
- ResNet152 generally outperformed DenseNet161 in this specific task.
- Although results were decent, there’s room for improvement:
  - Better image pre-processing.
  - Experimenting with different learning rates and optimization techniques.

---

## **Future Work**
- Build another image classification model using transfer learning.
- Improve generalization to better handle real-world data.
