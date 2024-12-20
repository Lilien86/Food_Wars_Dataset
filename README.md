# Food Image Classification 🍔

A CNN model to recognize an images of "pizza", "suchi", "steak". It uses a custom dataset base on [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with basic data augmentation.

## Data Exemples

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/a084014d-7836-473d-83e4-f04c4d1a9509" alt="Image 1" style="margin-right: 10px; width: 150px;"/>
  <img src="https://github.com/user-attachments/assets/2a131899-1607-4fa6-bfdd-22d00f9a6d53" alt="Image 2" style="margin-right: 10px; width: 150px;"/>
  <img src="https://github.com/user-attachments/assets/eaf948b2-7eec-4884-b403-bc12e0d113d9" alt="Image 3" style="width: 150px;"/>
</div>


## Data Preprocessing

The images are preprocessed using the following steps:
- **Resize**: The images are resized to `64x64` pixels using `transforms.Resize`.
- **ToTensor**: The images are converted to PyTorch tensors using `transforms.ToTensor()`.
- **Data Augmentation**: During training, basic data augmentation is applied using `transforms.TrivialAugmentWide`, which includes random transformations like rotation, flipping, brightness/contrast adjustment, etc., to improve the model's generalization capabilities.

<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/413ba192-a020-4312-864f-516e47c8c520" width="95%" />
  <br />
  <strong></strong>
</p>


## Model Architecture

Simple Convolutional Neural Network (CNN) inspired by the TinyVGG model. The architecture is as follows:

1. **Conv Block 1**:
   - 2 Convolutional layers with ReLU activations.
   - MaxPooling layer with a kernel size of 2x2.

2. **Conv Block 2**:
   - 2 Convolutional layers with ReLU activations.
   - MaxPooling layer with a kernel size of 2x2.

3. **Fully Connected Layer**:
   - A flattened tensor is passed through a fully connected layer to output the class probabilities.


  ## Training
  1. **Optimizer**: Adam optimizer
  2. **Loss Function**: CrossEntropyLoss for multi-class classification
  3. **Epochs**: 5 epochs
  4. **Metrics**:Training and testing losses and accuracies are printed for each epoch

## Results
<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/c035a4b8-aae7-47b2-80e7-ba942a53f048" width="75%" />
  <br />
  <strong>Data transformed</strong>
</p>

