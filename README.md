# Food Image Classification 🍔

A CNN model to recognize an images of "pizza", "suchi", "steak". It uses a custom dataset base on [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with basic data augmentation.

## Data Preprocessing

The images are preprocessed using the following steps:
- **Resize**: The images are resized to `64x64` pixels using `transforms.Resize`.
- **ToTensor**: The images are converted to PyTorch tensors using `transforms.ToTensor()`.
- **Data Augmentation**: During training, basic data augmentation is applied using `transforms.TrivialAugmentWide`, which includes random transformations like rotation, flipping, brightness/contrast adjustment, etc., to improve the model's generalization capabilities.

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
