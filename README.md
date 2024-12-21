# Capturing Images with Words: A Transformer-Based Image Captioning Model

## Authors

**Manoj Alexender**  
*Department of CIS, University of Michigan-Dearborn, Dearborn, MI, USA*  
Email: manojale@umich.edu  

**Gajalakshmi Soundarrajan**  
*Department of CIS, University of Michigan-Dearborn, Dearborn, MI, USA*  
Email: sgaja@umich.edu  

---

## Abstract

Image captioning, a complex task at the crossroads of **computer vision** and **natural language processing**, involves converting visual content into meaningful and coherent textual descriptions. This project focuses on the development of a custom **transformer-based image captioning model** aimed at enhancing caption quality and accuracy. By leveraging pre-trained **Convolutional Neural Networks (CNNs)** for image feature extraction and utilizing advanced transformer architectures for caption generation, the model demonstrates its ability to generate contextually relevant captions.  

The model was evaluated on the **Flickr8k dataset**, with performance measured using BLEU scores, highlighting the effectiveness of transformer-based methods for image captioning tasks.

---

## Keywords  

Image Captioning, Transformer Model, Feature Extraction, Natural Language Processing (NLP), Computer Vision (CV), Vision-Language Models, Contextual Caption Generation

---

## Introduction

The ability to generate descriptive captions from images has widespread applications, ranging from enhancing accessibility for visually impaired individuals to improving content recommendation systems. This project combines the strengths of modern **image processing** and **natural language generation** techniques to address this challenge.  

The key contributions include:  
1. Implementing and evaluating **transformer-based** and **LSTM-based models with attention** for caption generation.  
2. Leveraging pre-trained models like **ResNet** for feature extraction.  
3. Exploring methods that balance computational efficiency and accuracy.

---

## Overview of Models

This project explores four methods for image captioning:  

1. **ResNet-Transformer (PyTorch):**  
   Combines ResNet-18 for feature extraction with a Transformer Decoder for text generation.

2. **CNN-LSTM (PyTorch):**  
   Uses a Convolutional Neural Network (CNN) to extract image features, followed by an LSTM for sequential text generation.

3. **Attention-Based CNN-LSTM (PyTorch):**  
   Enhances the CNN-LSTM model with an **Attention Mechanism**, allowing the model to focus on specific image regions.

4. **Transformer-Based Model (TensorFlow 1.9.1):**  
   Implements an **end-to-end Transformer architecture** in TensorFlow to generate captions without LSTM-based recurrence.

---

## Dataset: Flickr8k  

The **Flickr8k Dataset** is used for training and evaluation. It contains:  
- **8,000 images** in `.jpg` format.  
- **40,000 captions** (five captions per image).  

### Folder Structure:

```
data/
├── Flickr8k_Dataset/          # Image files
├── Flickr8k_text/             # Caption annotations
│   ├── Flickr8k.token.txt     # Captions
```

---

## Requirements

Install the following dependencies:

```bash
pip install torch torchvision tensorflow==1.9.1 pandas numpy matplotlib tqdm
```

TensorFlow **1.9.1** is required specifically for the fourth model.

---

## Project Structure

```
.
├── README.md
├── Image_Captioning_Transformers_pytorch.ipynb            # Notebook 1: ResNet + Transformer (PyTorch)
├── image-captioning_LSTM.ipynb              # Notebook 2: CNN + LSTM (PyTorch)
├── image_captioning_using_attention.ipynb   # Notebook 3: Attention-Based CNN-LSTM (PyTorch)
├── Image_Captioning_using_Transformers_TF.ipynb  # Notebook 4: Transformer-based model (TF 1.9.1)
├── data/                                    # Flickr8k Dataset
```

---
## Pre-recorded Presentation Video

``` https://drive.google.com/drive/folders/1XLLTf7awft-R8du7idvgkLpnLXR2y3T0 ```

## Dataset Link, Presentation and Report Link 

``` https://drive.google.com/drive/folders/1uCi7SEJgELaYK9uzRt4PMDcmy29oNixp ```

## Youtube Video Link 

``` https://youtu.be/539H5MwR__Q ```

## Steps to Run

### 1. Dataset Setup  

The dataset is automatically downloaded and extracted by running the following code in any notebook:

```python
import os
import tensorflow as tf

# Download Flickr8k dataset
flickr_image_folder = '/Flickr8k_Dataset/'
flickr_annotation_folder = '/Flickr8k_text/'

# Image dataset
if not os.path.exists(os.getcwd() + flickr_image_folder):
    tf.keras.utils.get_file('Flickr8k_Dataset.zip',
                            origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                            extract=True)
# Annotation dataset
if not os.path.exists(os.getcwd() + flickr_annotation_folder):
    tf.keras.utils.get_file('Flickr8k_text.zip',
                            origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                            extract=True)
```

---

### 2. Running the Models  

Run the notebooks sequentially:  

#### **Model 1: ResNet + Transformer**  
**Notebook**: `Image_Captioning_ResNet.ipynb`  
- Framework: PyTorch  
- Description: Uses ResNet-18 for feature extraction and a Transformer Decoder for text generation.  

#### **Model 2: CNN + LSTM**  
**Notebook**: `image-captioning_LSTM.ipynb`  
- Framework: PyTorch  
- Description: Combines CNN for image features with LSTM for sequential caption generation.  

#### **Model 3: Attention-Based CNN-LSTM**  
**Notebook**: `image_captioning_using_attention.ipynb`  
- Framework: PyTorch  
- Description: Introduces an Attention Mechanism for context-based image captioning.  

#### **Model 4: Transformer-Based Model**  
**Notebook**: `Image_Captioning_using_Transformers_TF.ipynb`  
- Framework: TensorFlow 1.9.1  
- Description: Implements an end-to-end Transformer for sequence generation, optimized for image captioning.

---

## Methods Table  

| **Method**                     | **Framework**    | **Description**                                          |
|--------------------------------|------------------|--------------------------------------------------------|
| **ResNet + Transformer**       | PyTorch          | ResNet-18 extracts image features; Transformer generates captions. |
| **CNN + LSTM**                 | PyTorch          | CNN extracts image features; LSTM generates text sequentially.     |
| **Attention-Based CNN-LSTM**   | PyTorch          | CNN + LSTM with an Attention Mechanism for context focus.          |
| **Transformer (TensorFlow 1.9.1)** | TensorFlow       | End-to-end Transformer model for image captioning.                 |

---

## Results  

The **Transformer-based model** demonstrated superior performance, particularly in generating longer and contextually accurate captions due to its self-attention mechanism.  

Example Captions Generated:  

1. **"man is sitting on bench with woman and man who is looking at the camera"**  
2. **"man in red shirt is climbing rock"**  
3. **"a boy in a red shirt is standing on a rock overlooking a stream"**  

---

## Future Improvements  

1. Incorporate **Vision Transformers (ViT)** for hierarchical image feature extraction.  
2. Use larger datasets like **MS-COCO** for improved generalization.  
3. Evaluate model performance using advanced metrics such as **CIDEr** or **METEOR**.  

---

## References  

1. Vinyals, O., Toshev, A., et al. *Show and Tell: A Neural Image Caption Generator.*  
2. Xu, K., Ba, J., et al. *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.*  
3. Cornia, M., et al. *Meshed-Memory Transformer for Image Captioning.*  
4. Radford, A., et al. *CLIP: Learning Transferable Visual Models from Natural Language Supervision.*  
5. Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*  

---

## Contact  

**Manoj Alexender**  
Email: `manojale@umich.edu`  

**Gajalakshmi Soundarrajan**  
Email: `sgaja@umich.edu`  
