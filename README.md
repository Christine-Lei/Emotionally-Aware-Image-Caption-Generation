# Emotionally-Aware-Image-Caption-Generation
Authors: Tomisin Adeyemi, Christine Lei, and MinJoo Kim

## Overview
This project aims to study emotion recognition through transfer learning, employing the student-teacher model architecture. We utilize the Scoratis dataset, comprising 2075 unique images and 12489 total data points, as our target dataset and the basis for our Teacher Model. Each image is annotated with captions and emotions. Notably, repeated images exhibit varying emotions across subject annotations. We utilize the popular COCO Dataset, comprising of over 100k+ image-caption pairs, as our source dataset and the basis for the Student Model. The goal of this project is to generate emotions on the COCO dataset based on our trained teacher model. Ultimately, after augmenting the COCO dataset with emotions and building student and teacher models, we experiment with different techniques for finetuning.

## Datasets
- COCO dataset: raw data imported into code from the huggging face API. The dataset containing the noisy predictions, `coco_predictions.csv`, was too large to upload to github.
- Scoratis dataset: cleaned version contained in [`cleaned_data.csv`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/cleaned_data.csv). Images are too big to upload.

## Methodology
![Model Sketch](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/assets/98556351/3cf2db35-6b0a-49b4-b089-e596a0a8239e)
Step 1. Teacher model training:  
The teacher model is trained on the Socratis dataset, and consists of a VisualBERT layer followed by a fully connected layer. Since VisualBERT takes in image embeddings rather than raw embeddings, we use a pre-trained Fast R-CNN to extract the needed image embeddings and variables. We use a fully connected layer after the VisualBERT layer to map the high-dimensional pooled output from VisualBERT to a lower-dimensional space corresponding to the number of emotions. 
Step 2. COCO prediction and sampling:
We used the teacher model to create to make emotion predictions on COCO. We decided on p = 4, since the average number of emotions per image in Scoratis is 3.98. 
Step 3. Pre-training Student model: 
1. Randomly sample 10% of the data from our dataset, since it is simply too big (118,287rows) and 2. Change to a simpler student model. For the text processing component is the embedding layer that uses a vocabulary size matching BERT's base model. Following the embeddings, the model employs a GRU with 256 hidden units to process the sequence of words. On the visual side, the model processes images through a sequence of convolutional layers that increase in complexity. These layers extract a hierarchy of features from the images, starting from basic edges and textures to more complex patterns, by using 3x3 kernels and stride settings that gradually reduce the spatial dimensions. An adaptive average pooling layer then condenses these features into a consistent size output, setting the stage for merging with the text data. The fusion of text and image features is straightforward: the model concatenates these features into a single vector. Following the fusion, the model uses fully connected layers, including a ReLU-activated layer, to further process the combined features and map them to the final output. The output layer, activated by a sigmoid function, where each label is determined independently.
Step 4. Fine Tuning:
Given our source dataset COCO (A), which includes COCO and the labeled dataset from our teacher model prediction, and target dataset, Socratis (B), we will experiment with 2 fine tuning approaches. In both cases, initialize the parameters of the teacher model (B) with the pre-trained weights of the student model (A).
AnB: finetuning but keeping all the layers frozen
AnB+: finetuning but not freezing the layers
 
## Preprocessing
- **Images**: For the Teacher Model, we employ Fast R-CNN for visual feature extraction; for the Student Model, we use simpler resizing and transforming techniques.
- **Text**: We use the pretrained BERT Tokenizer to tokenize the captions.

## Model Architecture
### Teacher Model
- [`Socratis_Teacher_Model.ipynb`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/Socratis_Teacher_Model.ipynb)
- **Architecrure**: We used the pretrained VisualBERT model, designed for handling both textual and visual inputs; followed by a fully connected linear layer to map the high-dimensional pooled outputs to our emotion classes.
-  The trained teacher model was used to infer emotion probabilities for the COCO dataset, the code is located in the file [`COCO Prediction + Sampling.ipynb`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/COCO%20Prediction%20%2B%20Sampling.ipynb).

### Student Model
- [`studentModel.ipynb`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/studentModel.ipynb)
- **Text Processing**: Embedding layer utilizing BERT's pretrained embeddings and a GRU layer.
- **Visual Processing**: Series of convolutional layers with adaptive average pooling.
- **Fusion and Output**: Concatenation of text and image features, followed by fully connected layers and a sigmoid-activated output layer.

## Training
- **Optimizer**: Adam with a learning rate of 0.01.
- **Loss Function**: Binary cross-entropy with logits, ideal for multilabel classification.

## Fine-tuning
The student model, pre-trained on COCO, is fine-tuned on the Scoratis dataset to enhance its specificity in emotion recognition by leveraging learned features from a large-scale generic dataset.

## Results and Discussion
Detailed analysis of model performance, including accuracy metrics and discussions on the effectiveness of the teacher-student model architecture in multimodal emotion recognition.

## Reference 
inspired by [Socratis: Are large multimodal models emotionally aware?](https://arxiv.org/abs/2308.16741), [How transferable are features in deep neural
networks?](https://proceedings.neurips.cc/paper_files/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf), and [Billion-scale semi-supervised learning for image classification](https://arxiv.org/abs/1905.00546)
