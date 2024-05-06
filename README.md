# Emotionally-Aware-Image-Caption-Generation
Authors: Tomisin Adeyemi, Christine Lei, and MinJoo Kim

## Overview
This project aims to study emotion recognition through transfer learning, employing the student-teacher model architecture. We utilize the Scoratis dataset, comprising 2075 unique images and 12489 total data points, as our target dataset and the basis for our Teacher Model. Each image is annotated with captions and emotions. Notably, repeated images exhibit varying emotions across subject annotations. We utilize the popular COCO Dataset, comprising of over 100k+ image-caption pairs, as our source dataset and the basis for the Student Model. The goal of this project is to generate emotions on the COCO dataset based on our trained teacher model. Ultimately, after augmenting the COCO dataset with emotions and building student and teacher models, we experiment with different techniques for finetuning.
![Model Sketch](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/assets/98556351/3cf2db35-6b0a-49b4-b089-e596a0a8239e)

## Datasets
- COCO dataset: raw data imported into code from the huggging face API. The dataset containing the noisy predictions, `coco_predictions.csv`, was too large to upload to github.
- Scoratis dataset: cleaned version contained in [`cleaned_data.csv`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/cleaned_data.csv). Images are too big to upload.

## Methodology
- Data preprocessing: We started by preprocessing the Socratis dataset, which contains 2075 unique images and approximately 12k data points total (many images were repeated with different emotions each time). However, given that there were around 983 unique emotions, some of which were empty or did not seem to be emotions at all, we cleaned the dataset by only keeping emotions that appeared more than 100 times in the data. This reduced the total number of unique emotions from 983 to 29. Lastly, the images were one-hot encoded for easy model interpretation. For the COCO Dataset, not much preprocessing on the original data was needed as it had already been preprocessed.

- Teacher model: The teacher model is trained on the Socratis dataset, and consists of a VisualBERT layer (a variant of BERT that interprets both textual and visual inputs) followed by a fully connected layer. Since VisualBERT takes in image embeddings rather than raw embeddings, we use a pre-trained Fast R-CNN to extract the needed image embeddings and variables. We use a fully connected layer after the VisualBERT layer to map the high-dimensional pooled output from VisualBERT to a lower-dimensional space corresponding to the number of emotions. The architecture of the teacher model is shown to the right.

- Training teacher model on Scoratis: We use Binary cross-entropy with logit loss as the loss function.  Adam is configured with a learning rate of 0.01 to optimize the parameters of the teacher model. BCEWithLogitsLoss is perfect for us because it is suited for tasks with labels that are not mutually exclusive, as it combines a sigmoid activation with the binary cross-entropy loss in a single function. In the training loop, for each epoch, the model parameters are updated to minimize the loss calculated between the predicted probabilities and the actual labels. The results will be discussed later on. 
COCO prediction and sampling: first we did similar data preprocessing with the COCO dataset in order to get the text and visual embeddings that are suitable for the teacher model. Since the COCO dataset does not have emotion labels associated with the image, we used the teacher model just created to make predictions on COCO. Our goal is to output the probability that the image-caption pair belongs to one of the 29 emotion classes. In this case, ‘P’ would be the classes with the P highest scores for each image that would be retained and K means that you would pick K after making all predictions. We experimented with different values of P(p >0.5) and K, and finally decided k = 4, since the average number of emotions per image in Scoratis is 3.98. 
- Pre-training Student model: Originally, we used the same architecture as the teacher model, however, it takes 11 hours to run 1 epoch. Later, we experimented with Bert for text and ResNet50 as the architecture for the student model, yet, this time it takes 25 hours to run 1 epoch. In the end, we decided our solution is to 1. Randomly sample 10% of the data from our dataset, since it is simply too big (118,287rows) and 2. Change to a simpler student model. Here is the final architecture of the student model. For the text processing component is the embedding layer that uses a vocabulary size matching BERT's base model; with the use of BERT's pretrained embeddings,it can significantly improve the model’s ability to understand textual nuances. Following the embeddings, the model employs a GRU with 256 hidden units to process the sequence of words. GRUs are also more efficient and have fewer parameters. On the visual side, the model processes images through a sequence of convolutional layers that increase in complexity. These layers extract a hierarchy of features from the images, starting from basic edges and textures to more complex patterns, by using 3x3 kernels and stride settings that gradually reduce the spatial dimensions. This setup ensures that as we go deeper into the network, it can identify and utilize more detailed visual information. An adaptive average pooling layer then condenses these features into a consistent size output, setting the stage for merging with the text data. The fusion of text and image features is straightforward: the model concatenates these features into a single vector. This fusion allows the model to integrate insights from both modalities, providing a richer representation of the input data. Following the fusion, the model uses fully connected layers, including a ReLU-activated layer, to further process the combined features and map them to the final output. The output layer, activated by a sigmoid function, where each label is determined independently.
- Fine Tuning: Given our source dataset COCO (A), which includes COCO and the labeled dataset from our teacher model prediction, and target dataset, Socratis (B), we will experiment with 2 fine tuning approaches based on Yosinski et al. In both cases, initialize the parameters of the teacher model (B) with the pre-trained weights of the student model (A).
AnB: finetuning but keeping all the layers frozen
AnB+: finetuning but not freezing the layers
However, as the architectures and dimensions for models A and B are very different, here is how we account for that: we initialize the last layer (fc) in model B with a modified version of the last two layers (fc1 and fc2) in model A. This involves adapting the dimensions of the weights to ensure compatibility between the models. In the AnB case, we would freeze all layers in model B except the last layer (fc), allowing it to adapt to the target task while keeping the rest of the model fixed. Conversely, in the AnB+ case, we would not freeze any layer in model B, enabling all layers, including the last layer (fc), to be fine-tuned on the target dataset. Both approaches are transferring from the COCO (student) model to the Socratis (teacher). The key distinction between AnB and AnB+ lies in whether all layers of the target model are fine-tuned after transferring knowledge from the source model (AnB+) or if only the higher layers are trained while the transferred layers remain frozen (AnB). 


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
