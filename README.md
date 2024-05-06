# Emotionally-Aware-Image-Caption-Generation
Authors: Tomisin Adeyemi, Christine Lei, and MinJoo Kim

## Overview
This project aims to study emotion recognition through transfer learning, employing the student-teacher model architecture. We utilize the Scoratis dataset, comprising 2075 unique images and 12489 total data points, as our target dataset and the basis for our Teacher Model. Each image is annotated with captions and emotions. Notably, repeated images exhibit varying emotions across subject annotations. We utilize the popular COCO Dataset, comprising of over 100k+ image-caption pairs, as our source dataset and the basis for the Student Model. The goal of this project is to generate emotions on the COCO dataset based on our trained teacher model. Ultimately, after augmenting the COCO dataset with emotions and building student and teacher models, we experiment with different techniques for finetuning.

## Datasets
- COCO dataset: raw data imported into code from the huggging face API. The dataset containing the noisy predictions, `coco_predictions.csv`, was too large to upload to github.
- Scoratis dataset: cleaned version contained in ['cleaned_data.csv'](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/cleaned_data.csv). Images are too big to upload.


## Preprocessing
- **Images**: Employ Fast R-CNN for visual feature extraction.
- **Text**: Utilize BERT tokenizer for generating necessary input structures for the model.

## Model Architecture
### Teacher Model
- 'Socratis_Teacher_Model.ipynb'
- **Core**: VisualBERT, designed for handling both textual and visual inputs.
- **Output**: Fully connected layer to map high-dimensional pooled outputs to emotion classes.

### Student Model
- 'studentModel.ipynb'
- **Text Processing**: Embedding layer utilizing BERT's pretrained embeddings and a GRU layer.
- **Visual Processing**: Series of convolutional layers with adaptive average pooling.
- **Fusion and Output**: Concatenation of text and image features, followed by fully connected layers and a sigmoid-activated output layer.

## Training
- **Optimizer**: Adam with a learning rate of 0.01.
- **Loss Function**: Binary cross-entropy with logits, ideal for multilabel classification.

## Predictions on COCO
- 'COCO Prediction + Sampling.ipynb'
Using the trained teacher model to infer emotion probabilities for the COCO dataset, setting thresholds to refine and focus on predominant emotions per image.

## Fine-tuning
The student model, pre-trained on COCO, is fine-tuned on the Scoratis dataset to enhance its specificity in emotion recognition by leveraging learned features from a large-scale generic dataset.


## Results and Discussion
Detailed analysis of model performance, including accuracy metrics and discussions on the effectiveness of the teacher-student model architecture in multimodal emotion recognition.
