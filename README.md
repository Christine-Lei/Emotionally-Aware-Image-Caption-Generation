# Emotionally-Aware-Image-Caption-Generation
Authors: Tomisin Adeyemi, Christine Lei, and MinJoo Kim

## Overview
This project explores emotion recognition from image-caption pairs using a multimodal teacher-student model architecture. We leverage the Scoratis dataset, which consists of 2075 images with 12489 interpretations per image, identifying predominant emotions such as happiness, sadness, and curiosity. By utilizing pretrained models for initial feature extraction and employing VisualBERT for integration and prediction, we aim to accurately classify emotional contexts from complex multimodal inputs.

## Dataset
- COCO dataset
- Scoratis dataset (cleaned version: 'cleaned_data.csv')
- 'coco_predictions.csv'
The Scoratis dataset includes a diverse range of emotions with detailed annotations. We preprocess the dataset by filtering emotions occurring over 100 times, reducing dimensionality to 29 significant emotions, and applying one-hot encoding for model compatibility. Additionally, the COCO dataset was used for further training using generated emotion predictions as proxy labels.

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
