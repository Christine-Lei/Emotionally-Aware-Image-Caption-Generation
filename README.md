# Emotionally-Aware-Image-Caption-Generation
Authors: Tomisin Adeyemi, Christine Lei, and MinJoo Kim

## Overview
This project aims to study emotion recognition through transfer learning, employing the student-teacher model architecture. We utilize the Scoratis dataset, comprising 2075 unique images and 12489 total data points, as our target dataset and the basis for our Teacher Model. Each image is annotated with captions and emotions. Notably, repeated images exhibit varying emotions across subject annotations. We utilize the popular COCO Dataset, comprising of over 100k+ image-caption pairs, as our source dataset and the basis for the Student Model. The goal of this project is to generate emotions on the COCO dataset based on our trained teacher model. Ultimately, after augmenting the COCO dataset with emotions and building student and teacher models, we experiment with different techniques for finetuning.

## Datasets
- COCO dataset: raw data imported into code from the huggging face API. The dataset containing the noisy predictions, `coco_predictions.csv`, was too large to upload to github.
- Scoratis dataset: cleaned version contained in [`cleaned_data.csv`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/cleaned_data.csv). Images are too big to upload.

## Methodology
![Model Sketch](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/assets/98556351/3cf2db35-6b0a-49b4-b089-e596a0a8239e) 

Inspired by Yalnize et Al, we employ a student-teacher architecture for the transfer learning process. We first train a teacher model on the target dataset, make predictions on the source using the teacher model, resulting in a new dataset. We then pretrain a student model on the new dataset, then finetune it for better predictions. The main reason for adopting this approach is that our target dataset is a lot smaller than the source, so transferability is perfect for our task. The image below summarizes our training approach.

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

## Fine-tuning
- [`FineTuning.ipynb`](https://github.com/Christine-Lei/Emotionally-Aware-Image-Caption-Generation/blob/main/FineTuning.ipynb)
The layers of the trained student model (A) are transferred to the layers of the teacher model (B). And we experiment with two finetuning techniques: AnB & AnB+, where AnB involves freezing all the layers and AnB+ involves not freezing the layers.

## Results and Discussion
Although the AnB architecture has a better convergence than the AnB+ architecture, the AnB+ architecture performs better across all metrics we tested. The plots of the training loss as well as the metrics are shown below:


## Reference 
Inspired by [Socratis: Are large multimodal models emotionally aware?](https://arxiv.org/abs/2308.16741), [How transferable are features in deep neural
networks?](https://proceedings.neurips.cc/paper_files/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf), and [Billion-scale semi-supervised learning for image classification](https://arxiv.org/abs/1905.00546)
