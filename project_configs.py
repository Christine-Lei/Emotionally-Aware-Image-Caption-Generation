import os
import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models

from transformers import BertTokenizer, VisualBertModel,  BertConfig, BertModel

# from visualbert
from processing_image import Preprocess
from utils import Config
from modeling_frcnn import GeneralizedRCNN

NUM_EMOTIONS = 29

class ImageProcessor:
    def __init__(self, device='cuda'):
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn_cfg.MODEL.DEVICE = device
        self.device = device

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

        self.frcnn_cfg = frcnn_cfg
        self.image_preprocess = Preprocess(frcnn_cfg)

    def get_visual_embeddings(self, image_path):
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_path)

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features").detach().cpu()
        return features

class TeacherModel(nn.Module):
    def __init__(self, visualbert_model):
        super(TeacherModel, self).__init__()
        self.visualbert = visualbert_model
        self.fc = nn.Linear(self.visualbert.config.hidden_size, NUM_EMOTIONS)

    def forward(self, input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask, labels):
        visualbert_outputs = self.visualbert(input_ids=input_ids.squeeze(1),
                                             attention_mask=attention_mask.squeeze(1),
                                             token_type_ids=token_type_ids.squeeze(1),
                                             visual_embeds=visual_embeds.squeeze(1),
                                            visual_token_type_ids=visual_token_type_ids.squeeze(1),
                                            visual_attention_mask=visual_attention_mask.squeeze(1))
        pooled_output = visualbert_outputs['pooler_output']

        # Emotion prediction
        logits = self.fc(pooled_output) # Loss function operates from logits
        
        return logits

class StudentModel(nn.Module):
    def __init__(self, visualbert_model, device='cuda'):
        super(StudentModel, self).__init__()
        self.visualbert = visualbert_model
        self.fc = nn.Linear(self.visualbert.config.hidden_size, NUM_EMOTIONS)

    def forward(self, input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask, labels):
        visualbert_outputs = self.visualbert(input_ids=input_ids.squeeze(1),
                                             attention_mask=attention_mask.squeeze(1),
                                             token_type_ids=token_type_ids.squeeze(1),
                                             visual_embeds=visual_embeds.squeeze(1),
                                            visual_token_type_ids=visual_token_type_ids.squeeze(1),
                                            visual_attention_mask=visual_attention_mask.squeeze(1))
        pooled_output = visualbert_outputs['pooler_output']
        

        # Emotion prediction
        logits = self.fc(pooled_output) # Loss function operates from logits

        return logits

# Preprocessed COCO Dataset in coco_predictions.csv
class CocoCaptionsDataset(Dataset):
    def __init__(self, dataframe, device='cuda'):
        self.dataframe = dataframe

        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.visual_extractor = ImageProcessor(device=device)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_url = row['coco_url']
        captions = eval(row['captions'])  # Evaluating the string to get the list and taking the first item

        # Image processing
        visual_embeds = self.visual_extractor.get_visual_embeddings(image_url)
        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float)

        # we need handle the fact that there are multiple captions (as opposed to one caption in Socratis)
        # We handle this by taking an average of the numeric values associated with the words
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for caption in captions:
            inputs = self.tokenizer(caption, return_tensors="pt", max_length=32, truncation=True, padding='max_length')
            input_ids_list.append(inputs["input_ids"])
            token_type_ids_list.append(inputs["token_type_ids"])
            attention_mask_list.append(inputs["attention_mask"])

        input_ids = torch.cat(input_ids_list, dim=0).float().mean(dim=0).long()
        token_type_ids = torch.cat(token_type_ids_list, dim=0).float().mean(dim=0).long()
        attention_mask = torch.cat(attention_mask_list, dim=0).float().mean(dim=0).long()

        # Labels - extracting the last 29 columns as classes
        labels = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)

        return (input_ids, token_type_ids, attention_mask, 
                visual_embeds, visual_token_type_ids, visual_attention_mask,
                labels)

label_map = {'curious': 0, 'amazed': 1, 'fear': 2, 'awe': 3, 'neutral': 4, 'disgusted': 5, 'worried': 6, 'intrigued': 7,
 'confused': 8, 'beautiful': 9, 'happy': 10, 'annoyed': 11, 'impressed': 12, 'sad': 13, 'proud': 14, 'inspired': 15, 'angry': 16,
 'excited': 17, 'nostalgic': 18, 'upset': 19, 'concerned': 20, 'good': 21, 'hopeful': 22, 'anger': 23, 'joy': 24, 'interested': 25,
 'calm': 26, 'bored': 27, 'scared': 28}

# Preprocessed Socratis Dataset in cleaned_data.csv
class SocratisDataset(Dataset):
    def __init__(self, data, images_base_path, device='cuda'):
        self.df = data
        self.images_base_path = images_base_path

        # feature extractors
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.visual_extractor = ImageProcessor(device=device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        captions, image_name = self.df.iloc[idx]['caption'], self.df.iloc[idx]['image_name']

        # get image embedings
        image_path = os.path.join(self.images_base_path, image_name)
        visual_embeds = self.visual_extractor.get_visual_embeddings(image_path)
        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float)

        # get text embeddings
        inputs = self.tokenizer(captions, return_tensors="pt", max_length=32, truncation=True, padding='max_length')
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        emotions_string = self.df['emotions'][idx]
        emotions_list = eval(emotions_string)
        one_hot_encoded = torch.zeros(len(label_map))
        for emotion in emotions_list:
            if emotion in label_map:
                idx = label_map[emotion]
                one_hot_encoded[idx] = 1

        labels = one_hot_encoded

        return (input_ids, token_type_ids, attention_mask,
                visual_embeds, visual_token_type_ids, visual_attention_mask,
                labels)