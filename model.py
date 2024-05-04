import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")  # Pre-trained BERT model
        self.fc = nn.Linear(768,144)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(144,1)
        self.softmax=torch.nn.Softmax(dim=1)
        self.loss= nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask,logits_score,lm_score):
        outputs = self.bert(input_ids, attention_mask)["last_hidden_state"][:,0,:]
        reshaped_tensor =outputs.view(math.ceil(outputs.shape[0]/10), -1, 768)
        reshaped_tensor=self.fc(reshaped_tensor)
        reshaped_tensor=self.relu(reshaped_tensor)
        reshaped_tensor=self.fc1(reshaped_tensor).squeeze(2)
        logits = self.softmax(reshaped_tensor)
        # print("PREDICT: ",torch.argmax(logits,dim=1))

        # print(logits.shape)
        # print(labels.shape)
        return logits
class MyInferModel(nn.Module):
    def __init__(self):
        super(MyInferModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")  # Pre-trained BERT model
        self.fc = nn.Linear(768,144)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(144,1)
        self.softmax=torch.nn.Softmax(dim=1)
        self.loss= nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask,logits_score,lm_score):
        outputs = self.bert(input_ids, attention_mask)["last_hidden_state"][:,0,:]
        reshaped_tensor = outputs.view(math.ceil(outputs.shape[0]/10), 10, 768)
        reshaped_tensor=self.fc(reshaped_tensor)
        reshaped_tensor=self.relu(reshaped_tensor)
        reshaped_tensor=self.fc1(reshaped_tensor).squeeze(2)
        logits = self.softmax(reshaped_tensor)
        # print("PREDICT: ",torch.argmax(logits,dim=1))

        # print(logits.shape)
        # print(labels.shape)
        return logits
import math
class MyFusionModel(nn.Module):
    def __init__(self):
        super(MyFusionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")  # Pre-trained BERT model
        self.fc = nn.Linear(768,144)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(144,1)
        self.softmax=torch.nn.Softmax(dim=1)
        self.loss= nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask,logits_score,lm_score):
        outputs = self.bert(input_ids, attention_mask)["last_hidden_state"][:,0,:]
        reshaped_tensor =outputs.view(math.ceil(outputs.shape[0]/10), -1, 768)
        reshaped_tensor=self.fc(reshaped_tensor)
        reshaped_tensor=self.relu(reshaped_tensor)
        reshaped_tensor=self.fc1(reshaped_tensor).squeeze(2)
        logits = self.softmax(reshaped_tensor)
        lm_score=self.softmax(lm_score)
        final=logits*lm_score
        return final
class MyInferModel(nn.Module):
    def __init__(self):
        super(MyInferModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")  # Pre-trained BERT model
        self.fc = nn.Linear(768,144)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(144,1)
        self.softmax=torch.nn.Softmax(dim=1)
        self.loss= nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)["last_hidden_state"][:,0,:]
        reshaped_tensor = outputs.view(1, outputs.shape[0], 768)
        reshaped_tensor=self.fc(reshaped_tensor)
        reshaped_tensor=self.relu(reshaped_tensor)
        reshaped_tensor=self.fc1(reshaped_tensor).squeeze(2)
        logits = self.softmax(reshaped_tensor)
        # print("PREDICT: ",torch.argmax(logits,dim=1))

        # print(logits.shape)
        # print(labels.shape)
        return logits
