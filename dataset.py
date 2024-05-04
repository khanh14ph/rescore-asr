import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import re
from jiwer import wer
import pandas as pd
from torchaudio.utils import download_asset
import torchaudio.functional as F
import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(self.path)
        
        self.df=self.df.fillna("")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        prev_text=row["prev_text"]
        min_wer=1000
        ref=row["text"]
        n_hypthesis=row["top_N"].split("|")[:10]
        logits_score=row["logits_score"].split("|")[:10]
        logits_score=[float(v) for v in logits_score]
        lm_score=row["lm_score"].split("|")[:10]
        lm_score=[float(j) for j in lm_score]

        for index,i in enumerate(n_hypthesis):
            # print("row",row[1])
            # print("i ",i)
            temp=wer(ref,i)
            
            if temp<min_wer:
                label=index
                min_wer=temp
        return prev_text,n_hypthesis, logits_score,lm_score,label,ref
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    text_lst=[]
    prev_text,n_hypothesis,logits_score,lm_score, labels,ref = zip(*batch)
    for i,m in zip(n_hypothesis,prev_text):
        prev_lst=m.split("|")[-2:]
        m=".".join(prev_lst)
        # m_lst=m.split()[-200:]
        # print("len m_lst",len(m_lst))

        # m_text=" ".join(m_lst)
        for j in i:
            text_lst.append(m+" </s> "+j)
    encodings = tokenizer(text_lst, padding=True, truncation=True, return_tensors="pt")
    # print("encodings",encodings["input_ids"].shape)
    input_ids=encodings["input_ids"]
    attention_mask=encodings["attention_mask"]
    # input_ids = torch.cat((encodings["input_ids"][:,0:1],encodings["input_ids"][:,-257:]),dim=1)
    # attention_mask = torch.cat((encodings["attention_mask"][:,0:1],encodings["attention_mask"][:,-257:]),dim=1)
    
    # Convert labels to tensors (assuming you have integer labels)
    labels = torch.tensor(labels)
    logits_score=torch.tensor(logits_score)
    lm_score=torch.tensor(lm_score)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        "logits_score":logits_score,
        "lm_score": lm_score,
        'labels': labels,
        "n_hypothesis":n_hypothesis,
        "ref":ref

    }

class MyDataset1(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(self.path)
        
        self.df=self.df.fillna("")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        prev_text=row["prev_text"]
        min_wer=1000
        ref=row["text"]
        n_hypthesis=row["top_N"].split("|")[:10]
        logits_score=row["logits_score"].split("|")[:10]
        logits_score=[float(v) for v in logits_score]
        lm_score=row["lm_score"].split("|")[:10]
        lm_score=[float(j) for j in lm_score]
        wer_lst=[]
        for index,i in enumerate(n_hypthesis):
            # print("row",row[1])
            # print("i ",i)
            temp=wer(ref,i)
            
            wer_lst.append(temp)
        min_wer=min(wer_lst)
        max_wer=max(wer_lst)
        wer_range=max_wer-min_wer
        wer_lst=[]
        return prev_text,n_hypthesis, logits_score,lm_score,label
def collate_fn_padd1(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    text_lst=[]
    prev_text,n_hypothesis,logits_score,lm_score, labels = zip(*batch)
    for i,m in zip(n_hypothesis,prev_text):
        prev_lst=m.split("|")[-2:]
        m=".".join(prev_lst)
        # m_lst=m.split()[-200:]
        # print("len m_lst",len(m_lst))

        # m_text=" ".join(m_lst)
        for j in i:
            text_lst.append(m+" </s> "+j)
    encodings = tokenizer(text_lst, padding=True, truncation=True, return_tensors="pt")
    # print("encodings",encodings["input_ids"].shape)
    input_ids=encodings["input_ids"]
    attention_mask=encodings["attention_mask"]
    # input_ids = torch.cat((encodings["input_ids"][:,0:1],encodings["input_ids"][:,-257:]),dim=1)
    # attention_mask = torch.cat((encodings["attention_mask"][:,0:1],encodings["attention_mask"][:,-257:]),dim=1)
    
    # Convert labels to tensors (assuming you have integer labels)
    labels = torch.tensor(labels)
    logits_score=torch.tensor(logits_score)
    lm_score=torch.tensor(lm_score)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        "logits_score":logits_score,
        "lm_score": lm_score,
        'labels': labels
    }