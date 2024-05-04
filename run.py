from transformers import AdamW
import torch
from dataset import MyDataset,collate_fn_padd
from model import MyFusionModel,MyModel
from torch.utils.data import DataLoader
from jiwer import wer
device="cuda"
data=MyDataset("/home4/khanhnd/pbert/train.csv")
test_data=MyDataset("/home4/khanhnd/pbert/dev.csv")
dataloader = DataLoader(data, batch_size=3, collate_fn=collate_fn_padd,shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=3, collate_fn=collate_fn_padd,shuffle=True, drop_last=True)
model=MyModel().to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params num: ", count_parameters(model))
learning_rate=0.00001
from torch.optim.lr_scheduler import ExponentialLR
# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)
criterion = torch.nn.CrossEntropyLoss()
num_epochs=20
from tqdm import tqdm
import torch.nn as nn
# model.load_state_dict(torch.load("/home4/khanhnd/pbert/best_checkpoint_10_hidden.pt"))
# Training loop








all_loss=[]



import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home4/khanhnd/pbert/logs_base")

from sklearn.metrics import f1_score
min_wer=1000
for epoch in range(num_epochs):
    model.train()
    losses=[]
    progress=tqdm(dataloader)
    loss=0
    max_l=len(dataloader)
    train_predict_lst=[]
    train_label_lst=[]
    for ind,batch in enumerate(progress):
        
        inputs = batch['input_ids'].to(device)
        attention_mask= batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits_score=batch['logits_score'].to(device)
        lm_score=batch['lm_score'].to(device)
        n_hypothesis=batch["n_hypothesis"]
        optimizer.zero_grad()
        logits = model(inputs,attention_mask,logits_score,lm_score)

        
        loss=criterion(logits, labels)
        res=torch.argmax(logits,dim=1).cpu().tolist()
        labels=labels.cpu().tolist()
        train_predict_lst+=res
        train_label_lst+=labels
        loss.backward()
        optimizer.step()
        progress.set_postfix_str(f"Epoch {epoch}: "+str(float(loss.detach().cpu())))
        writer.add_scalar('Loss/train', float(loss.detach().cpu()), epoch*max_l+ind)
    scheduler.step()
    torch.save(model.state_dict(), "/home4/khanhnd/pbert/checkpoint_10_didden.pt")
    final=f1_score(train_label_lst,train_predict_lst,average='micro')
    check=[0 for i in train_predict_lst]
    final1=f1_score(train_label_lst,check,average='micro')
    print("Training F1: ",final)
    print("Normal     : ",final1)
    predict_lst=[]
    label_lst=[]
    wer_lst=[]
    wer_norma_lst=[]
    with torch.no_grad():
        progress_eval=tqdm(test_dataloader)
        for batch in progress_eval:
            
            inputs = batch['input_ids'].to(device)
            attention_mask= batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits_score=batch['logits_score'].to(device)
            lm_score=batch['lm_score'].to(device)
            n_hypothesis=batch["n_hypothesis"]
            ref=batch["ref"]
            logits = model(inputs,attention_mask,logits_score,lm_score)
            res=torch.argmax(logits,dim=1).cpu().tolist()
            labels=labels.cpu().tolist()
            wer_new=[wer(j.replace("_"," "),n_hypothesis[idx][res[idx]].replace("_"," ")) for idx,j in enumerate(ref)]
            wer_new_norma=[wer(j.replace("_"," "),n_hypothesis[idx][0].replace("_"," ")) for idx,j in enumerate(ref)]
            wer_lst+=wer_new
            wer_norma_lst+=wer_new_norma
    val_wer=sum(wer_lst)/len(wer_lst)
    val_wer_norma=sum(wer_norma_lst)/len(wer_norma_lst)
    writer.add_scalar('WER', float(val_wer), epoch)

    if val_wer<min_wer:
        print("FOUND NEW BEST")
        torch.save(model.state_dict(), "/home4/khanhnd/pbert/best_checkpoint_10_hidden.pt")
        min_wer=val_wer
    print("Valid wer: ",val_wer)
    print("Valid wer norma: ",val_wer_norma)





    


