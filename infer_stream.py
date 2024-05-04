import py_vncorenlp

# py_vncorenlp.download_model(save_dir='/home2/khanhnd')
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
# py_vncorenlp.download_model(save_dir='/home4/khanhnd/')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home4/khanhnd/')
def convert(x):
    return "".join(rdrsegmenter.word_segment(x))
import pandas as pd
df_train=pd.read_csv("/home4/khanhnd/VLSP/train/train_simple.csv")
df_dev=pd.read_csv("/home4/khanhnd/VLSP/dev/dev_simple.csv")
df_test=pd.read_csv("/home4/khanhnd/VLSP/test/test_simple.csv")
# def rm(x):
#   x=x.replace("/home2/khanhnd/conformer-rnnt/", "")
#   return x
def get_lecture(x):
  name=x.split("/")[-1].rstrip(".wav").split("-")[0]
  return int(name)
def get_order(x):
    time=x.split("/")[-1].rstrip(".wav").split("-")[1]
    return int(time)
df_dev["lecture"]=df_dev["audio"].apply(get_lecture)
df_dev["text"]=df_dev["text"].apply(lambda x: x.replace("_"," "))
df_test["lecture"]=df_test["audio"].apply(get_lecture)
df_dev["time"]=df_dev["audio"].apply(get_order)
df_test["time"]=df_test["audio"].apply(get_order)

import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

from pyctcdecode.alphabet import Alphabet
from pyctcdecode.language_model import LanguageModel, MultiLanguageModel
from pyctcdecode.decoder import BeamSearchDecoderCTC
import kenlm
vocab_dict={"ẻ": 0, "6": 1, "ụ": 2, "í": 3, "3": 4, "ỹ": 5, "ý": 6, "ẩ": 7, "ở": 8, "ề": 9, "õ": 10, "7": 11, "ê": 12, "ứ": 13, "ỏ": 14, "v": 15, "ỷ": 16, "a": 17, "l": 18, "ự": 19, "q": 20, "ờ": 21, "j": 22, "ố": 23, "à": 24, "ỗ": 25, "n": 26, "é": 27, "ủ": 28, "у": 29, "ô": 30, "u": 31, "y": 32, "ằ": 33, "4": 34, "w": 35, "b": 36, "ệ": 37, "ễ": 38, "s": 39, "ì": 40, "ầ": 41, "ỵ": 42, "8": 43, "d": 44, "ể": 45, "r": 47, "ũ": 48, "c": 49, "ạ": 50, "9": 51, "ế": 52, "ù": 53, "ỡ": 54, "2": 55, "t": 56, "i": 57, "g": 58, "́": 59, "ử": 60, "̀": 61, "á": 62, "0": 63, "ậ": 64, "e": 65, "ộ": 66, "m": 67, "ẳ": 68, "ợ": 69, "ĩ": 70, "h": 71, "â": 72, "ú": 73, "ọ": 74, "ồ": 75, "ặ": 76, "f": 77, "ữ": 78, "ắ": 79, "ỳ": 80, "x": 81, "ó": 82, "ã": 83, "ổ": 84, "ị": 85, "̣": 86, "z": 87, "ả": 88, "đ": 89, "è": 90, "ừ": 91, "ò": 92, "ẵ": 93, "1": 94, "ơ": 95, "k": 96, "ẫ": 97, "p": 98, "ấ": 99, "ẽ": 100, "ỉ": 101, "ớ": 102, "ẹ": 103, "ă": 104, "o": 105, "ư": 106, "5": 107, "|": 46, "<unk>": 108, "<pad>": 109}
labels= list({k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}.keys())
alphabet = Alphabet.build_alphabet(labels)
alpha = 0.5
beta = 1
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.language_model import LanguageModel, MultiLanguageModel
from pyctcdecode.decoder import BeamSearchDecoderCTC
# kenlm_model_base = LanguageModel(kenlm.Model('/home2/khanhnd/wav2vec2/vi_lm_4grams.bin'),alpha=alpha, beta=beta)
kenlm_model_lecture=LanguageModel(kenlm.Model('/home4/khanhnd/pbert/4gram_correct.arpa'),alpha=alpha, beta=beta)
# multi_lm = MultiLanguageModel([kenlm_model_base, kenlm_model_lecture])
decoder=BeamSearchDecoderCTC(alphabet, kenlm_model_lecture)
import torch
from jiwer import wer
device="cuda"
import librosa
import time
from model import MyInferModel,MyFusionModel,MyModel
model=MyModel().to(device)
from transformers import AutoModel, AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model.load_state_dict(torch.load("/home4/khanhnd/pbert/best_checkpoint_10_hidden.pt",map_location=torch.device(device)))
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
wav2vec_tokenizer = Wav2Vec2CTCTokenizer("/home4/khanhnd/pbert/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=wav2vec_tokenizer)
wav2vec= Wav2Vec2ForCTC.from_pretrained("/home4/khanhnd/checkpoint").to(device)
for df in [df_dev,df_test]:
  lect_dict=dict()
  lect_dict_gold=dict()
  lec_lst=set(df["lecture"])
  for i in lec_lst:
    lect_dict[i]=[]
    lect_dict_gold[i]=[]
  bert_err=[]
  norma_err=[]
  siu_err=[]
  bert_ref_err=[]
  oracle_err=[]
  oracle_err_ref=[]
  softmax=torch.nn.Softmax(dim=1)
  # df_dev=df_dev[0:100]
  wav2vec_lst=[]
  rescore_lst=[]
  for index, i in tqdm(df.iterrows(),  total=df.shape[0], desc=f'Reading DF'):
    ref=i["text"]
    link=i["audio"]
    lecture_name=i["lecture"]
    time_wav2vec_start=time.time()
    audio_input, sr = librosa.load(link, sr=16000)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    logits_finetune = wav2vec(input_values).logits
    logits1 = logits_finetune.cpu().detach().numpy()[0]
    
    prediction_LM_base = decoder.decode_beams(logits1)
    
    prediction_LM_base=prediction_LM_base[0:10]
    prediction_LM=[r[0] for r in prediction_LM_base]
    logits_score=torch.tensor([r[-2] for r in prediction_LM_base],dtype=torch.float32).to(device).unsqueeze(0)
    lm_score=torch.tensor([r[-1] for r in prediction_LM_base],dtype=torch.float32).to(device).unsqueeze(0)

    #using stream
    prev_text=".".join(lect_dict[lecture_name])
    norma_prediction=prediction_LM[0]
    norma_wer=wer(ref, norma_prediction)
    sentences=[convert(prev_text)+" </s> "+ convert(j) for j in prediction_LM]
    input_ids=tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    time_wav2vec_end=time.time()
    wav2vec_lst.append(time_wav2vec_end-time_wav2vec_start)
    res=model.forward(input_ids["input_ids"],input_ids["attention_mask"],logits_score,lm_score)[0]
    # res=int(torch.argmax(res))
    # final_res=res*softmax(logits_score)[0]*softmax(lm_score)[0]
    final_res=res
    final_res=int(torch.argmax(final_res))
    final=prediction_LM[final_res]
    time_recore_end=time.time()
    rescore_lst.append(time_recore_end-time_wav2vec_end)
    min_wer=1000
    for p in prediction_LM:
      tem=wer(ref,p)
      if tem < min_wer:
          min_wer=tem
      
    oracle_err.append(min_wer)


    #using ref
    prev_text=".".join(lect_dict_gold[lecture_name])

    norma_prediction=prediction_LM[0]
    norma_wer=wer(ref, norma_prediction)
    sentences=[convert(prev_text)+" </s> "+ convert(j) for j in prediction_LM]
    input_ids=tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

    res=model.forward(input_ids["input_ids"],input_ids["attention_mask"],logits_score,lm_score)[0]
    # final_res=res*softmax(logits_score)[0]*softmax(lm_score)[0]
    final_res=res
    final_res=int(torch.argmax(final_res))
    final_gold=prediction_LM[final_res]

    min_wer=1000
    for p in prediction_LM:
      tem=wer(ref,p)
      if tem < min_wer:
          min_wer=tem
      
    oracle_err_ref.append(min_wer)

    bert_wer=wer(ref,final)
    bert_ref_wer=wer(ref,final_gold)
    # if bert_ref_wer<norma_wer and (final_gold==ref):
    #    print("PREV: ",prev_text)
    #    print("norma_prediction: ",norma_prediction  )
    #    print("final_gold: ",final_gold  )
       
    #    print("ref: ", ref)




    temp=lect_dict[lecture_name]
    temp.append(final)
    while len(temp)>3:
      temp.pop(0)
    temp1=lect_dict_gold[lecture_name]
    temp1.append(ref)
    while len(temp1)>3:
      temp1.pop(0)
    lect_dict[lecture_name]=temp
    lect_dict_gold[lecture_name]=temp1
    norma_err.append(norma_wer)
    bert_err.append(bert_wer)
    bert_ref_err.append(bert_ref_wer)
    

  print()
  print("-----")
  print("BERT WER: ", sum(bert_err)/len(bert_err))
  print("BERT REF WER: ", sum(bert_ref_err)/len(bert_ref_err))
  print("NORMA WER: ", sum(norma_err)/len(norma_err))
  print("ORACLE WER: ", sum(oracle_err)/len(oracle_err))
  print("ORACLE WER ref: ", sum(oracle_err_ref)/len(oracle_err_ref))
  print(sum(rescore_lst)/len(rescore_lst))
  print(sum(wav2vec_lst)/len(wav2vec_lst))


