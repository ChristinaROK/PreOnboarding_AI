import os
import sys
import pandas as pd
import numpy as np 
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split

from transformers import BertTokenizer, BertModel

# GPU setups
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    return device

# Load Data 
def load_clean_data(data_path:str, type:str = "train", using_colab:bool = True):
    """
    - data_path : 자신의 구글 드라이브 내 nsmc 디렉토리 경로를 입력
    - type : "train" or "test"
    구글 드라이브의 train 또는 test 데이터를 로드한 후 결측치를 제거한 데이터프레임을 반환
    """
    if using_colab:
        from google.colab import drive
        drive.mount("/content/drive")
    
    _DATA_DIR = os.path.abspath(data_path)
    
    df = pd.read_csv(os.path.join(_DATA_DIR, f"ratings_{type}.txt"), delimiter="\t")
    print(f"dataframe shape: {df.shape}")
    
    # df에서 결측치 제거
    df=df[~df.document.isna()]
    print(f"na data removed dataframe shape: {df.shape}")
    
    return df
    
# Sampling Data
def label_evenly_balanced_dataset_sampler(df, sample_size):
    """
    - df : 0과 1의 label을 갖고 있는 데이터프레임
    - sample_size: df에서 추출할 데이터 개수
    """
    df = df.reset_index(drop=True) # Index로 iloc하기 위해서는 df의 index를 초기화해줘야 함
    neg_idx = df.loc[df.label==0].index
    neg_idx_sample = random.sample(neg_idx.to_list(), k=int(sample_size/2))

    pos_idx = df.loc[df.label==1].index
    pos_idx_sample = random.sample(pos_idx.to_list(), k=int(sample_size/2))

    return df.iloc[neg_idx_sample+pos_idx_sample]

# Custom Dataset
class CustomDataset(Dataset):
    """
    - input_data: list of string
    - target_data: list of int
    """
    
    def __init__(self, input_data:list, target_data:list) -> None:
        self.X = input_data
        self.Y = target_data
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# Custom collate_fn 
def custom_collate_fn(batch):
    """
    한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
    이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
    
    한 배치 내 레이블(target)은 텐서화 함.
    
    - batch: list of tuples (input_data(string), target_data(int))
    """
    input_list, target_list = [], []

    tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
    
    for _input, _target in batch:
        input_list.append(_input)
        target_list.append(_target)
    
    tensorized_input = tokenizer_bert(
        input_list,
        add_special_tokens=True,
        padding="longest", # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
        truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
        max_length=512,
        return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
    )
    
    tensorized_label = torch.tensor(target_list)
    
    return tensorized_input, tensorized_label
    
# Custom Classifer
class CustomClassifier(nn.Module):

    def __init__(self, hidden_size: int, n_label: int, freeze_base: bool = False):
        super(CustomClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("klue/bert-base")

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad=False

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(linear_layer_hidden_size, n_label)
        )

    

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        last_hidden_states = outputs[0] # last hidden states (batch_size, sequence_len, hidden_size)
        cls_token_last_hidden_states = last_hidden_states[:,0,:] # (batch_size, first_token, hidden_size)

        logits = self.classifier(cls_token_last_hidden_states)

        return logits

