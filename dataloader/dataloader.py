import os
import sys
import pandas as pd
import numpy as np 
import random

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split

from transformers import BertTokenizer

N_TRAIN = 10000
N_TEST = 1000
BATCH_SIZE = {
    "train" : 32,
    "valid" : 64,
    "test"  : 64,
}

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



class NSMCDataLoader:

    def __init__(self) -> None:
        self.train_dataset, self.valid_dataset, self.test_dataset = self.make_dataset()

    
    def _label_evenly_balanced_dataset_sampler(self, df, sample_size):
        """
        - df : 0과 1의 label을 갖고 있는 데이터프레임
        - sample_size: df에서 추출할 데이터 개수
        데이터프레임에서 레이블 비율을 동일하게 유지하면서 sample size 만큼 추출
        """
        df = df.reset_index(drop=True) # Index로 iloc하기 위해서는 df의 index를 초기화해줘야 함
        neg_idx = df.loc[df.label==0].index
        neg_idx_sample = random.sample(neg_idx.to_list(), k=int(sample_size/2))

        pos_idx = df.loc[df.label==1].index
        pos_idx_sample = random.sample(pos_idx.to_list(), k=int(sample_size/2))

        return df.iloc[neg_idx_sample+pos_idx_sample]


    def make_dataset(self):
        """
       데이터를 로드해 결측치를 제거한 후  train, valid, test Dataset 인스턴스 반환
        """

        _DATA_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "data", "nsmc"))
        print(_DATA_DIR)        
        train_df = pd.read_csv(os.path.join(_DATA_DIR, f"ratings_train.txt"), delimiter="\t")
        test_df =  pd.read_csv(os.path.join(_DATA_DIR, f"ratings_test.txt"), delimiter="\t")
        
        # df에서 결측치 제거
        train_df=train_df[~train_df.document.isna()]
        test_df=test_df[~test_df.document.isna()]

        # df row개수를 줄이기
        train_df = self._label_evenly_balanced_dataset_sampler(train_df, N_TRAIN)
        test_df = self._label_evenly_balanced_dataset_sampler(test_df, N_TEST)

        # Dataset 인스턴스 생성
        train_dataset = CustomDataset(train_df.document.to_list(), train_df.label.to_list())
        test_dataset = CustomDataset(test_df.document.to_list(), test_df.label.to_list())
        
        # train dataset을 9:1 비율로 학습 검증 셋으로 분리 
        n_train = int(N_TRAIN*0.9)
        n_valid = N_TRAIN - n_train 
        train_dataset, valid_dataset = random_split(train_dataset, [n_train, n_valid])

        print(f"Train Dataset: {len(train_dataset):,}\nDev Dataset: {len(valid_dataset):,}\nTest Dataset: {len(test_dataset):,}")

        return train_dataset, valid_dataset, test_dataset
    

    def _custom_collate_fn(self, batch):
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


    def make_dataloader(self, type:str, bsz:int=None):
        """
        - type : train|valid|test
        - bsz : 데이터로더 배치 사이즈

        type (train|valid|test)에 맞는 DataLoader 반환
        """
        dataset = getattr(self, f"{type}_dataset")
        sampler_fn = RandomSampler if type == "train" else SequentialSampler

        return DataLoader(
            dataset,
            batch_size = BATCH_SIZE.get(type) if bsz is None else bsz,
            sampler = sampler_fn(dataset),
            collate_fn = self._custom_collate_fn
        )
