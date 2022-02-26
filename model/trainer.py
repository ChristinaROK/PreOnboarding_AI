import os
import sys

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import CustomClassifier, CrossEntropyLoss


def initializer(train_dataloader, epochs=2, lr=2e-5):
    """
    모델, 옵티마이저, 스케쥴러 초기화
    """
    
    model = CustomClassifier(hidden_size=768, n_label=2)

    optimizer = AdamW(
        model.parameters(), # update 대상 파라미터를 입력
        lr=lr,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * epochs
    print(f"Total train steps with {epochs} epochs: {total_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, # 여기서는 warmup을 사용하지 않는다.
        num_training_steps = total_steps
    )

    return model, optimizer, scheduler