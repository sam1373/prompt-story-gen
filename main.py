from data import Dataset

import torch
import torch.nn as nn


from transformers import GPT2Tokenizer

from model import FullModel

import matplotlib.pyplot as plt

import torchvision

from utils import num_params, sample_sequence, random_truncate, collate_fn_masked, get_pos_embeddings

from transformers.modeling_utils import Conv1D

from os import path as p

import numpy as np

from torch.optim.lr_scheduler import StepLR

from optimizers import *

from eval import evaluate_dataset

import gc

BATCH_SIZE = 8
PROMPT_LEN = 128
MAX_CONTEXT = 64

EPOCHS = 100

DATASET_DIR = "/home/samuel/Data/writingPrompts/"

train_dataset = Dataset(DATASET_DIR + "train.wp_source", DATASET_DIR + "train.wp_target")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)

val_dataset = Dataset(DATASET_DIR + "valid.wp_source", DATASET_DIR + "valid.wp_target")

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)

test_dataset = Dataset(DATASET_DIR + "test.wp_source", DATASET_DIR + "test.wp_target")

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)

iterable_val_loader = iter(val_loader)


model = FullModel()

all_params = [p for p in model.parameters() if p.requires_grad]

for par in all_params:
    print(par.shape)

optimizer = torch.optim.AdamW(all_params, lr=0.001)

scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

if p.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt"), strict=False)

BATCH_SIZE_SAMPLE = BATCH_SIZE

ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

for ep in range(EPOCHS):
    for i, (prompt, story) in enumerate(train_loader):

        optimizer.zero_grad()

        model.train()

        prompt = model.process_prompt(prompt, prompt_len=PROMPT_LEN).cuda()

        story, story_target, story_mask = model.process_story(story, max_context=MAX_CONTEXT)

        story = story.cuda()
        story_target = story_target.cuda()
        story_mask = story_mask.cuda()

        print(story.shape, prompt.shape)

        logits = model(story, prompt)

        num_logits = logits.size(-1)

        ce_loss = ce_loss_fn(logits.view(-1, num_logits), story_target.view(-1)).float().mean()

        print(ce_loss)

        ce_loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.01)

        optimizer.step()

        del story, story_target, story_mask, prompt
        gc.collect()

        if i % 100 == 0:
            print(i, "/", len(train_loader))


    scheduler.step()

    torch.save(model.state_dict(), "model.pt")

    print("EPOCH", ep + 1, "DONE")

    print()
    print("-------------------------------------------")
    print()

