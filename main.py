import gc
import logging
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from data import load_data
# from eval import evaluate_dataset
from model import FullModel
# from utils import num_params, sample_sequence, random_truncate, collate_fn_masked, get_pos_embeddings

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

## Parameters
DATASET_DIR = "../writingPrompts/"
EPOCHS = 100
PROMPT_LEN = 128
MAX_CONTEXT = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 8

train_loader, val_loader, test_loader = load_data(DATASET_DIR, BATCH_SIZE, max_samples=1000)

print('Creating Model...')
model = FullModel().to(device)

all_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

if os.path.exists("model.pt"):
    print('Restoring trained model!')
    model.load_state_dict(torch.load("model.pt"), strict=False)

# BATCH_SIZE_SAMPLE = BATCH_SIZE

print('Training Model for %d epochs...' % EPOCHS)
ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
model.train()
for ep in range(1, EPOCHS+1):
    epoch_loss = 0
    for prompt, story in tqdm(train_loader):
        optimizer.zero_grad()

        prompt = model.process_prompt(prompt, prompt_len=PROMPT_LEN).to(device)

        story, story_target, story_mask = model.process_story(story, max_context=MAX_CONTEXT)
        story = story.to(device)
        story_target = story_target.to(device)
        story_mask = story_mask.to(device)

        logits = model(story, prompt)
        num_logits = logits.size(-1)

        ce_loss = ce_loss_fn(logits.view(-1, num_logits), story_target.view(-1)).mean()
        epoch_loss += ce_loss.item()
        ce_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        optimizer.step()

        del story, story_target, story_mask, prompt
        gc.collect()

    scheduler.step()

    torch.save(model.state_dict(), "model.pt")

    print('EPOCH: %3d\t Loss: %.4f' % (ep, epoch_loss/len(train_loader)))
