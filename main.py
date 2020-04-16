import argparse
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
from utils import set_all_seeds

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-bs', '--batch_size', type=int, default=8)
parser.add_argument('-pl', '--prompt_len', type=int, default=128)
parser.add_argument('-mc', '--max_context', type=int, default=64)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-ms', '--max_samples', type=int, default=-1)
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(123)

########## Parameters ##########
DATASET_DIR = '../writingPrompts/'
EPOCHS = args.epochs
PROMPT_LEN = args.prompt_len
MAX_CONTEXT = args.max_context
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None 

########## Hyperparameters ##########
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay

train_loader, val_loader, test_loader = load_data(DATASET_DIR, BATCH_SIZE, NUM_WORKERS, MAX_SAMPLES)

print('Creating Model...')
model = FullModel().to(device)

if os.path.exists('model.pt'):
    print('Restoring trained Model!')
    model.load_state_dict(torch.load('model.pt'), strict=False)

all_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

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

    torch.save(model.state_dict(), 'model.pt')

    print('EPOCH: %3d/%d\t Loss: %.4f' % (ep, EPOCHS, epoch_loss/len(train_loader)))
