import argparse
import gc
import logging
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from data import load_data
from eval_ppl import evaluate_ppl
from eval_prompt_rank import prompt_accuracy
from model import FullModel
from utils import set_all_seeds, sample_sequence
from optimizers import AdamW

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
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-ms', '--max_samples', type=int, default=-1)
parser.add_argument('-dir', '--dataset_dir', type=str, default='../writingPrompts/')
parser.add_argument('-gc', '--gpt2_config', type=str, default='gpt2')
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(123)

########## Parameters ##########
DATASET_DIR = args.dataset_dir
EPOCHS = args.epochs
PROMPT_LEN = args.prompt_len
MAX_CONTEXT = args.max_context
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None

model_save_path = 'saved_model.pt'

########## Hyperparameters ##########
GPT2_CONFIG = args.gpt2_config
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay

train_dataset, val_dataset, test_dataset = load_data(DATASET_DIR, True, MAX_SAMPLES)
_, val_dataset_raw, test_dataset_raw = load_data(DATASET_DIR, False, MAX_SAMPLES)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print('Creating Model...')
model = FullModel(gpt2_config=GPT2_CONFIG).to(device)

########## for debugging ##########
a, b = prompt_accuracy(model, val_dataset, PROMPT_LEN, MAX_CONTEXT, device, BATCH_SIZE)
print(a, b)

word_ppl, bpe_ppl, token_diffs = evaluate_ppl(model, val_dataset, val_dataset_raw, PROMPT_LEN, MAX_CONTEXT, device, BATCH_SIZE)
########## for debugging ##########

all_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

start_epoch = 0
best_val_ppl = float('inf')
if os.path.exists(model_save_path):
    print('Restoring trained model from %s' % model_save_path)
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    best_val_ppl = checkpoint['val_ppl']

print('Training Model for %d epochs...' % EPOCHS)
ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
model.train()
for ep in range(start_epoch, start_epoch + EPOCHS):

    # print("Epoch", ep, "start")

    # prompt = ["Reddit buys the moon"]
    # prompt = model.process_prompt(prompt, prompt_len=PROMPT_LEN).to(device)

    # encoded_prompt, _ = model.encoder(prompt, token_type_ids=None, attention_mask=prompt > 0)

    # out, logprobs = sample_sequence(model.decoder, 100, encoder_hidden=encoded_prompt, batch_size=BATCH_SIZE, context=[model.decoder_tokenizer.encoder['<|endoftext|>']], eos_token=model.decoder_tokenizer.encoder['<|endoftext|>'], 
    #     sample=True, top_p=0.95, device=device)

    # for i in out:
    #     print(model.decoder_tokenizer.decode(i))
        
    # print()

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

    val_word_ppl, bpe_ppl, token_diffs = evaluate_ppl(model, val_dataset, val_dataset_raw, PROMPT_LEN, MAX_CONTEXT, device, BATCH_SIZE)
    model.train()
    if val_word_ppl < best_val_ppl:
        best_val_ppl = val_word_ppl

        checkpoint = {'model': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'epoch': ep, 'val_ppl': val_word_ppl}
        torch.save(checkpoint, model_save_path)

    print('EPOCH: %3d/%d\t Loss: %.4f' % (ep, EPOCHS, epoch_loss/len(train_loader)))
