from datetime import datetime
import numpy as np
from random import randint
import  re
import torch
from torch.utils.data.dataloader import default_collate

from eval_ppl import compute_logprobs


def prompt_accuracy(model, d_val, prompt_len, max_context, device, batch_size):
    """
    Calculates the prompt ranking accuracy given a model.
    """
    print("Evaluating prompt rank...")
    model.eval()

    num_samples = 30 

    d_len = len(d_val)
    get_indices = list(np.random.randint(d_len, size=num_samples))
    hit = 0
    total = 0
    start = datetime.now()

    for i in get_indices:
        sample_ten = []
        sample_prompts = []
        text = d_val[i]
        prompt_story = []
        sample_ten.append(text)

        # get story
        # cur_prompt, cur_story = text.split('---\n')
        cur_prompt, cur_story = text
        prompt_story.append((cur_prompt, cur_story))

        sample_prompts.append('Prompt: ' + cur_prompt.strip() + '\n---\n')
        # sample 9 separate prompts
        while True:
            sample_nine = list(np.random.randint(d_len, size=10))
            if i not in sample_nine:
                break
        for j in sample_nine:
            # wrong_prompt = d_val[j].split('---\n')[0]
            wrong_prompt = d_val[j][0]
            
            wrong_text = 'Prompt: ' + wrong_prompt.strip() + '\n---\n' + cur_story.strip()
            prompt_story.append((wrong_prompt, cur_story))
            sample_ten.append(wrong_text)
            sample_prompts.append('Prompt: ' + wrong_prompt.strip() + '\n---\n')
        
        print('Running evaluation...')
        with torch.no_grad():
            logls = []
            batch = []

            prompt_lens = [len(model.decoder_tokenizer.encode(p)) for p in sample_prompts]
            # tokenize and create batch
            for text in sample_ten:
                bpe_tokens = model.decoder_tokenizer.encode(text)
                bpe_tokens = bpe_tokens[:1025] # TODO (This limit applies to GPT2)
                batch.append((bpe_tokens + [0] * (1025 - len(bpe_tokens)), len(bpe_tokens))) # Pad

            x, x_lens = zip(*batch)
            token_tensor = torch.tensor(x, dtype=torch.long, device=device)

            # Compute log probs
            prompt_batch, story_batch = default_collate(prompt_story)
            lps = compute_logprobs(model, prompt_batch, story_batch, prompt_len, max_context, device)
            token_tensor = token_tensor.cpu().numpy()

            # Compute individually
            for i in range(lps.shape[0]):
                log_probs = lps[i, prompt_lens[i]:x_lens[i] - 1]
                logls.append(-log_probs.float().mean().item())
            
            if logls[0] == min(logls):
                hit += 1
            total += 1
            batch = []
    
            return(datetime.now() - start, hit / total)
