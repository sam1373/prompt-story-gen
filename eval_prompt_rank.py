"""
Calculates the prompt ranking accuracy given a model.
"""
import pickle
import  re
import torch
import numpy as np
from random import randint
from datetime import datetime
from torch.utils.data.dataloader import default_collate
from eval_ppl import compute_logprobs

def prompt_accuracy(model, d_val, prompt_len, max_context, device, batch_size):
    print("Evaluating prompt rank...")
    num_samples = 30 

    #if args.model_path:
    #    if args.model_path == 'random':
    #        model.apply(model.init_weights)
    #    else:
    #        state = torch.load(args.model_path, map_location='cpu')
    #        try:
    #            model.load_state_dict(state)
    #        except:
    #            print('Failed to load weights strictly. Trying unstrict...')
    #            model.load_state_dict(state, strict=False)

    model.eval()
    print('Model loaded.')
    
    print('Data loaded.')


    d_len = len(d_val)
    get_indices = list(np.random.randint(d_len, size=num_samples))
    hit = 0
    total = 0
    start = datetime.now()

    for i in get_indices:
        sample_ten = []
        sample_prompts = []
        text = d_val[i]
        prompt = []
        story = []
        sample_ten.append(text)

        # get story
        #cur_prompt, cur_story = text.split('---\n')
        cur_prompt, cur_story = text
        prompt.append(cur_prompt)
        story.append(cur_story)
        sample_prompts.append('Prompt: ' + cur_prompt.strip() + '\n---\n')
        # sample 9 separate prompts
        while True:
            sample_nine = list(np.random.randint(d_len, size=10))
            if i not in sample_nine:
                break
        for j in sample_nine:
            #wrong_prompt = d_val[j].split('---\n')[0]
            wrong_prompt = d_val[j][0]
            
            wrong_text = 'Prompt: ' + wrong_prompt.strip() + '\n---\n' + cur_story.strip()
            prompt.append(wrong_prompt)
            story.append(cur_story)
            sample_ten.append(wrong_text)
            sample_prompts.append('Prompt: ' + wrong_prompt.strip() + '\n---\n')
        
        print('Running evaluation...')
        with torch.no_grad():
            logls = []
            batch = []

            prompt_lens = [len(model.decoder_tokenizer.encode(p)) for p in sample_prompts]
            # tokenise and create batch
            for text in sample_ten:
                bpe_tokens = model.decoder_tokenizer.encode(text)
                # TODO (This limit applies to GPT2)
                bpe_tokens = bpe_tokens[:1025]
                # Pad
                batch.append((bpe_tokens + [0] * (1025 - len(bpe_tokens)), len(bpe_tokens)))

            x, x_lens = zip(*batch)
            token_tensor = torch.tensor(x, dtype=torch.long, device=device)

            # Compute log probs
            lps = compute_logprobs(model, prompt, story, prompt_len, max_context, device)
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
