# Adapted from https://github.com/calclavia/story-generation/blob/master/analysis/eval_ppl.py

import re
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np

def compute_logprobs(model, prompt, story, prompt_len, max_context, device):
    prompt, story = default_collate([(prompt, story)])

    prompt = model.process_prompt(prompt, prompt_len).to(device)

    story, story_target, story_mask = model.process_story(story, max_context)
    story = story.to(device)
    story_target = story_target.to(device)
    story_mask = story_mask.to(device)

    logits = model(story, prompt)

    lprobs = torch.log_softmax(logits, dim=-1)
    # Extract the probability of the target token at each position
    lprobs = lprobs.gather(-1, story.unsqueeze(-1)).squeeze(-1)
    
    return lprobs

def word_level_ppl(target_tokens, lprobs, tokenizer, raw_token=None):
    assert len(target_tokens) == len(lprobs), (len(target_tokens), len(lprobs))

    # Convert BPE lprobs to word lprobs
    word_lprobs = []
    cur_lp = []
    new_add = ''
    i = 0
    start = False

    for token, lp in zip(target_tokens, lprobs):
        # Follow how it's detokenized.
        chars = tokenizer.decoder[token]
        new_add += bytearray([tokenizer.byte_decoder[c] for c in chars]).decode('utf-8', errors=tokenizer.errors)
        cur_lp.append(lp)

        if not start:
            # Wait for end of prompt
            start = '---\n' in new_add
            if start:
                cur_lp = []
                new_add = ''
            continue

        # Reverse preprocessing
        text = new_add
        text = re.sub('"', ' " ', text)
        text = re.sub('(\'|\.|\,|\:|\?|\!|;)', ' \g<1>', text)
        # Fix contraction
        text = text.replace("n 't", " n't")
        text = text.replace('\n', ' <newline> ')
        text = re.sub(' +', ' ', text)
        text = text.replace('. . .', '...')
        # Edge cases
        text = text.replace("ca n't-", "can't-")
        text = text.replace("St .", "St.")
        text = re.sub(r"//www \.(.*) \.(.*)/", r"//www\.\g<1>\.\g<1>\/", text)

        tokens = text.strip().split(' ')

        # Once a new word is starting to be formed, remove the previous one
        if len(tokens) > i + 1:
            # Token length changed, which means new word has been added.
            # Grab all but the last prob (excluding the unformed next word)
            word_lprobs.append(sum(cur_lp[:-1]))
            cur_lp = cur_lp[-1:]
            i += 1

    # Add final token
    word_lprobs.append(sum(cur_lp))

    token_diff = None
    if raw_token is not None:
        token_diff = abs(len(word_lprobs) - len(raw_token))

    word_lprobs = torch.tensor(word_lprobs)
    ppl = torch.exp(-word_lprobs.mean()).item()
    
    if ppl == float('inf'):
        raise Exception('Infinite PPL', raw_token)

    if ppl > 1000:
        print(ppl)
        print(word_lprobs)
        print(len(word_lprobs), len(raw_token))
        
        raise Exception('Large PPL', tokens, raw_token)
    return ppl, token_diff
    
def evaluate_ppl(model, val_dataset, val_dataset_raw, prompt_len, max_context, device, batch_size):
    """
    Calculates the perplexity given a model.
    """
    print('Computing model perplexity...')
    
    model.eval()
    with torch.no_grad():
        ppls = []
        word_ppls = []
        token_diffs = []
        num_errs = 0

        batch = []
        for sample_id, ((prompt, story), (prompt_raw, story_raw)) in enumerate(zip(val_dataset, val_dataset_raw)):
            text = 'Prompt: ' + prompt.strip() + '\n---\n' + story.strip()
            bpe_tokens = [model.decoder_tokenizer.encoder['<|endoftext|>']] + model.decoder_tokenizer.encode(text)
            
            bpe_tokens = bpe_tokens[:1025] # This limit applies to GPT2
            # Pad
            batch.append((bpe_tokens + [0] * (1025 - len(bpe_tokens)), len(bpe_tokens), story_raw.strip().split(' ')))

            if len(batch) == batch_size or len(word_ppls) == len(val_dataset) - 1:
                x, x_lens, raw_tokens = zip(*batch)
                token_tensor = torch.tensor(x, dtype=torch.long, device=device)

                # Compute log probs
                lps = compute_logprobs(model, prompt, story, prompt_len, max_context, device)
                #token_tensor = token_tensor.cpu().numpy()

                # Compute individually
                for i in range(lps.shape[0]):
                    try:
                        # Mask out some tokens
                        target_tokens = token_tensor[i, 1:x_lens[i]]
                        log_probs = lps[i, :x_lens[i] - 1]
                        ppl, token_diff = word_level_ppl(target_tokens, log_probs.cpu().float().numpy(), model.decoder_tokenizer, raw_tokens[i])
                        token_diffs.append(token_diff)
                        word_ppls.append(ppl)
                        ppls.append(torch.exp(-log_probs.mean()).item())
                    except Exception as e:
                        print('Skipping anomaly.')
                        print(e)
                        num_errs += 1
                print('World Level PPL {:.2f} BPE PPL {:.2f} Diff {:.2f} Done: {:.2f}% Skip {}'.format(
                    np.mean(word_ppls), np.mean(ppls), np.mean(token_diffs),
                    sample_id / len(val_dataset) * 100, num_errs
                ))
                batch = []
    
    return np.mean(word_ppls), np.mean(ppls), np.mean(token_diffs)
