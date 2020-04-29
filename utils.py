import numpy as np
import random
import torch
import torch.nn.functional as F


def pil_loader(path):
    from PIL import Image
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def top_p_logits(logits, p=0.9):
    """
    Masks everything but the top probability entries as -infinity.
    """
    if p == 1:
        return logits
    else:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)
        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < p
        # First mask must always be pickable
        mask = F.pad(mask[:, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        return torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)


def sample_sequence(model, length, encoder_hidden=None, batch_size=None, context=None, temperature=1, top_p=0.9, device='cuda', sample=True, eos_token=None, tokenizer=None):
    assert context is not None

    if not torch.is_tensor(context):
        context = torch.tensor(context, device=device, dtype=torch.long)
    if len(context.size()) < 2:
        context = context.unsqueeze(0).repeat(batch_size, 1)
    
    prev = context
    output = context
    logprobs = 0
    mem = None
    with torch.no_grad():
        for i in range(length):
            logits, mem = model(prev, past=mem, encoder_hidden=encoder_hidden)
            logits = logits[:, -1, :] / temperature

            logits = top_p_logits(logits, p=top_p)
                
            probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(probs, num_samples=1)
            else:
                _, prev = torch.topk(probs, k=1, dim=-1)

            logprobs += probs.gather(1, prev).squeeze(-1)
            
            output = torch.cat((output, prev), dim=1)

            # print(output)
            # input()
            
            # Early break
            if prev.size(0) == 1 and prev.item() == eos_token:
                break
    
    return output, logprobs / output.size(1)

"""
def beam_search(data, k):
    # algo outline for beam search
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences
"""

def random_truncate(text, window):
    """ Randomly truncates text to window size """
    if len(text) > window:
        # Randomly truncate the text
        start_idx = random.randint(0, len(text) - window - 1)
        text = text[start_idx:start_idx+window] # to account sep and cls tokens
    
    return text


def collate_fn_masked(samples):
    """ Creates a batch out of samples """
    max_len = max(map(len, samples))
    
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(x) + [0] * (max_len - len(x)) for x in samples])
    x = torch.LongTensor([x + [0] * (max_len - len(x)) for x in samples])
    
    return x[:, :-1], x[:, 1:].contiguous(), x_mask[:, 1:]


def get_pos_enc_table(side_len, emb_dim=16):
    """ Init the sinusoid position encoding table """

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(side_len)])
    
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_pos_embeddings(side_len, emb_dim=16, device='cuda'):
    x = get_pos_enc_table(side_len, emb_dim).view(1, emb_dim, side_len, 1).repeat(1, 1, 1, side_len)
    y = get_pos_enc_table(side_len, emb_dim).view(1, emb_dim, 1, side_len).repeat(1, 1, side_len, 1)
    # print(x.shape, y.shape)

    return torch.cat((x, y), dim=1).to(device)


def wp_preprocess(text):
    # Standardize some symbols
    text = text.replace('<newline>', '\n')
    text = text.replace('``', '"')
    text = text.replace("''", '"')

    # Detokenize
    text = re.sub(' +', ' ', text)
    text = re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', text)
    text = re.sub('" (.*) "', '"\g<1>"', text)
    text = text.replace(" n't", "n't")

    return text


def set_all_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
