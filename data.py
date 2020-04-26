import numpy as np
import re
import torch


class PromptsDataset(torch.utils.data.Dataset):
    def __init__(self, source, target, preprocess=True):
        super(PromptsDataset, self).__init__()

        self.preprocess = preprocess

        with open(source, errors='ignore') as fs:
            with open(target, errors='ignore') as ft:
                self.samples = list(zip(fs.readlines(), ft.readlines()))


    def wp_preprocess(self, text):
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

    def __getitem__(self, index):
        prompt, story = self.samples[index]

        prompt = prompt[prompt.find("]") + 1:]
        prompt = re.sub('\[ (.*) \]', '', prompt)

        if self.preprocess:
            prompt = self.wp_preprocess(prompt.strip())
            story = self.wp_preprocess(story.strip())
        
        return prompt, story


    def __len__(self):
        return len(self.samples)


def load_data(dataset_dir, preprocess=True, max_samples=None):
    train_dataset = PromptsDataset(dataset_dir + '/train.wp_source', dataset_dir + '/train.wp_target', preprocess)
    val_dataset = PromptsDataset(dataset_dir + '/valid.wp_source', dataset_dir + '/valid.wp_target', preprocess)
    test_dataset = PromptsDataset(dataset_dir + '/test.wp_source', dataset_dir + '/test.wp_target', preprocess)
    
    if max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(max_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(max_samples))
    
    print('Number of Samples --> Train:%d, Val:%d, Test:%d' %(len(train_dataset), len(val_dataset), len(test_dataset)))

    return train_dataset, val_dataset, test_dataset