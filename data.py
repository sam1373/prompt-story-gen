import numpy as np
import re
import torch


class PromptsDataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        super(PromptsDataset, self).__init__()

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
        prompt = self.wp_preprocess(prompt.strip())

        story = self.wp_preprocess(story.strip())
        
        return prompt, story


    def __len__(self):
        return len(self.samples)


def load_data(dataset_dir, batch_size, num_workers=0, max_samples=None):
    train_dataset = PromptsDataset(dataset_dir + '/train.wp_source', dataset_dir + '/train.wp_target')
    val_dataset = PromptsDataset(dataset_dir + '/valid.wp_source', dataset_dir + '/valid.wp_target')
    test_dataset = PromptsDataset(dataset_dir + '/test.wp_source', dataset_dir + '/test.wp_target')
    
    if max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(max_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(max_samples))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('Number of Samples --> Train:%d\t Val:%d\t Test:%d' %(len(train_dataset), len(val_dataset), len(test_dataset)))

    return train_loader, val_loader, test_loader