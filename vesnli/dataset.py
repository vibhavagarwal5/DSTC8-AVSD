import json
import logging
import pickle
import os
from itertools import chain

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>",
                  "<image>", '<label>', '<hypothesis>', '<expl>']
SPECIAL_TOKENS_DICT = {
    'bos_token': "<bos>",
    'eos_token': "<eos>",
    'pad_token': "<pad>",
    'additional_special_tokens': ["<image>", '<label>', '<hypothesis>', '<expl>'],
}
PADDED_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.encode(obj)
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


def get_data(tokenizer, data_path, data_type, to_save=False):
    files = ['images_tensor', 'expl_1', 'labels', 's2']
    data = {fl: [tokenizer.encode(line.rstrip()) for line in open(
        os.path.join(data_path, f"{fl}.{data_type}"), 'r')] for fl in files[1:]}
    data['image'] = torch.load(os.path.join(data_path,
                                            f"{files[0]}.{data_type}"))
    data['image'] = data['image'].reshape((data['image'].shape[0],
                                           1,
                                           data['image'].shape[1])).numpy()

    input_ids = []
    token_type_ids = []
    additional_special_tokens = tokenizer.additional_special_tokens_ids
    for i in range(len(data['labels'])):
        input_ids.append([additional_special_tokens[1]] + data['labels'][i] +
                         [additional_special_tokens[2]] + data['s2'][i] +
                         [additional_special_tokens[3]] + data['expl_1'][i] +
                         [tokenizer.eos_token_id])
        token_type_ids.append(
            # [additional_special_tokens[0]] +
            [additional_special_tokens[1]] * (len(data['labels'][i]) + 1) +
            [additional_special_tokens[2]] * (len(data['s2'][i]) + 1) +
            [additional_special_tokens[3]] * (len(data['expl_1'][i]) + 1 + 1))
    data['input_ids'] = input_ids
    data['token_type_ids'] = token_type_ids
    data['lm_labels'] = [
        [-1] * (len([additional_special_tokens[1]] * (len(data['labels'][i]) + 1) +
                    [additional_special_tokens[2]] * (len(data['s2'][i]) + 1)))
        + [additional_special_tokens[3]] + data['expl_1'][i] +
        [tokenizer.eos_token_id] for i in range(len(data['expl_1']))
    ]

    if to_save:
        pickle.dump(data, open(f"./{data_type}_vesnli_gpt2.pkl", 'wb'))
    return data


class ImageTextDataset(Dataset):
    def __init__(self, all_data, tokenizer):
        self.all_data = all_data
        self.tokenizer = tokenizer
        # input_ids = []
        # token_type_ids = []
        # additional_special_tokens = self.tokenizer.additional_special_tokens_ids
        # for i in range(len(self.all_data['labels'])):
        #     input_ids.append([additional_special_tokens[1]] + self.all_data['labels'][i] +
        #                      [additional_special_tokens[2]] + self.all_data['s2'][i] +
        #                      [additional_special_tokens[3]] + self.all_data['expl_1'][i] +
        #                      [self.tokenizer.eos_token_id])
        #     token_type_ids.append(
        #         # [additional_special_tokens[0]] +
        #         [additional_special_tokens[1]] * (len(self.all_data['labels'][i]) + 1) +
        #         [additional_special_tokens[2]] * (len(self.all_data['s2'][i]) + 1) +
        #         [additional_special_tokens[3]] * (len(self.all_data['expl_1'][i]) + 1 + 1))
        # self.all_data['input_ids'] = input_ids
        # self.all_data['token_type_ids'] = token_type_ids
        # self.all_data['lm_labels'] = [
        #     [-1] * (len([additional_special_tokens[1]] * (len(self.all_data['labels'][i]) + 1) +
        #                 [additional_special_tokens[2]] * (len(self.all_data['s2'][i]) + 1)))
        #     + [additional_special_tokens[3]] + self.all_data['expl_1'][i] +
        #     [self.tokenizer.eos_token_id] for i in range(len(self.all_data['expl_1']))
        # ]

    def __len__(self):
        return len(self.all_data['image'])

    def __getitem__(self, index):
        image = self.all_data['image'][index]
        input_ids = torch.tensor(self.all_data['input_ids'][index]).long()
        token_type_ids = torch.tensor(
            self.all_data['token_type_ids'][index]).long()
        lm_labels = torch.tensor(self.all_data['lm_labels'][index]).long()
        return image, input_ids, token_type_ids, lm_labels


def collate_fn(batch, pad_token):
    def padding(seq, pad_token):
        max_len = max(len(s) for s in seq)
        # print(max_len, seq[0].size(), pad_token)
        padded_mask = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            padded_mask[i, :len(seq[i])] = seq[i]
        # print(padded_mask[1])
        # print(tokenizer.convert_ids_to_tokens(padded_mask[1]))
        return padded_mask

    image, input_ids, token_type_ids, lm_labels = [], [], [], []
    for i in batch:
        image.append(i[0])
        input_ids.append(i[1])
        token_type_ids.append(i[2])
        lm_labels.append(i[3])

    image = torch.tensor(image)
    input_ids = padding(input_ids, pad_token)
    token_type_ids = padding(token_type_ids, pad_token)

    # print(image.shape)
    lm_labels = padding(lm_labels, -1)
    image_label = torch.ones((len(image), 1)).long() * -1
    lm_labels = torch.cat([image_label, lm_labels], dim=1)

    input_mask = input_ids != pad_token
    input_mask = input_mask.long()
    image_mask = torch.ones((len(image), 1)).long()
    input_mask = torch.cat([image_mask, input_mask], dim=1)
    # print(input_mask)
    sec_mask = torch.zeros(input_mask.shape)
    # print(sec_mask)
    # print(sec_mask.shape, input_mask.shape)
    return image, input_ids, token_type_ids, lm_labels, input_mask, sec_mask


'''main'''
if __name__ == "__main__":
    from transformers import *
    from torch.utils.data import DataLoader

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    data = get_data(
        tokenizer, '../../mycode-vesnli/dataset/e-SNLI-VE', 'dev', to_save=True)
    dataset = ImageTextDataset(data, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    l = next(iter(dataloader))
    for i in l:
        print(i.shape)
