import json
import logging
import pickle
import os
from itertools import chain

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<pad>", "<image>", '<label>', '<hypothesis>', '<expl>']
SPECIAL_TOKENS_DICT = {
    'pad_token': "<pad>",
    'additional_special_tokens': ["<image>", '<label>', '<hypothesis>', '<expl>'],
}
LABEL_TOKENS = ['neutral', 'contradiction', 'entailment']
PADDED_INPUTS = ["input_ids", "token_type_ids",
                 "lm_labels_expl", "lm_labels_lbl"]


def get_data(tokenizer, data_path, data_type, to_save=False, final_data_path=None):
    files = ['images_tensor', 'expl_1', 'labels', 's2']
    data = {fl: [tokenizer.encode(line.rstrip()) for line in open(
        os.path.join(data_path, f"{fl}.{data_type}"), 'r')] for fl in files[1:]}
    data['image'] = torch.load(os.path.join(data_path,
                                            f"{files[0]}.{data_type}"))
    data['image'] = data['image'].reshape((data['image'].shape[0],
                                           1,
                                           data['image'].shape[1])).numpy()

    input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl = [], [], [], []
    additional_special_tokens = tokenizer.additional_special_tokens_ids
    for i in range(len(data['labels'])):
        input_ids.append([additional_special_tokens[1]] +
                         [additional_special_tokens[2]] + data['s2'][i] +
                         [additional_special_tokens[3]] + data['expl_1'][i] +
                         [tokenizer.eos_token_id])
        token_type_ids.append(
            [additional_special_tokens[0]] * (len(data['image'][i])) +
            [additional_special_tokens[1]] * (len(data['labels'][i])) +
            [additional_special_tokens[2]] * (len(data['s2'][i]) + 1) +
            [additional_special_tokens[3]] * (len(data['expl_1'][i]) + 1 + 1))
        lm_labels_expl.append([-1] * (len(
            [additional_special_tokens[0]] * (len(data['image'][i])) +
            [additional_special_tokens[1]] * (len(data['labels'][i])) +
            [additional_special_tokens[2]] * (len(data['s2'][i]) + 1))) +
            [additional_special_tokens[3]] + data['expl_1'][i] + [tokenizer.eos_token_id])
        lm_labels_lbl.append(
            [-1] * (len([additional_special_tokens[0]] * (len(data['image'][i])))) +
            data['labels'][i] +
            [-1] * len([additional_special_tokens[2]] * (len(data['s2'][i]) + 1) +
                       [additional_special_tokens[3]] * (len(data['expl_1'][i]) + 1 + 1)))
    data['input_ids'] = input_ids
    data['token_type_ids'] = token_type_ids
    data['lm_labels_expl'] = lm_labels_expl
    data['lm_labels_lbl'] = lm_labels_lbl

    if to_save:
        if final_data_path is not None:
            file_dest = os.path.join(final_data_path,
                                     f"{data_type}_vesnli_gpt2.pkl")
            pickle.dump(data, open(file_dest, 'wb'))
        else:
            print('No save location.')
    return data


class ImageTextDataset(Dataset):
    def __init__(self, all_data, tokenizer):
        self.all_data = all_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.all_data['image'])

    def __getitem__(self, index):
        image = self.all_data['image'][index]
        input_ids = torch.tensor(self.all_data['input_ids'][index]).long()
        token_type_ids = torch.tensor(
            self.all_data['token_type_ids'][index]).long()
        lm_labels_expl = torch.tensor(
            self.all_data['lm_labels_expl'][index]).long()
        lm_labels_lbl = torch.tensor(
            self.all_data['lm_labels_lbl'][index]).long()
        return image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl


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

    image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl = [], [], [], [], []
    for i in batch:
        image.append(i[0])
        input_ids.append(i[1])
        token_type_ids.append(i[2])
        lm_labels_expl.append(i[3])
        lm_labels_lbl.append(i[4])

    image = torch.tensor(image)
    input_ids = padding(input_ids, pad_token)
    token_type_ids = padding(token_type_ids, pad_token)
    # print(image.shape)
    lm_labels_expl = padding(lm_labels_expl, -1)
    lm_labels_lbl = padding(lm_labels_lbl, -1)

    input_mask = input_ids != pad_token
    input_mask = input_mask.long()
    image_mask = torch.ones((len(image), 1)).long()
    input_mask = torch.cat([image_mask, input_mask], dim=1)
    # print(input_mask)
    sec_mask = torch.zeros(input_mask.shape)
    # print(sec_mask)
    # print(sec_mask.shape, input_mask.shape)
    return image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask


'''main'''
if __name__ == "__main__":
    from transformers import *
    from torch.utils.data import DataLoader
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str,
                        default="dev", help="dev or train or test")
    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE", help="Path of the dataset")
    parser.add_argument("--to_save", action="store_true",
                        help="To save the dataset processed or not")
    parser.add_argument("--final_data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/DSTC8-AVSD-vibhav/vesnli/data/lbl1_expl_out", help="Path of the folder where dataset is to be stored")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens(LABEL_TOKENS)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    data = get_data(tokenizer, args.data_path, args.data_type,
                    args.to_save, args.final_data_path)
    dataset = ImageTextDataset(data, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    l = next(iter(dataloader))
    for i, v in enumerate(l):
        print(v.shape)
        if i in [1, 2, 3, 4]:
            print(v[0])
            print(tokenizer.convert_ids_to_tokens(v[0]))

    # t = list(tokenizer.get_vocab().keys())
    # data_path = '../../mycode-vesnli/dataset/e-SNLI-VE'
    # files = ['images_tensor', 'expl_1', 'labels', 's2']
    # dev = {fl: [line.rstrip() for line in open(
    #     os.path.join(data_path, f"{fl}.dev"), 'r')] for fl in files[1:]}
    # train = {fl: [line.rstrip() for line in open(
    #     os.path.join(data_path, f"{fl}.train"), 'r')] for fl in files[1:]}
    # test = {fl: [line.rstrip() for line in open(
    #     os.path.join(data_path, f"{fl}.test"), 'r')] for fl in files[1:]}
    # total_sen = dev['expl_1'] + dev['s2'] + train['expl_1'] + train['s2']
    # print(total_sen[:10])
    # v = list(itertools.chain(*[i.split(' ') for i in total_sen]))
    # print(v[:10])
    # print(set(v) - set(t))
