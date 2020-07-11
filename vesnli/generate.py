import copy
import json
import logging
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from transformers import *

from dataset import (LABEL_TOKENS, SPECIAL_TOKENS,
                     SPECIAL_TOKENS_DICT, get_data)
from ImageGPT2 import ImageGPT2LMHeadModel, ImageGPT2DoubleHeadsModel


def generate_data(data, expl_inp, tokenizer):
    input_ids, token_type_ids = [], []
    additional_special_tokens = tokenizer.additional_special_tokens_ids
    input_ids.append([additional_special_tokens[2]] + data['s2'] +
                     [additional_special_tokens[3]] + expl_inp)
    token_type_ids.append(
        [additional_special_tokens[0]] * (len(data['image'])) +
        [additional_special_tokens[2]] * (len(data['s2']) + 1) +
        [additional_special_tokens[3]] * (len(expl_inp) + 1))
    data['input_ids'] = input_ids
    data['token_type_ids'] = token_type_ids
    return data


def top_kp_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < \
            torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = \
            sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(data, tokenizer, model, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = []
    for i in range(args.max_length):
        data = generate_data(data, current_output, tokenizer)
        input_ids = torch.tensor(data["input_ids"],
                                 device=args.device)
        token_type_ids = torch.tensor(data["token_type_ids"],
                                      device=args.device)
        image = torch.tensor(data['image'], device=args.device)
        input_embs = model.transformer.wte(input_ids)
        image_embs = model.image_fc(image)
        input_embs = torch.cat([image_embs, input_embs], dim=1)

        output = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = output[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_kp_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        if args.no_sample:
            prev = torch.topk(probs, 1)[1]
        else:
            # greedy decoding
            prev = torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() == tokenizer.eos_token_id:
            break
        current_output.append(prev.item())

    return current_output


def greedy_decode(data, tokenizer, model, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = []

    for i in range(args.max_length):
        data = generate_data(data, current_output, tokenizer)
        input_ids = torch.tensor(data["input_ids"],
                                 device=args.device)
        token_type_ids = torch.tensor(data["token_type_ids"],
                                      device=args.device)
        image = torch.tensor(data['image'], device=args.device)
        input_embs = model.transformer.wte(input_ids)
        image_embs = model.image_fc(image)
        input_embs = torch.cat([image_embs, input_embs], dim=1)

        logits = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = logits[0][0]
        logits = logits.cpu().data.numpy()
        next_word = np.argsort(logits[-1])[-1]
        # if next_word == tokenizer.eos_token_id:
        #     break
        current_output.append(next_word)
    return current_output


def beam_search(data, tokenizer, model, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    hyplist = [([], 0, [])]
    best_state = None
    comp_hyplist = []

    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            data = generate_data(data, st, tokenizer)
            input_ids = torch.tensor(data["input_ids"],
                                     device=args.device)
            token_type_ids = torch.tensor(data["token_type_ids"],
                                          device=args.device)
            image = torch.tensor(data['image'], device=args.device)
            input_embs = model.transformer.wte(input_ids)
            image_embs = model.image_fc(image)
            input_embs = torch.cat([image_embs, input_embs], dim=1)

            output = model(input_embs, token_type_ids=token_type_ids)
            if "gpt2" == args.model:
                logits = output[0]
            logp = F.log_softmax(logits, dim=-1)[:, -1, :]
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + \
                    args.penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist),
                                     key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist),
                                     key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]


# Evaluation routine
def generate_response(tokenizer, model, data, args):
    result_expl = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for idx, expl in enumerate(data['expl_1']):
            logging.info('Org. expl: ' + tokenizer.decode(expl))
            # prepare input data
            data_ = {}
            data_['image'] = [data['image'][idx]]
            data_['s2'] = data['s2'][idx]
            start_time = time.time()
            if args.beam_search:
                hypstr = beam_search(data_, tokenizer, model, args)
                hypstr = hypstr[0][0]
            # elif args.no_sample:
            #     hypstr = greedy_decode(data_, tokenizer, model, args)
            else:
                hypstr = sample_sequence(data_, tokenizer, model, args)
            hypstr = tokenizer.decode(hypstr, skip_special_tokens=True)
            logging.info('Generated expl: ' + hypstr)
            result_expl.append(hypstr)
            logging.info('ElapsedTime: %f' % (time.time() - start_time))
            logging.info('-----------------------')

    return result_expl


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str,
                        default="log_without_caption_with_valid/", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true',
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", action='store_true',
                        help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=3,
                        help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float,
                        default=0.3, help="length penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7,
                        help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--test_set_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE")
    parser.add_argument("--output", type=str, default="result.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Loading model params from ' + args.model_checkpoint)

    tokenizer_class = GPT2Tokenizer if args.model == "gpt2" else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_tokens(LABEL_TOKENS)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_class = ImageGPT2DoubleHeadsModel if args.model == "gpt2" else OpenAIGPTLMHeadModel
    model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint,
                                        config=model_config)
    model.to(args.device)

    logging.info('Loading test data from ' + args.test_set_path)
    data = get_data(tokenizer, args.test_set_path, 'test')
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result = generate_response(tokenizer, model, data, args)
    logging.info('----------------')
    logging.info('Time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
