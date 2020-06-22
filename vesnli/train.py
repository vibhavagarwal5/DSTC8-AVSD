import math
import os
import pickle as pkl
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pprint import pformat

import torch
from ignite.contrib.handlers.param_scheduler import PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import (
    OptimizerParamsHandler, OutputHandler, TensorboardLogger)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from transformers import *

from dataset import (PADDED_INPUTS, SPECIAL_TOKENS, SPECIAL_TOKENS_DICT,
                     ImageTextDataset, collate_fn, get_data)
from ImageGPT2 import ImageGPT2LMHeadModel


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float,
                            device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_data_loaders(args, tokenizer):
    if args.no_pre_saved_data:
        dev_data = get_data(tokenizer, args.data_path, 'dev')
        train_data = get_data(tokenizer, args.data_path, 'train')
    else:
        dev_data_path = os.path.join(args.data_path, 'dev_vesnli_gpt2.pkl')
        train_data_path = os.path.join(args.data_path, 'train_vesnli_gpt2.pkl')
        dev_data = pkl.load(open(dev_data_path, 'rb'))
        train_data = pkl.load(open(train_data_path, 'rb'))

    dev_dataset = ImageTextDataset(dev_data, tokenizer)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=args.valid_batch_size,
                                shuffle=(not args.distributed),
                                num_workers=4,
                                collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    train_dataset = ImageTextDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed),
                                  num_workers=4,
                                  collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
    return train_dataloader, dev_dataloader


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE", help="Path of the dataset")
    parser.add_argument("--no_pre_saved_data", action="store_true",
                        help="Whether to use pre saved data or compute fresh")
    parser.add_argument("--model_checkpoint", type=str,
                        default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=4, help="Batch size for validation")
    parser.add_argument("--drop_rate", type=float,
                        default=0.5, help="drop rate for caption")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float,
                        default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float,
                        default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to 00, 01, 02 or 03 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--log_path", type=str,
                        default="log/", help="Log path")
    parser.add_argument("--debug", action="store_true",
                        help='Enable debug mode for only print')
    return parser.parse_args()


def train():
    args = get_args()

    '''Setup'''
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    print(f"Running process {args.local_rank}")
    print(f"Arguments: {pformat(args)}")

    '''Initialize distributed training if needed'''
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    print("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = ImageGPT2LMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    '''
    Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    '''
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
        model = model.module

    print("Prepare datasets")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    '''Training function and trainer'''
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        image, input_ids, token_type_ids, lm_labels, input_mask, sec_mask = batch
        input_embs = model.transformer.wte(input_ids)
        image_embs = model.image_fc(image)
        input_embs = torch.cat([image_embs, input_embs], dim=1)
        print(tokenizer.convert_ids_to_tokens(input_ids[0]))
        print(tokenizer.convert_ids_to_tokens(token_type_ids[0]))
        print(tokenizer.convert_ids_to_tokens(lm_labels[0]))
        loss = model(input_embs,
                     token_type_ids=token_type_ids,
                     labels=lm_labels,
                     attention_mask=[sec_mask, input_mask],)[0]

        loss = (loss) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    '''Evaluation function and evaluator (evaluator output is the input of the metrics)'''
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            image, input_ids, token_type_ids, lm_labels, input_mask, sec_mask = batch
            input_embs = model.transformer.wte(input_ids)
            image_embs = model.image_fc(image)
            input_embs = torch.cat([image_embs, input_embs], dim=1)

            lm_logits = model(input_embs,
                              token_type_ids=token_type_ids,
                              attention_mask=[sec_mask, input_mask])[0]
            # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1,
                                               :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    '''Engines'''
    trainer = Engine(update)
    evaluator = Engine(inference)

    '''
    Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    '''
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED,
                                  lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED,
                                  lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr",
                                [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    '''Prepare metrics - note how we compute distributed metrics'''
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1),
                    output_transform=lambda x: (x[0], x[1]))
    }
    metrics.update({
        "average_nll": MetricsLambda(average_distributed_scalar,
                                     metrics["nll"], args)
    })
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    '''
    On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    '''
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
            "Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir="./tb_logs")
        tb_logger.attach(trainer,
                         log_handler=OutputHandler(
                             tag="training",
                             metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer,
                         log_handler=OptimizerParamsHandler(optimizer),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator,
                         log_handler=OutputHandler(
                             tag="validation",
                             metric_names=list(metrics.keys()),
                             another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.log_path,
                                             'checkpoint',
                                             n_saved=8,
                                             require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1),
                                  checkpoint_handler,
                                  {'mymodel': getattr(model, 'module', model)})
        # "getattr" take care of distributed encapsulation

        torch.save(args, args.log_path + 'model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(
            os.path.join(args.log_path, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_path)

    '''Run the training'''
    trainer.run(train_loader, max_epochs=args.n_epochs)

    '''
    On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    '''
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(checkpoint_handler._saved[-1][1][-1],
                  os.path.join(args.log_path, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == "__main__":
    train()
