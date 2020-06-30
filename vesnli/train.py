import logging
import math
import os
import pickle as pkl
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
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
from torch.utils.data import DataLoader, Subset
from transformers import *

from dataset import (LABEL_TOKENS, SPECIAL_TOKENS, SPECIAL_TOKENS_DICT,
                     ImageTextDataset, collate_fn, get_data)
from ImageGPT2 import ImageGPT2LMHeadModel

logger = logging.getLogger(__file__)


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
        logger.info("No saved data loaded")
        dev_data = get_data(tokenizer, args.data_path, 'dev')
        train_data = get_data(tokenizer, args.data_path, 'train')
    else:
        logger.info("Loaded saved data")
        dev_data_path = os.path.join(
            args.pre_saved_data_path, 'dev_vesnli_gpt2.pkl')
        train_data_path = os.path.join(
            args.pre_saved_data_path, 'train_vesnli_gpt2.pkl')
        dev_data = pkl.load(open(dev_data_path, 'rb'))
        train_data = pkl.load(open(train_data_path, 'rb'))

    dev_dataset = ImageTextDataset(dev_data, tokenizer)
    train_dataset = ImageTextDataset(train_data, tokenizer)
    if args.small_data != -1:
        logger.info('Using small subset of data')
        dev_dataset = Subset(dev_dataset, list(range(args.small_data)))
        train_dataset = Subset(train_dataset, list(range(args.small_data)))

    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=args.valid_batch_size,
                                shuffle=(not args.distributed),
                                num_workers=4,
                                collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
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
    parser.add_argument("--pre_saved_data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/DSTC8-AVSD-vibhav/vesnli/data/lbl1_expl_out", help="Path of the folder where dataset is pre saved")
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
    parser.add_argument("--output_folder", type=str,
                        default='./output', help="output storage path")
    parser.add_argument("--small_data", type=int,
                        default=-1, help='small data size')
    return parser.parse_args()


def main():
    args = get_args()

    '''Setup'''
    t = datetime.today()
    output_dir = os.path.join(
        args.output_folder, f"{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(filename=os.path.join(output_dir, 'app.log'),
                        filemode='a',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # This is a logger.warning: it will be printed by all distributed processes
    logger.warning(f"Running process {args.local_rank}")
    logger.info(f"Arguments: {pformat(args)}")

    '''Initialize distributed training if needed'''
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    logger.info(
        "Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = ImageGPT2LMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_tokens(LABEL_TOKENS)
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

    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    '''Training function and trainer'''
    def train(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask = batch
        input_embs = model.transformer.wte(input_ids)
        image_embs = model.image_fc(image)
        input_embs = torch.cat([image_embs, input_embs], dim=1)

        expl_out = model(input_embs,
                         token_type_ids=token_type_ids,
                         labels=lm_labels_expl,
                         attention_mask=[sec_mask, input_mask],)
        lbl_out = model(input_embs,
                        token_type_ids=token_type_ids,
                        labels=lm_labels_lbl,
                        attention_mask=[sec_mask, input_mask],)
        expl_loss, _, _ = expl_out
        lbl_loss, lbl_lm_logits, _ = lbl_out

        loss = (expl_loss + lbl_loss) / args.gradient_accumulation_steps
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
        lbl_lm_logits = lbl_lm_logits.argmax(dim=2)
        lbl_accuracy = (lm_labels_lbl == lbl_lm_logits).float(
        ).sum() / len(lm_labels_lbl)
        return {
            'loss': loss.item(),
            'lbl_accuracy': lbl_accuracy.item()
        }

    '''Validation function and validator (validator output is the input of the metrics)'''
    def validation(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask = batch
            input_embs = model.transformer.wte(input_ids)
            image_embs = model.image_fc(image)
            input_embs = torch.cat([image_embs, input_embs], dim=1)

            expl_lm_logits = model(input_embs,
                                   token_type_ids=token_type_ids,
                                   attention_mask=[sec_mask, input_mask],)[0]
            lbl_lm_logits = model(input_embs,
                                  token_type_ids=token_type_ids,
                                  attention_mask=[sec_mask, input_mask],)[0]

            expl_lm_logits_flat_shifted = expl_lm_logits[..., :-1,
                                                         :].contiguous().view(-1, expl_lm_logits.size(-1))
            lm_labels_expl_flat_shifted = lm_labels_expl[..., 1:].contiguous(
            ).view(-1)
            lbl_lm_logits_flat_shifted = lbl_lm_logits[..., :-1,
                                                       :].contiguous().view(-1, lbl_lm_logits.size(-1))
            lm_labels_lbl_flat_shifted = lm_labels_lbl[..., 1:].contiguous(
            ).view(-1)

            return expl_lm_logits_flat_shifted, lm_labels_expl_flat_shifted, lbl_lm_logits_flat_shifted, lm_labels_lbl_flat_shifted

    '''Engines'''
    trainer = Engine(train)
    validator = Engine(validation)

    '''Linearly decrease the learning rate from lr to zero'''
    scheduler = PiecewiseLinear(optimizer, "lr",
                                [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    '''
    Attach validation to trainer: we evaluate when we start the training and at the end of each epoch
    '''
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: validator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED,
                                  lambda _: validator.run(val_loader))

    '''Prepare metrics - note how we compute distributed metrics'''
    RunningAverage(output_transform=lambda x: x['loss']).attach(
        trainer, "loss")
    RunningAverage(output_transform=lambda x: 100 * x['lbl_accuracy']).attach(
        trainer, "lbl_accuracy")
    RunningAverage(output_transform=lambda x: math.exp(
        average_distributed_scalar(x['loss'], args))).attach(trainer, "ppl")

    metrics = {
        "expl_loss": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1),
                          output_transform=lambda x: (x[0], x[1])),
        "lbl_loss": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1),
                         output_transform=lambda x: (x[2], x[3])),
        'lbl_accuracy_tmp1': Accuracy(output_transform=lambda x: (x[2], x[3]))
    }
    metrics["loss_tmp"] = MetricsLambda(lambda e, l, a: (e + l) / a.gradient_accumulation_steps,
                                        metrics["expl_loss"], metrics["lbl_loss"], args)
    metrics["loss"] = MetricsLambda(average_distributed_scalar,
                                    metrics["loss_tmp"], args)
    metrics["ppl"] = MetricsLambda(math.exp, metrics["loss"])
    metrics["lbl_accuracy"] = MetricsLambda(lambda x: 100 * x,
                                            metrics["lbl_accuracy_tmp1"])
    for name, metric in metrics.items():
        metric.attach(validator, name)

    '''
    On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    '''
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss", 'lbl_accuracy', 'ppl'])
        validator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message(
                                        "Validation: %s" % pformat(validator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=output_dir)
        tb_logger.attach(trainer,
                         log_handler=OptimizerParamsHandler(optimizer),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(trainer,
                         log_handler=OutputHandler(
                             tag="training",
                             metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer,
                         log_handler=OutputHandler(
                             tag="training",
                             metric_names=["lbl_accuracy", "ppl"]),
                         event_name=Events.EPOCH_COMPLETED)

        def global_step_transform(*args, **kwargs):
            return trainer.state.iteration
        tb_logger.attach(validator,
                         log_handler=OutputHandler(
                             tag="validation",
                             metric_names=[
                                 'ppl', 'loss', 'lbl_accuracy'],
                             global_step_transform=global_step_transform),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(output_dir,
                                             'checkpoint',
                                             n_saved=8,
                                             require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1),
                                  checkpoint_handler,
                                  {'mymodel': getattr(model, 'module', model)})

        # "getattr" take care of distributed encapsulation
        torch.save(args, os.path.join(output_dir, 'model_training_args.bin'))
        getattr(model, 'module', model).config.to_json_file(
            os.path.join(output_dir, CONFIG_NAME))
        # tokenizer.save_vocabulary(output_dir)

    '''Run the training'''
    trainer.run(train_loader, max_epochs=args.n_epochs)

    '''
    On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    '''
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(os.path.join(output_dir, checkpoint_handler._saved[-1][1]),
                  os.path.join(output_dir, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == "__main__":
    main()
