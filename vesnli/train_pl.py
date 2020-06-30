import math
import os
import pickle as pkl
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import *

from dataset import (LABEL_TOKENS, SPECIAL_TOKENS, SPECIAL_TOKENS_DICT,
                     ImageTextDataset, collate_fn, get_data)
from ImageGPT2 import ImageGPT2LMHeadModel


def average_distributed_scalar(scalar):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    scalar_t = scalar.float() / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t


class ImageGPT2_Model(pl.LightningModule):
    def __init__(self, hparams, tokenizer):
        super(ImageGPT2_Model, self).__init__()
        self.hparams = hparams
        self.model = ImageGPT2LMHeadModel.from_pretrained("gpt2")
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        # self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str,
                            default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE", help="Path of the dataset")
        parser.add_argument("--no_pre_saved_data", action="store_true",
                            help="Whether to use pre saved data or compute fresh")
        parser.add_argument("--pre_saved_data_path", type=str,
                            default="/home/hdd1/vibhav/VE-SNLI/DSTC8-AVSD-vibhav/vesnli/data/lbl1_expl_out", help="Path of the folder where dataset is pre saved")
        parser.add_argument("--train_batch_size", type=int,
                            default=4, help="Batch size for training")
        parser.add_argument("--valid_batch_size", type=int,
                            default=4, help="Batch size for validation")
        parser.add_argument("--n_epochs", type=int, default=10,
                            help="Number of training epochs")
        parser.add_argument("--lr", type=float,
                            default=6.25e-5, help="Learning rate")
        parser.add_argument("--max_norm", type=float,
                            default=1.0, help="Clipping gradient norm")
        parser.add_argument("--gradient_accumulation_steps", type=int,
                            default=8, help="Accumulate gradients on several steps")
        return parser

    def setup(self, stage):
        if self.hparams.no_pre_saved_data:
            dev_data = get_data(self.tokenizer, self.hparams.data_path, 'dev')
            train_data = get_data(
                self.tokenizer, self.hparams.data_path, 'train')
        else:
            dev_data_path = os.path.join(self.hparams.pre_saved_data_path,
                                         'dev_vesnli_gpt2.pkl')
            train_data_path = os.path.join(self.hparams.pre_saved_data_path,
                                           'train_vesnli_gpt2.pkl')
            dev_data = pkl.load(open(dev_data_path, 'rb'))
            train_data = pkl.load(open(train_data_path, 'rb'))

        self.train_dataset = ImageTextDataset(train_data, self.tokenizer)
        self.dev_dataset = ImageTextDataset(dev_data, self.tokenizer)

    def train_dataloader(self):
        log.info('Training data loader called.')
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.hparams.train_batch_size,
                                      #   num_workers=10,
                                      collate_fn=lambda x: collate_fn(x,
                                                                      self.tokenizer.pad_token_id))
        return train_dataloader

    def val_dataloader(self):
        log.info('Validation data loader called.')
        dev_dataloader = DataLoader(self.dev_dataset,
                                    batch_size=self.hparams.valid_batch_size,
                                    # num_workers=10,
                                    collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id))
        return dev_dataloader

    def forward(self, batch):
        image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask = batch
        input_embs = self.model.transformer.wte(input_ids)
        image_embs = self.model.image_fc(image)
        input_embs = torch.cat([image_embs, input_embs], dim=1)
        expl_out = self.model(input_embs,
                              token_type_ids=token_type_ids,
                              labels=lm_labels_expl,
                              attention_mask=[sec_mask, input_mask],)
        lbl_out = self.model(input_embs,
                             token_type_ids=token_type_ids,
                             labels=lm_labels_lbl,
                             attention_mask=[sec_mask, input_mask],)
        return expl_out, lbl_out

    def training_step(self, batch, batch_nb):
        image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask = batch
        expl_out, lbl_out = self.forward(batch)
        expl_loss = expl_out[0]
        lbl_loss, lbl_lm_logits, _ = lbl_out

        loss = (expl_loss + lbl_loss) / \
            self.hparams.gradient_accumulation_steps
        lbl_lm_logits = lbl_lm_logits.argmax(dim=2)
        lbl_accuracy = (lm_labels_lbl == lbl_lm_logits).float(
        ).sum() / len(lm_labels_lbl)
        expl_ppl = torch.exp(expl_loss / lm_labels_expl.shape[1])
        return {
            'loss': loss,
            'lbl_accuracy': lbl_accuracy,
            'expl_ppl': expl_ppl
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_lbl_accuracy = torch.stack(
            [x['lbl_accuracy'] for x in outputs]).mean()
        avg_expl_ppl = torch.stack(
            [x['expl_ppl'] for x in outputs]).mean()
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            avg_loss = average_distributed_scalar(avg_loss)
            avg_lbl_accuracy = average_distributed_scalar(avg_lbl_accuracy)
            avg_expl_ppl = average_distributed_scalar(avg_expl_ppl)
        return {
            'train_loss': avg_loss,
            'train_lbl_acc': avg_lbl_accuracy,
            'train_expl_ppl': avg_expl_ppl,
            'progress_bar': {
                'train_lbl_acc': avg_lbl_accuracy,
            },
            'log': {
                'train_loss': avg_loss,
                'train_lbl_acc': avg_lbl_accuracy,
                'train_expl_ppl': avg_expl_ppl,
            }
        }

    def validation_step(self, batch, batch_nb):
        image, input_ids, token_type_ids, lm_labels_expl, lm_labels_lbl, input_mask, sec_mask = batch
        expl_out, lbl_out = self.forward(batch)
        expl_loss = expl_out[0]
        lbl_loss, lbl_lm_logits, _ = lbl_out

        loss = (expl_loss + lbl_loss) / \
            self.hparams.gradient_accumulation_steps
        lbl_lm_logits = lbl_lm_logits.argmax(dim=2)
        lbl_accuracy = (lm_labels_lbl == lbl_lm_logits).float(
        ).sum() / len(lm_labels_lbl)
        expl_ppl = torch.exp(expl_loss / lm_labels_expl.shape[1])
        return {
            'loss': loss,
            'lbl_accuracy': lbl_accuracy,
            'expl_ppl': expl_ppl
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_lbl_accuracy = torch.stack(
            [x['lbl_accuracy'] for x in outputs]).mean()
        avg_expl_ppl = torch.stack(
            [x['expl_ppl'] for x in outputs]).mean()
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            avg_loss = average_distributed_scalar(avg_loss)
            avg_lbl_accuracy = average_distributed_scalar(avg_lbl_accuracy)
            avg_expl_ppl = average_distributed_scalar(avg_expl_ppl)
        return {
            'val_loss': avg_loss,
            'val_lbl_acc': avg_lbl_accuracy,
            'val_expl_ppl': avg_expl_ppl,
            'progress_bar': {
                'val_loss': avg_loss,
                'val_lbl_acc': avg_lbl_accuracy,
            },
            'log': {
                'val_loss': avg_loss,
                'val_lbl_acc': avg_lbl_accuracy,
                'val_expl_ppl': avg_expl_ppl,
            }
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       0, self.hparams.n_epochs)
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None):
        if current_epoch % self.hparams.gradient_accumulation_steps == 0:
            # if current_epoch > 0:
            # lr_red = self.hparams.lr / \
            #     (self.hparams.n_epochs * len(self.train_dataloader()))
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = self.learning_rate - lr_red
            optimizer.step()
            optimizer.zero_grad()


def main(hparams):
    pl.seed_everything(42)
    torch.cuda.manual_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_tokens(LABEL_TOKENS)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    # getattr(model, 'module', model).config.to_json_file(
    #         os.path.join(output_dir, CONFIG_NAME))

    t = datetime.today()
    tb_logger = pl.loggers.TensorBoardLogger(
        'lightning_logs',
        name='',
        version=f"{t.month}_{t.day}/{t.hour}_{t.minute}_{t.second}")
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(
            tb_logger.log_dir, 'checkpoints/{epoch}-{val_loss:.2f}'),
        save_top_k=8,
        verbose=True,
        monitor='val_loss',
        mode='min')
    lr_logger = LearningRateLogger()

    if hparams.use_checkpoint:
        model = ImageGPT2_Model.load_from_checkpoint(hparams.use_checkpoint,
                                                     tokenizer=tokenizer)
        trainer = pl.Trainer(
            gpus=hparams.num_gpus,
            resume_from_checkpoint=hparams.use_checkpoint,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback,
            logger=tb_logger,
        )
    else:
        model = ImageGPT2_Model(hyperparams, tokenizer)
        trainer = pl.Trainer(
            gpus=hparams.num_gpus,
            max_epochs=hparams.n_epochs,
            #  accumulate_grad_batches=hparams.gradient_accumulation_steps,
            #  distributed_backend='ddp_spawn',
            deterministic=True,
            gradient_clip_val=hyperparams.max_norm,
            num_sanity_val_steps=0,
            #  fast_dev_run=True,
            # overfit_batches=1,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback,
            logger=tb_logger,
        )
    trainer.fit(model)


if __name__ == "__main__":
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--use_checkpoint", type=str,
                               default=None, help="Checkpoint path if used else None")
    parent_parser.add_argument("--num_gpus", type=int,
                               default=2, help="No. of GPUs, if >1 then ddp used")
    parser = ImageGPT2_Model.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
    main(hyperparams)
