import os
import json
import math
import logging
from pathlib import Path
from typing import Tuple

import hydra
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from accelerate.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from transformers import AutoConfig, PreTrainedModel, AutoModelForCausalLM, get_scheduler

import wandb

from .midi_data_collator import MidiDataCollatorForCausalLM

load_dotenv()


logger = get_logger(__name__)


def setup_logging(
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_tokenizer(
    config: DictConfig,
) -> ExponentialTimeTokenizer:
    tokenizer_config = {
        "time_unit": config.tokenizer.min_time_unit,
        "max_time_step": config.tokenizer.max_time_step,
        "n_velocity_bins": config.tokenizer.n_velocity_bins,
        "n_special_ids": config.tokenizer.n_special_ids,
    }
    tokenizer = ExponentialTimeTokenizer.build_tokenizer(
        tokenizer_config,
    )
    return tokenizer


def load_and_prepare_dataset(
    config: DictConfig,
    tokenizer: ExponentialTimeTokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Load train/eval from JSONL directory. When streaming=True, train is streamed; eval is always full."""
    dataset_path = to_absolute_path(config.data.dataset_path)
    path = Path(dataset_path)

    if not config.data.load_from_disk:
        raise ValueError("config.data.load_from_disk must be True; load from JSONL directory.")
    if not path.is_dir() or not (path / "train.jsonl").exists():
        raise FileNotFoundError(
            f"No train.jsonl in {dataset_path}. "
            "Build with: python -m scripts.build_midi_tokenized_jsonl <MidiDataset_path>",
        )

    data_files = {
        "train": str(path / "train.jsonl"),
    }
    if (path / "validation.jsonl").exists():
        data_files["validation"] = str(path / "validation.jsonl")
    if (path / "test.jsonl").exists():
        data_files["test"] = str(path / "test.jsonl")

    streaming = config.data.get("streaming", False)
    train_split = config.data.train_split
    eval_split = config.data.eval_split

    if streaming:
        train_dataset = load_dataset(
            "json",
            data_files=data_files,
            split=train_split,
            streaming=True,
        )
        buffer_size = config.data.get("streaming_buffer_size", 10_000)
        train_dataset = train_dataset.shuffle(
            seed=config.training.get("seed", 42),
            buffer_size=buffer_size,
        )
        if config.training.get("debug_mode", False):
            train_dataset = train_dataset.take(500)
        if eval_split in data_files:
            eval_dataset = load_dataset(
                "json",
                data_files=data_files,
                split=eval_split,
            )
        else:
            train_full = load_dataset(
                "json",
                data_files=data_files,
                split="train",
            )
            eval_dataset = train_full.select(
                range(min(100, len(train_full))),
            )
        eval_dataset.set_format(
            type="torch",
            columns=["input_ids"],
        )
    else:
        ds_dict = load_dataset(
            "json",
            data_files=data_files,
        )
        train_dataset = ds_dict[train_split]
        eval_dataset = ds_dict[eval_split]
        if config.training.get("debug_mode", False):
            train_dataset = train_dataset.select(
                range(min(500, len(train_dataset))),
            )
            eval_dataset = eval_dataset.select(
                range(min(100, len(eval_dataset))),
            )
        train_dataset.set_format(
            type="torch",
            columns=["input_ids"],
        )
        eval_dataset.set_format(
            type="torch",
            columns=["input_ids"],
        )

    data_collator = MidiDataCollatorForCausalLM(
        tokenizer=tokenizer,
        mlm=False,
        pad_token_id=config.tokenizer.get("pad_token_id", 0),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=not streaming,
        drop_last=config.training.drop_last,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    return train_dataloader, eval_dataloader


def create_model(
    cfg: DictConfig,
    vocab_size: int,
) -> PreTrainedModel:
    if cfg.model.base_model_type == "custom_transformer":
        model_config = AutoConfig.from_pretrained(
            cfg.model.config_name,
            vocab_size=vocab_size,
            n_positions=cfg.model.max_position_embeddings,
            n_ctx=cfg.model.max_position_embeddings,
            n_embd=cfg.model.hidden_size,
            n_layer=cfg.model.num_hidden_layers,
            n_head=cfg.model.num_attention_heads,
            resid_pdrop=cfg.model.hidden_dropout_prob,
            attn_pdrop=cfg.model.attention_probs_dropout_prob,
            bos_token_id=0,
            eos_token_id=1,
        )
        model = AutoModelForCausalLM.from_config(
            model_config,
        )
    elif cfg.model.base_model_type == "gemma":
        model_config = AutoConfig.from_pretrained(
            "google/gemma-2b",
            vocab_size=vocab_size,
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            num_attention_heads=cfg.model.num_attention_heads,
            intermediate_size=cfg.model.intermediate_size or cfg.model.hidden_size * 4,
            max_position_embeddings=cfg.model.max_position_embeddings,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        )
        model = AutoModelForCausalLM.from_config(
            model_config,
        )
    else:
        model_config = AutoConfig.from_pretrained(
            cfg.model.config_name,
            vocab_size=vocab_size,
            n_positions=cfg.model.max_position_embeddings,
            n_ctx=cfg.model.max_position_embeddings,
            n_embd=cfg.model.hidden_size,
            n_layer=cfg.model.num_hidden_layers,
            n_head=cfg.model.num_attention_heads,
        )
        model = AutoModelForCausalLM.from_config(
            model_config,
        )

    model_size = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model created with {model_size/1_000_000: .2f}M parameters",
    )
    return model


def train(
    cfg: DictConfig,
) -> None:
    set_seed(
        cfg.training.seed,
    )
    setup_logging(
        cfg.logging.level,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb" if cfg.logging.use_wandb else None,
    )
    logger.info(
        f"Distributed training: {accelerator.distributed_type}",
    )
    logger.info(
        f"Mixed precision: {accelerator.mixed_precision}",
    )

    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(
                cfg,
                resolve=True,
            ),
        )

    tokenizer = create_tokenizer(
        cfg,
    )
    model = create_model(
        cfg,
        len(tokenizer.vocab),
    )
    train_dataloader, eval_dataloader = load_and_prepare_dataset(
        cfg,
        tokenizer,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_epsilon,
        weight_decay=0.0,
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
    )

    streaming = cfg.data.get("streaming", False)
    if streaming:
        max_train_steps = cfg.training.get("max_steps")
        if max_train_steps is None or max_train_steps <= 0:
            raise ValueError(
                "When data.streaming=true, set training.max_steps to the desired number of optimization steps.",
            )
        num_update_steps_per_epoch = max_train_steps // cfg.training.num_train_epochs
    else:
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / cfg.training.gradient_accumulation_steps,
        )
        max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info("***** Running training *****")
    if streaming:
        logger.info("  Train examples = streaming (unbounded)")
    else:
        logger.info(
            f"  Num examples = {len(train_dataloader.dataset)}",
        )
    logger.info(
        f"  Num eval examples = {len(eval_dataloader.dataset)}",
    )
    logger.info(
        f"  Num Epochs = {cfg.training.num_train_epochs}",
    )
    logger.info(
        f"  Per device batch size = {cfg.training.per_device_train_batch_size}",
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}",
    )
    logger.info(
        f"  Total optimization steps = {max_train_steps}",
    )

    completed_steps = 0
    best_eval_loss = float("inf")

    os.makedirs(
        to_absolute_path(cfg.model.output_dir),
        exist_ok=True,
    )
    with open(
        os.path.join(
            to_absolute_path(cfg.model.output_dir),
            "train_config.json",
        ),
        "w",
    ) as f:
        json.dump(
            OmegaConf.to_container(
                cfg,
                resolve=True,
            ),
            f,
            indent=4,
        )

    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                train_loss += loss.detach().float()

                if cfg.training.max_gradient_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.max_gradient_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()

            if step % cfg.logging.logging_steps == 0 and step > 0:
                avg_loss = accelerator.gather(train_loss).mean().item() / cfg.logging.logging_steps
                train_loss = 0.0
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Loss: {avg_loss:.4e}",
                )
                if cfg.logging.use_wandb and accelerator.is_main_process:
                    if streaming:
                        train_epoch = completed_steps / max_train_steps * cfg.training.num_train_epochs
                    else:
                        train_epoch = epoch + step / len(train_dataloader)
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": train_epoch,
                            "train/step": completed_steps,
                        },
                    )

            if step % cfg.training.eval_steps == 0 and step > 0:
                model.eval()
                eval_loss = 0.0
                for eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                    eval_loss += outputs.loss.detach().float()
                eval_loss = accelerator.gather(eval_loss).mean().item() / len(eval_dataloader)
                perplexity = math.exp(eval_loss)
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Eval Loss: {eval_loss:.4e}, Perplexity: {perplexity:.4e}",
                )
                if cfg.logging.use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "eval/loss": eval_loss,
                            "eval/perplexity": perplexity,
                            "eval/step": completed_steps,
                        },
                    )
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if accelerator.is_main_process:
                        logger.info(
                            f"New best model with eval loss: {best_eval_loss:.4e}",
                        )
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        best_model_dir = os.path.join(
                            to_absolute_path(cfg.model.output_dir),
                            "best_eval_loss_model",
                        )
                        os.makedirs(
                            best_model_dir,
                            exist_ok=True,
                        )
                        unwrapped_model.save_pretrained(
                            best_model_dir,
                            save_function=accelerator.save,
                            is_main_process=accelerator.is_main_process,
                        )
                        with open(
                            os.path.join(
                                to_absolute_path(cfg.model.output_dir),
                                "tokenizer.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                tokenizer.to_dict(),
                                f,
                                indent=4,
                            )
                model.train()

            if accelerator.sync_gradients:
                completed_steps += 1
                lr_scheduler.step()
            if completed_steps >= max_train_steps:
                break

        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch} completed",
            )
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            epoch_output_dir = os.path.join(
                to_absolute_path(cfg.model.output_dir),
                f"epoch_{epoch}",
            )
            os.makedirs(
                epoch_output_dir,
                exist_ok=True,
            )
            unwrapped_model.save_pretrained(
                epoch_output_dir,
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process,
            )
            with open(
                os.path.join(
                    epoch_output_dir,
                    "tokenizer.json",
                ),
                "w",
            ) as f:
                json.dump(
                    tokenizer.to_dict(),
                    f,
                    indent=4,
                )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        final_output_dir = os.path.join(
            to_absolute_path(cfg.model.output_dir),
            "final",
        )
        os.makedirs(
            final_output_dir,
            exist_ok=True,
        )
        unwrapped_model.save_pretrained(
            final_output_dir,
            save_function=accelerator.save,
        )
        with open(
            os.path.join(
                final_output_dir,
                "tokenizer.json",
            ),
            "w",
        ) as f:
            json.dump(
                tokenizer.to_dict(),
                f,
                indent=4,
            )
        logger.info(
            f"Model saved to {final_output_dir}",
        )
    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


@hydra.main(
    config_path="../configs",
    config_name="next_token_prediction",
    version_base=None,
)
def main(
    config: DictConfig,
) -> None:
    print(
        OmegaConf.to_yaml(config),
    )
    train(
        config,
    )


if __name__ == "__main__":
    main()
