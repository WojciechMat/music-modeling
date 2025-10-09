import os
import json
import math
import logging
from typing import Any, Dict, List, Tuple

import hydra
import torch
import wandb
from dotenv import load_dotenv
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from accelerate.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from transformers import AutoConfig, PreTrainedModel, AutoModelForCausalLM, get_scheduler

from datasets import load_dataset
from training.midi_data_collator import MidiDataCollatorForCausalLM

load_dotenv()


logger = get_logger(__name__)


def setup_logging(
    log_level: str = "INFO",
) -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_tokenizer(
    config: DictConfig,
) -> ExponentialTimeTokenizer:
    """Create and configure the MIDI tokenizer based on config."""
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
    """Load and prepare the dataset for training and evaluation."""

    dataset_config = {
        "context_length": config.data.max_seq_length,
        "min_time_unit": config.tokenizer.min_time_unit,
        "max_time_step": config.tokenizer.max_time_step,
        "n_velocity_bins": config.tokenizer.n_velocity_bins,
        "n_special_ids": config.tokenizer.n_special_ids,
        "sliding_window_stride": config.data.get("sliding_window_stride", None),
        "aggregated_dataset_path": to_absolute_path(config.data.get("source_dataset_path", "./datasets/MidiDataset")),
    }

    # Load the dataset
    if config.data.load_from_disk:
        train_dataset = load_dataset(
            to_absolute_path(config.data.dataset_path),
            split=config.data.train_split,
            trust_remote_code=True,
            **dataset_config,
        )
        eval_dataset = load_dataset(
            to_absolute_path(config.data.dataset_path),
            split=config.data.eval_split,
            trust_remote_code=True,
            **dataset_config,
        )
    else:
        # TODO: Implement other dataset loading methods if needed
        pass

    # For subset testing during development
    if config.training.debug_mode:
        train_dataset = train_dataset.select(
            range(min(500, len(train_dataset))),
        )
        eval_dataset = eval_dataset.select(
            range(min(100, len(eval_dataset))),
        )

    # Set the format to PyTorch tensors
    train_dataset.set_format(
        type="torch",
        columns=["input_ids"],
    )
    eval_dataset.set_format(
        type="torch",
        columns=["input_ids"],
    )

    # Create dataloaders
    data_collator = MidiDataCollatorForCausalLM(
        tokenizer=tokenizer,
        mlm=False,
        pad_token_id=config.tokenizer.get("pad_token_id", 0),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
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
    config: DictConfig,
    vocab_size: int,
) -> PreTrainedModel:
    """Create a model based on the specified configuration."""

    if config.model.base_model_type == "custom_transformer":
        # Create a custom transformer configuration
        model_config = AutoConfig.from_pretrained(
            config.model.config_name or "gpt2",
            vocab_size=vocab_size,
            n_positions=config.model.max_position_embeddings,
            n_ctx=config.model.max_position_embeddings,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_hidden_layers,
            n_head=config.model.num_attention_heads,
            resid_pdrop=config.model.hidden_dropout_prob,
            attn_pdrop=config.model.attention_probs_dropout_prob,
            bos_token_id=0,  # Adjust if needed
            eos_token_id=1,  # Adjust if needed
        )

        # Initialize a new model
        model = AutoModelForCausalLM.from_config(
            model_config,
        )

    elif config.model.base_model_type == "gemma":
        # Load Gemma configuration but with custom parameters
        model_config = AutoConfig.from_pretrained(
            "google/gemma-2b",
            vocab_size=vocab_size,
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            intermediate_size=config.model.intermediate_size or config.model.hidden_size * 4,
            max_position_embeddings=config.model.max_position_embeddings,
            hidden_dropout_prob=config.model.hidden_dropout_prob,
            attention_probs_dropout_prob=config.model.attention_probs_dropout_prob,
        )

        # Initialize a new model
        model = AutoModelForCausalLM.from_config(
            model_config,
        )

    else:
        # Load any other model architecture
        model_config = AutoConfig.from_pretrained(
            config.model.config_name,
            vocab_size=vocab_size,
            n_positions=config.model.max_position_embeddings,
            n_ctx=config.model.max_position_embeddings,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_hidden_layers,
            n_head=config.model.num_attention_heads,
        )

        model = AutoModelForCausalLM.from_config(
            model_config,
        )

    # Log model size
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {model_size/1_000_000: .2f}M parameters")

    return model


def get_grouped_params(
    model: PreTrainedModel,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Get parameters grouped for optimization with different weight decay values.
    """
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


def train(
    config: DictConfig,
) -> None:
    """Main training function."""

    # Setup
    set_seed(
        config.training.seed,
    )
    setup_logging(
        config.logging.level,
    )

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if config.logging.use_wandb else None,
    )

    logger.info(f"Distributed training: {accelerator.distributed_type}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")

    # Initialize W&B if requested
    if config.logging.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    tokenizer = create_tokenizer(
        config,
    )

    model = create_model(
        config,
        len(tokenizer.vocab),
    )

    train_dataloader, eval_dataloader = load_and_prepare_dataset(
        config,
        tokenizer,
    )

    optimizer = torch.optim.AdamW(
        get_grouped_params(model, config.training.weight_decay),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
    )

    # Prepare everything with accelerator for training
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
    )

    # Initialize learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = config.training.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(f"  Per device batch size = {config.training.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    best_eval_loss = float("inf")

    # Training loop
    for epoch in range(config.training.num_train_epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # Log training metrics
                train_loss += loss.detach().float()

                # Clip gradients
                if config.training.max_gradient_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.training.max_gradient_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Log and evaluate
            if step % config.logging.logging_steps == 0 and step > 0:
                avg_loss = accelerator.gather(train_loss).mean().item() / config.logging.logging_steps
                train_loss = 0.0

                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {avg_loss: .4f}")

                if config.logging.use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(train_dataloader),
                            "train/step": completed_steps,
                        },
                    )

            # Evaluate
            if step % config.training.eval_steps == 0 and step > 0:
                model.eval()
                eval_loss = 0.0

                for eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**eval_batch)

                    eval_loss += outputs.loss.detach().float()

                eval_loss = accelerator.gather(eval_loss).mean().item() / len(eval_dataloader)
                perplexity = math.exp(eval_loss)

                logger.info(f"Epoch: {epoch}, Step: {step}, Eval Loss: {eval_loss: .4f}, Perplexity: {perplexity: .2f}")

                if config.logging.use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "eval/loss": eval_loss,
                            "eval/perplexity": perplexity,
                            "eval/step": completed_steps,
                        },
                    )

                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if accelerator.is_main_process:
                        logger.info(f"New best model with eval loss: {best_eval_loss: .4f}")

                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)

                        os.makedirs(config.model.output_dir, exist_ok=True)
                        unwrapped_model.save_pretrained(
                            config.model.output_dir,
                            save_function=accelerator.save,
                            is_main_process=accelerator.is_main_process,
                        )
                        with open(os.path.join(config.model.output_dir, "tokenizer.json"), "w") as f:
                            json.dump(
                                tokenizer.to_dict(),
                                f,
                                indent=4,
                            )

                model.train()

            completed_steps += 1

            # Stop if we reach max steps
            if completed_steps >= max_train_steps:
                break

        # Save at end of epoch
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} completed")

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            epoch_output_dir = os.path.join(config.model.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            unwrapped_model.save_pretrained(
                epoch_output_dir,
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process,
            )

            with open(os.path.join(epoch_output_dir, "tokenizer.json"), "w") as f:
                json.dump(
                    tokenizer.to_dict(),
                    f,
                    indent=4,
                )

    # Finish training and save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        final_output_dir = os.path.join(config.model.output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        unwrapped_model.save_pretrained(
            final_output_dir,
            save_function=accelerator.save,
        )

        with open(os.path.join(final_output_dir, "tokenizer.json"), "w") as f:
            json.dump(
                tokenizer.to_dict(),
                f,
                indent=4,
            )

        logger.info(f"Model saved to {final_output_dir}")

    if config.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


@hydra.main(config_path="configs", config_name="midi_transformer")
def main(
    config: DictConfig,
) -> None:
    """Main entry point for the training script."""
    print(OmegaConf.to_yaml(config))

    train(
        config,
    )


if __name__ == "__main__":
    main()
