"""
Note embedder training: loads a pretrained next-token-prediction decoder.
ECHO: repeat input_ids n times, forward, take token encodings from the last repetition, pool to 1D embedding.
Training: pairs (notes_first, notes_second); embed both with ECHO; loss = distance between embeddings (minimize).
Saves checkpoints in HuggingFace style.
"""

import os
import json
import math
import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from transformers import AutoModelForCausalLM, get_scheduler

import wandb
from training.embedder.pairwise_collator import pairwise_collate
from training.embedder.echo_dataset import pairwise_dataset_from_hf, load_embedding_pairs_dataset

load_dotenv()


logger = logging.getLogger(
    __name__,
)


def setup_logging(
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            log_level,
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_tokenizer_from_pretrained(
    pretrained_path: str,
) -> ExponentialTimeTokenizer:
    tokenizer_path = (
        Path(
            pretrained_path,
        )
        / "tokenizer.json"
    )
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json missing at {pretrained_path}. Train next-token-prediction first.",
        )
    with open(
        tokenizer_path,
        "r",
    ) as f:
        data = json.load(
            f,
        )
    if "parameters" in data and "tokenizer_config" in data["parameters"]:
        tokenizer_config = data["parameters"]["tokenizer_config"]
    else:
        tokenizer_config = data
    return ExponentialTimeTokenizer.build_tokenizer(
        tokenizer_config,
    )


def mean_pool_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """(batch, seq, hidden) -> (batch, hidden) using attention_mask."""
    mask = attention_mask.unsqueeze(
        -1,
    ).float()
    sum_hidden = (last_hidden_state * mask).sum(
        dim=1,
    )
    sum_mask = mask.sum(
        dim=1,
    ).clamp(
        min=1e-9,
    )
    return sum_hidden / sum_mask


def embed_echo(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    echo_repetitions: int,
) -> torch.Tensor:
    """ECHO: input repeated echo_repetitions times. Forward, last-rep token encodings, mean-pool -> (batch, hidden).
    Padding is excluded via attention_mask in mean_pool_hidden.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    last_hidden = outputs.hidden_states[-1]
    seq_len = input_ids.size(1)
    segment_len = seq_len // echo_repetitions
    if segment_len == 0:
        segment_len = seq_len
    last_rep_hidden = last_hidden[:, -segment_len:, :]
    last_rep_mask = attention_mask[:, -segment_len:]
    return mean_pool_hidden(last_rep_hidden, last_rep_mask)


def get_model_max_length(
    model: torch.nn.Module,
) -> int | None:
    """Return model's maximum input length (max_position_embeddings or model_max_length)."""
    config = getattr(
        model,
        "config",
        None,
    )
    if config is None:
        return None
    return getattr(
        config,
        "max_position_embeddings",
        None,
    ) or getattr(
        config,
        "model_max_length",
        None,
    )


def load_and_prepare_pairwise_datasets(
    cfg: DictConfig,
    tokenizer: ExponentialTimeTokenizer,
    echo_repetitions: int,
    model_max_length: int | None = None,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """Build train and eval dataloaders. Returns (train_dl, eval_dl, train_truncation_stats, eval_truncation_stats)."""
    dataset_path = to_absolute_path(
        cfg.data.dataset_path,
    )
    max_seq_length = cfg.data.get(
        "max_seq_length",
        512,
    )
    if model_max_length is not None and max_seq_length > model_max_length:
        max_seq_length = model_max_length
    num_proc = cfg.data.get(
        "num_proc",
        8,
    )
    train_hf, eval_hf = load_embedding_pairs_dataset(
        dataset_path,
        cfg.data.train_split,
        cfg.data.eval_split,
    )
    if cfg.training.get(
        "debug_mode",
        False,
    ):
        train_hf = train_hf.select(
            range(min(500, len(train_hf))),
        )
        eval_hf = eval_hf.select(
            range(min(100, len(eval_hf))),
        )
    train_ds, train_trunc = pairwise_dataset_from_hf(
        train_hf,
        tokenizer,
        max_seq_length,
        echo_repetitions=echo_repetitions,
        num_proc=num_proc,
    )
    eval_ds, eval_trunc = pairwise_dataset_from_hf(
        eval_hf,
        tokenizer,
        max_seq_length,
        echo_repetitions=echo_repetitions,
        num_proc=num_proc,
    )
    pad_token_id = (
        getattr(
            tokenizer,
            "pad_token_id",
            None,
        )
        or 0
    )

    def collate_fn(
        examples,
    ):
        return pairwise_collate(
            examples,
            pad_token_id=pad_token_id,
        )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_device_train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=cfg.training.drop_last,
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=cfg.training.per_device_eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    return train_dataloader, eval_dataloader, train_trunc, eval_trunc


def train(
    cfg: DictConfig,
) -> None:
    set_seed(
        cfg.training.seed,
    )
    setup_logging(
        cfg.logging.level,
    )

    pretrained_path = cfg.get(
        "pretrained_model_path",
    )
    if not pretrained_path:
        raise ValueError(
            "embedder training requires pretrained_model_path (next-token-prediction checkpoint).",
        )
    pretrained_path = to_absolute_path(
        pretrained_path,
    )
    if not Path(
        pretrained_path,
    ).is_dir():
        raise FileNotFoundError(
            f"Pretrained model path not found: {pretrained_path}",
        )

    echo_repetitions = cfg.embedder.get(
        "echo_repetitions",
        1,
    )
    if echo_repetitions < 1:
        raise ValueError(
            "embedder.echo_repetitions must be >= 1.",
        )
    logger.info(
        f"Embedder: ECHO (repetitions={echo_repetitions}), pairwise distance loss",
    )

    tokenizer = load_tokenizer_from_pretrained(
        pretrained_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
    )

    model_max_length = get_model_max_length(
        model,
    )
    data_max_seq_length = cfg.data.get(
        "max_seq_length",
        512,
    )
    if model_max_length is not None and data_max_seq_length > model_max_length:
        logger.warning(
            "data.max_seq_length (%s) exceeds model max position length (%s); "
            "sequences will be truncated to model max.",
            data_max_seq_length,
            model_max_length,
        )

    train_dataloader, eval_dataloader, train_trunc, eval_trunc = load_and_prepare_pairwise_datasets(
        cfg,
        tokenizer,
        echo_repetitions,
        model_max_length=model_max_length,
    )

    def log_truncation_warning(
        name: str,
        stats: dict,
    ) -> None:
        n_first = stats.get(
            "num_truncated_first",
            0,
        )
        n_second = stats.get(
            "num_truncated_second",
            0,
        )
        total = stats.get(
            "total",
            0,
        )
        if total and (n_first > 0 or n_second > 0):
            logger.warning(
                "%s: %s examples had notes_first truncated, %s had notes_second truncated (total: %s)",
                name,
                n_first,
                n_second,
                total,
            )

    log_truncation_warning(
        "Train",
        train_trunc,
    )
    log_truncation_warning(
        "Eval",
        eval_trunc,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb" if cfg.logging.use_wandb else None,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(
            cfg.training.adam_beta1,
            cfg.training.adam_beta2,
        ),
        eps=cfg.training.adam_epsilon,
        weight_decay=0.0,
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
    )

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

    output_dir = to_absolute_path(
        cfg.model.output_dir,
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    train_config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
    )

    logger.info("***** Running embedder training *****")
    logger.info(
        f"  Num train examples = {len(train_dataloader.dataset)}",
    )
    logger.info(
        f"  Num eval examples = {len(eval_dataloader.dataset)}",
    )
    logger.info(
        f"  Echo repetitions = {echo_repetitions}",
    )
    logger.info(
        f"  Num Epochs = {cfg.training.num_train_epochs}",
    )
    logger.info(
        f"  Total optimization steps = {max_train_steps}",
    )

    completed_steps = 0
    best_eval_loss = float(
        "inf",
    )

    for epoch in range(
        cfg.training.num_train_epochs,
    ):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(
            train_dataloader,
        ):
            with accelerator.accumulate(
                model,
            ):
                e1 = embed_echo(
                    model,
                    batch["input_ids_first"],
                    batch["attention_mask_first"],
                    echo_repetitions,
                )
                e2 = embed_echo(
                    model,
                    batch["input_ids_second"],
                    batch["attention_mask_second"],
                    echo_repetitions,
                )
                loss = F.mse_loss(
                    e1,
                    e2,
                )
                accelerator.backward(
                    loss,
                )
                train_loss += loss.detach().float()

                if cfg.training.max_gradient_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.max_gradient_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()

            if step % cfg.logging.logging_steps == 0 and step > 0:
                avg_loss = (
                    accelerator.gather(
                        train_loss,
                    )
                    .mean()
                    .item()
                    / cfg.logging.logging_steps
                )
                train_loss = 0.0
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Loss: {avg_loss:.4e}",
                )
                if cfg.logging.use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch
                            + step
                            / len(
                                train_dataloader,
                            ),
                            "train/step": completed_steps,
                        },
                    )

            if step % cfg.training.eval_steps == 0 and step > 0:
                model.eval()
                eval_loss = 0.0
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        e1 = embed_echo(
                            model,
                            eval_batch["input_ids_first"],
                            eval_batch["attention_mask_first"],
                            echo_repetitions,
                        )
                        e2 = embed_echo(
                            model,
                            eval_batch["input_ids_second"],
                            eval_batch["attention_mask_second"],
                            echo_repetitions,
                        )
                        eval_loss += (
                            F.mse_loss(
                                e1,
                                e2,
                            )
                            .detach()
                            .float()
                        )
                eval_loss = accelerator.gather(
                    eval_loss,
                ).mean().item() / len(
                    eval_dataloader,
                )
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Eval Loss: {eval_loss:.4e}",
                )
                if cfg.logging.use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "eval/loss": eval_loss,
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
                        unwrapped = accelerator.unwrap_model(
                            model,
                        )
                        best_dir = os.path.join(
                            output_dir,
                            "best_eval_loss_model",
                        )
                        os.makedirs(
                            best_dir,
                            exist_ok=True,
                        )
                        unwrapped.save_pretrained(
                            best_dir,
                            save_function=accelerator.save,
                            is_main_process=accelerator.is_main_process,
                        )
                        with open(
                            os.path.join(
                                best_dir,
                                "tokenizer.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                tokenizer.to_dict(),
                                f,
                                indent=4,
                            )
                        with open(
                            os.path.join(
                                best_dir,
                                "train_config.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                train_config_dict,
                                f,
                                indent=4,
                            )
                        with open(
                            os.path.join(
                                best_dir,
                                "embedding_config.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                {"echo_repetitions": echo_repetitions},
                                f,
                                indent=2,
                            )
                model.train()

            if accelerator.sync_gradients:
                completed_steps += 1
                lr_scheduler.step()

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(
        model,
    )
    if accelerator.is_main_process:
        final_dir = os.path.join(
            output_dir,
            "final",
        )
        os.makedirs(
            final_dir,
            exist_ok=True,
        )
        unwrapped.save_pretrained(
            final_dir,
            save_function=accelerator.save,
        )
        with open(
            os.path.join(
                final_dir,
                "tokenizer.json",
            ),
            "w",
        ) as f:
            json.dump(
                tokenizer.to_dict(),
                f,
                indent=4,
            )
        with open(
            os.path.join(
                final_dir,
                "train_config.json",
            ),
            "w",
        ) as f:
            json.dump(
                train_config_dict,
                f,
                indent=4,
            )
        with open(
            os.path.join(
                final_dir,
                "embedding_config.json",
            ),
            "w",
        ) as f:
            json.dump(
                {"echo_repetitions": echo_repetitions},
                f,
                indent=2,
            )
        logger.info(
            f"Model saved to {final_dir}",
        )
    if cfg.logging.use_wandb and accelerator.is_main_process:
        wandb.finish()


@hydra.main(
    config_path="../configs",
    config_name="embedder",
    version_base=None,
)
def main(
    config: DictConfig,
) -> None:
    print(
        OmegaConf.to_yaml(
            config,
        ),
    )
    train(
        config,
    )


if __name__ == "__main__":
    main()
