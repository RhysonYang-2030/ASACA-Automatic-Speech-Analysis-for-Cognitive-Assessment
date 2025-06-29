#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import shutil
import time
import psutil  # 用于监控内存使用情况
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Union
import sympy
from scipy.signal import medfilt  # 使用 SciPy 的 medfilt

import ctc_segmentation as cs
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,  # 引入 EarlyStoppingCallback
)

from jiwer import wer as jiwer_wer

import syllapy

os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 指定调试输出文件
DEBUG_LOG_FILE = "debug_output_train.txt"

def debug_print(msg: str):
    print(msg)
    try:
        with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"Error writing to debug log: {e}")

def count_syllables(word: str) -> int:
    return syllapy.count(word)
debug_print("Using syllapy for syllable count.")

# -------------------- 性能监控辅助函数 --------------------
def log_memory_usage(stage: str):
    try:
        mem = psutil.virtual_memory()
        debug_print(f"{stage} - Memory usage: {mem.percent}%")
    except Exception as e:
        debug_print(f"psutil error in {stage}: {e}")

def log_time(start, stage: str):
    elapsed = time.perf_counter() - start
    debug_print(f"{stage} completed in {elapsed:.2f} seconds")
    return elapsed

# -------------------- 音频加载与重采样 --------------------
import soundfile as sf
import resampy

def load_audio_file(audio_path: str, sr: int = 16000):
    try:
        speech, original_sr = sf.read(audio_path)
        # 如果是多通道，则转换为单通道
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        # 如果采样率不匹配，则进行重采样
        if original_sr != sr:
            try:
                speech = resampy.resample(speech, original_sr, sr)
                debug_print(f"Resampled audio from {original_sr} Hz to {sr} Hz: {audio_path}")
            except Exception as resex:
                debug_print(f"Error during resampling: {resex}")
                return None
        debug_print(f"Loaded audio file: {audio_path}")
        return speech.astype(np.float32)
    except Exception as e:
        debug_print(f"EXCEPTION loading {audio_path}: {e}")
        return None

def compute_audio_duration(speech, sr):
    return len(speech) / sr

def safe_index_bounds(index, max_val):
    return max(0, min(index, max_val - 1))

# -------------------- SpecAugment: 时间遮罩 --------------------
def apply_time_mask(audio: np.ndarray, mask_percentage: float = 0.05) -> np.ndarray:
    """随机将一段连续样本置零，mask_percentage 表示最多遮罩的比例"""
    T = len(audio)
    mask_length = int(T * mask_percentage)
    if mask_length < 1:
        return audio
    start = np.random.randint(0, T - mask_length)
    audio[start:start+mask_length] = 0
    debug_print(f"Applied time mask: start={start}, mask_length={mask_length}")
    return audio

# -------------------- 数据预处理 --------------------
def prepare_dataset(batch, sr: int, processor: Wav2Vec2Processor, apply_specaugment: bool = False, mask_percentage: float = 0.05):
    start_time = time.perf_counter()
    audio_path = batch.get("audio_path")
    speech = load_audio_file(audio_path, sr=sr)
    if speech is None:
        return {}
    if apply_specaugment:
        speech = apply_time_mask(speech, mask_percentage=mask_percentage)
    log_memory_usage(f"After loading {audio_path}")
    try:
        audio_features = processor.feature_extractor(
            speech, sampling_rate=sr, return_tensors="pt", padding=True
        )
        batch["input_values"] = audio_features["input_values"][0]
        # 文本预处理：文本已在构建数据集时做了 strip/lower/空白替换
        text_encoded = processor(text=batch["text"], return_tensors="pt").input_ids[0]
        batch["labels"] = text_encoded
        debug_print(f"Processed dataset for: {audio_path}")
    except Exception as e:
        debug_print(f"EXCEPTION in prepare_dataset for {audio_path}: {e}")
        return {}
    log_time(start_time, f"prepare_dataset for {audio_path}")
    return batch

# -------------------- DataCollator --------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features if "input_values" in f]
        label_features = [{"input_ids": f["labels"]} for f in features if "labels" in f]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", padding=self.padding
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", padding=self.padding
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        debug_print("Data collator processed a batch.")
        return batch

# -------------------- Metrics --------------------
# -------------------- Metrics --------------------
def get_compute_metrics(processor):
    """
    使用 jiwer 计算 WER，完全本地计算，无需网络。
    """
    def compute_metrics(pred):
        # 1) 预测转文字
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.tokenizer.batch_decode(pred_ids,
                                                   skip_special_tokens=True)

        # 2) 标签转文字（去掉 -100）
        labels = pred.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(labels,
                                                     skip_special_tokens=True)
        # 避免空字符串导致 jiwer 抛错
        label_str = [ref if ref.strip() else " " for ref in label_str]

        # 3) 计算 WER（先参考，后预测的顺序）
        wer = jiwer_wer(label_str, pred_str)
        return {"wer": wer}
    return compute_metrics


# -------------------- 新的 ModelCheckpointManager 类 --------------------
class ModelCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_model_config(self, model, filename="model_config.json"):
        config_path = os.path.join(self.save_dir, filename)
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)
        debug_print(f"Saved model config to {config_path}")

    def save_model_weights(self, model, filename="model.safetensors"):
        try:
            from safetensors.torch import save_file
            model_state = model.state_dict()
            save_file(model_state, os.path.join(self.save_dir, filename))
            debug_print(f"Saved model weights to {os.path.join(self.save_dir, filename)} using safetensors.")
        except ImportError:
            torch.save(model.state_dict(), os.path.join(self.save_dir, "model.pt"))
            debug_print(f"Saved model weights to {os.path.join(self.save_dir, 'model.pt')} using torch.save.")

    def save_processor(self, processor):
        # 保存 processor 完整信息（包括词表、配置等）
        processor.save_pretrained(self.save_dir)
        debug_print(f"Saved processor to {self.save_dir}")

    def save_optimizer_state(self, optimizer, filename="optimizer.pt"):
        torch.save(optimizer.state_dict(), os.path.join(self.save_dir, filename))
        debug_print(f"Saved optimizer state to {os.path.join(self.save_dir, filename)}")

    def save_scheduler_state(self, scheduler, filename="scheduler.pt"):
        torch.save(scheduler.state_dict(), os.path.join(self.save_dir, filename))
        debug_print(f"Saved scheduler state to {os.path.join(self.save_dir, filename)}")

    def save_trainer_state(self, trainer_state, filename="trainer_state.json"):
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(trainer_state, f, indent=2, default=str)
        debug_print(f"Saved trainer state to {os.path.join(self.save_dir, filename)}")

    def save_training_args(self, training_args, filename="training_args.bin"):
        torch.save(training_args, os.path.join(self.save_dir, filename))
        debug_print(f"Saved training arguments to {os.path.join(self.save_dir, filename)}")

    def save_rng_state(self, filename="rng_state.pth"):
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        torch.save(rng_state, os.path.join(self.save_dir, filename))
        debug_print(f"Saved RNG state to {os.path.join(self.save_dir, filename)}")

    def save_complete_checkpoint(self, model, processor, optimizer, scheduler, trainer_state, training_args):
        self.save_model_config(model)
        self.save_model_weights(model)
        self.save_processor(processor)
        if optimizer is not None:
            self.save_optimizer_state(optimizer)
        if scheduler is not None:
            self.save_scheduler_state(scheduler)
        self.save_trainer_state(trainer_state)
        self.save_training_args(training_args)
        self.save_rng_state()
        debug_print(f"Complete checkpoint saved to {self.save_dir}")

# -------------------- 自定义 Callback --------------------
class SaveBestModelAndProcessorCallback(TrainerCallback):
    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, best_model_dir: str):
        self.model = model
        self.processor = processor
        self.best_model_dir = best_model_dir
        self.best_checkpoint = None
        self.best_metric = None
        self.checkpoint_manager = ModelCheckpointManager(best_model_dir)

    def on_save(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        current_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        # 这里假设 state.best_metric 为当前 metric 值，根据实际情况可能需要调整
        current_metric = state.best_metric if hasattr(state, "best_metric") else None

        if current_metric is not None and (self.best_metric is None or current_metric < self.best_metric):
            self.best_metric = current_metric
            self.best_checkpoint = current_checkpoint
            # 调用统一的保存函数保存所有内容
            self.checkpoint_manager.save_complete_checkpoint(
                model=self.model,
                processor=self.processor,
                optimizer=trainer.optimizer if trainer is not None else None,
                scheduler=trainer.lr_scheduler if trainer is not None else None,
                trainer_state=state.__dict__,
                training_args=args
            )
            debug_print(f"New best model found at checkpoint: {current_checkpoint}")
            debug_print(f"Saved best model and associated files to: {self.best_model_dir}")
        return control

# -------------------- 主函数（训练流程） --------------------
def main(args):
    try:
        df = pd.read_excel(args.transcription_excel)
        debug_print("Excel file loaded successfully.")
    except Exception as e:
        debug_print(f"EXCEPTION reading Excel file {args.transcription_excel}: {e}")
        return
    data_list = []
    for i, row in df.iterrows():
        data_id = str(row["DataID"]).strip()
        transcription = re.sub(r'\s+', ' ', str(row["Transcription"]).strip().lower())
        label = int(row["Label"])
        audio_path = os.path.join(args.raw_audio_folder, data_id)
        if not os.path.isfile(audio_path):
            debug_print(f"Audio file not found: {audio_path}")
            continue
        data_list.append({
            "audio_path": audio_path,
            "text": transcription,
            "label": label
        })
    dataset = Dataset.from_list(data_list)
    tmp = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = tmp["train"]
    test_dataset = tmp["test"]
    tmp2 = train_val.train_test_split(test_size=0.125, seed=42)
    train_dataset = tmp2["train"]
    eval_dataset = tmp2["test"]
    debug_print(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")

    processor = Wav2Vec2Processor.from_pretrained(
        args.pretrained_processor,
        local_files_only=True
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_model,
        ctc_loss_reduction="mean",    # 使用平均CTC Loss
        ctc_zero_infinity=True,       # 开启 zero_infinity 防止梯度爆炸
        pad_token_id=processor.tokenizer.pad_token_id,
        local_files_only=True
    )
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # torchrun 会写入
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    debug_print(f"Using device: {device}")
    model.to(device)
    model.freeze_feature_encoder()

    debug_print("Starting dataset preprocessing...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, sr=args.sr, processor=processor, apply_specaugment=args.apply_specaugment, mask_percentage=args.mask_percentage),
        num_proc=args.num_proc
    )
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, sr=args.sr, processor=processor, apply_specaugment=False),
        num_proc=args.num_proc
    )
    test_dataset = test_dataset.map(
        lambda batch: prepare_dataset(batch, sr=args.sr, processor=processor, apply_specaugment=False),
        num_proc=args.num_proc
    )
    debug_print("Dataset preprocessing completed.")

    train_steps = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    warmup_steps = int(train_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed_config if args.deepspeed_config != "" else None,
        weight_decay=args.weight_decay,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False
    )

    data_collator = DataCollatorCTCWithPadding(processor)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6),
                   SaveBestModelAndProcessorCallback(model, processor, args.best_model_dir)]
    )

    debug_print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        debug_print(f"EXCEPTION during training: {e}")
    processor.save_pretrained(args.output_dir)
    debug_print("Training completed and processor saved.")
    print("Training completed.")
    print(f"Model and processor saved to: {args.output_dir}")
    print(f"Best model and associated files saved to: {args.best_model_dir}")

# -------------------- Colab模式下参数配置 --------------------
class ColabArgs:
    transcription_excel = "Trans.xlsx"
    raw_audio_folder = "Pretrain"
    pretrained_processor = "/home/users/xyang2/wav2vec2_AD/wav2vec2_xlsr53"
    pretrained_model = "/home/users/xyang2/wav2vec2_AD/wav2vec2_xlsr53"
    output_dir = "out"
    best_model_dir = "best"
    num_train_epochs = 100
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    learning_rate = 5e-5
    warmup_ratio = 0.05
    save_total_limit = 6
    weight_decay = 0.005
    sr = 16000
    fp16 = True
    gradient_checkpointing = True
    deepspeed_config = ""  # 如果使用 DeepSpeed，请提供配置文件路径
    num_proc = 8
    apply_specaugment = True
    mask_percentage = 0.05

if __name__ == "__main__":
    args = ColabArgs()
    main(args)
