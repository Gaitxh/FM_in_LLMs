# -*- coding: utf-8 -*-
"""
Experiment 1: Reconstruction-matched scaling sanity check

Built from the user's original SAE training code.
What this script adds:
1) final reconstruction metrics on the test split:
   - explained variance (EV)
   - normalized MSE (NMSE)
   - Pearson correlation
2) metrics.json / metrics.csv saving per run
3) summary mode for sweeping over trained runs and automatically finding
   EV-matched configurations across model scales.

Notes:
- This script only implements the "reconstruction-matching" stage of Exp1.
- Recomputing downstream active-module statistics (e.g. Figure 12-style analyses)
  still requires the user's separate analysis code, which was not provided here.
"""

import os
import glob
import csv
import json
import math
import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import GPTNeoXModel, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset


# -----------------------------
# DDP helpers
# -----------------------------
def ddp_init_if_needed(is_ddp: bool, local_rank: int, world_size: int, master_addr: str, master_port: str):
    if not is_ddp:
        return
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=int(os.environ["RANK"])
    )
    torch.cuda.set_device(local_rank)


def ddp_cleanup(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def is_rank0(is_ddp: bool, rank: int) -> bool:
    return (not is_ddp) or (rank == 0)


def allreduce_mean_2floats(is_ddp: bool, device: torch.device, a: float, b: float) -> Tuple[float, float]:
    if not is_ddp:
        return a, b
    t = torch.tensor([a, b], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t[0].item()), float(t[1].item())


def allreduce_sum_tensor(is_ddp: bool, t: torch.Tensor) -> torch.Tensor:
    if is_ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# -----------------------------
# Dataset / Model lazy loading (per-process)
# -----------------------------
class OnlineEmbedder:
    """
    每个进程初始化一次：dataset + tokenizer + config
    模型常驻 GPU：进程内只加载一次 GPTNeoXModel，后续反复复用
    输出 token 级 embedding 拼接 [sum_tokens, d_activation]，强制 float32
    """
    def __init__(self, para: dict):
        self.para = para
        self.device = para["device"]
        self.max_length = int(para.get("max_length", 2048))

        self.model_dtype = torch.float32

        # ---- load dataset once ----
        self.ds = self._load_text_dataset(para)

        # ---- model/tokenizer paths ----
        self.model_path = os.path.join(para["model_dir"], para["model_name"])
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"模型目录不存在：{self.model_path}")

        # tokenizer 常驻 CPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # config 常驻 CPU（不加载权重）
        cfg = AutoConfig.from_pretrained(self.model_path)
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_hidden_layers)
        self.d_activation = self.hidden_size * self.num_layers

        # ---- LM 常驻（每个进程/每张卡只加载一次）----
        self.lm = None
        self._ensure_lm_loaded()

    def _load_text_dataset(self, para: dict):
        if para["dataset"] == "pile-val-backup":
            path = os.path.join(para["data_dir"], "pile-val-backup", "val.jsonl.zst")
            if not os.path.exists(path):
                raise FileNotFoundError(f"未找到：{path}")
            return load_dataset("json", data_files=path, split="train")

        if para["dataset"] == "pile-10k":
            parquet_dir = os.path.join(para["data_dir"], "pile-10k")
            fixed = os.path.join(parquet_dir, "train-00000-of-00001-4746b8785c874cc7.parquet")
            if os.path.exists(fixed):
                return Dataset.from_parquet(fixed)

            files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
            if len(files) == 0:
                raise FileNotFoundError(f"未找到 parquet：{parquet_dir}/*.parquet")
            return Dataset.from_parquet(files[0])

        raise ValueError(f"Unknown dataset: {para['dataset']}")

    @staticmethod
    def torch_zscore_fp32(x_fp32: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
        mean = x_fp32.mean(dim=dim, keepdim=True)
        std = x_fp32.std(dim=dim, keepdim=True, unbiased=False)
        z = (x_fp32 - mean) / (std + eps)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _ensure_lm_loaded(self) -> GPTNeoXModel:
        if self.lm is None:
            lm = GPTNeoXModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            lm.eval()
            for p in lm.parameters():
                p.requires_grad_(False)
            self.lm = lm
        return self.lm

    def unload_lm(self):
        self.lm = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def embed_from_indices(self, selec_x: torch.Tensor) -> torch.Tensor:
        lm = self._ensure_lm_loaded()

        idx_list = selec_x.detach().cpu().tolist()
        texts = []
        for idx in idx_list:
            ex = self.ds[int(idx)]
            txt = ex.get("text", "")
            texts.append(txt.replace("\n", " "))

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        x = torch.cat(
            lm(**inputs, output_hidden_states=True, use_cache=False).hidden_states[1:],
            dim=-1
        )

        attn = inputs["attention_mask"].bool()
        batch_data = []
        for b in range(x.size(0)):
            xb = x[b, attn[b]]
            xb_fp32 = self.torch_zscore_fp32(xb.float(), dim=0)
            batch_data.append(xb_fp32.to(torch.float32))

        out = torch.cat(batch_data, dim=0)
        return out


def get_embedder_cached(para: dict) -> OnlineEmbedder:
    if "_embedder" not in para:
        para["_embedder"] = OnlineEmbedder(para)
    return para["_embedder"]


class TopKSAE(nn.Module):
    def __init__(self, d_activation: int, n_features: int, sparsity: int):
        super().__init__()
        self.d_activation = d_activation
        self.n_features = n_features
        self.sparsity = sparsity
        self.dict = nn.Parameter(torch.randn(n_features, d_activation))

    def normalized_dict(self) -> torch.Tensor:
        return self.dict / (self.dict.norm(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_dict = self.normalized_dict()
        scores = torch.einsum("ij,bj->bi", normed_dict, x)
        topk = torch.topk(scores, self.sparsity, dim=-1).indices
        code = torch.zeros_like(scores)
        code.scatter_(dim=-1, index=topk, src=scores.gather(-1, topk))
        code = F.relu(code)
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        normed_dict = self.normalized_dict()
        return torch.einsum("bi,ij->bj", code, normed_dict)

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.forward(x)
        recon = self.decode(code)
        return code, recon

    def loss(self, x_idx: torch.Tensor, para: dict) -> torch.Tensor:
        embedder = get_embedder_cached(para)
        x = embedder.embed_from_indices(x_idx)
        _, recon = self.reconstruct(x)
        return F.mse_loss(recon, x)


# -----------------------------
# Reconstruction metrics
# -----------------------------
def init_stats_tensor(device: torch.device) -> torch.Tensor:
    # [count, sum_x, sum_y, sum_x2, sum_y2, sum_xy, sse]
    return torch.zeros(7, device=device, dtype=torch.float64)


@torch.no_grad()
def update_stats_tensor(stats: torch.Tensor, x: torch.Tensor, recon: torch.Tensor):
    x64 = x.reshape(-1).to(torch.float64)
    y64 = recon.reshape(-1).to(torch.float64)
    err = y64 - x64
    stats[0] += x64.numel()
    stats[1] += x64.sum()
    stats[2] += y64.sum()
    stats[3] += (x64 * x64).sum()
    stats[4] += (y64 * y64).sum()
    stats[5] += (x64 * y64).sum()
    stats[6] += (err * err).sum()


@torch.no_grad()
def evaluate_reconstruction_metrics(model: nn.Module, dataloader_test: DataLoader, para: dict) -> Dict[str, float]:
    is_ddp = para["is_ddp"]
    device = para["device"]
    module = model.module if is_ddp else model
    module.eval()
    embedder = get_embedder_cached(para)

    stats = init_stats_tensor(device)

    for x_idx in dataloader_test:
        x = embedder.embed_from_indices(x_idx)
        _, recon = module.reconstruct(x)
        update_stats_tensor(stats, x, recon)

    stats = allreduce_sum_tensor(is_ddp, stats)

    count, sum_x, sum_y, sum_x2, sum_y2, sum_xy, sse = [float(v) for v in stats.tolist()]
    eps = 1e-12
    mse = sse / max(count, eps)
    nmse = sse / max(sum_x2, eps)

    sst = sum_x2 - (sum_x * sum_x) / max(count, eps)
    explained_variance = 1.0 - (sse / max(sst, eps))

    var_x = sum_x2 - (sum_x * sum_x) / max(count, eps)
    var_y = sum_y2 - (sum_y * sum_y) / max(count, eps)
    cov_xy = sum_xy - (sum_x * sum_y) / max(count, eps)
    pearson_corr = cov_xy / max(math.sqrt(max(var_x, eps) * max(var_y, eps)), eps)

    metrics = {
        "num_elements": int(count),
        "mse": mse,
        "normalized_mse": nmse,
        "explained_variance": explained_variance,
        "pearson_corr": pearson_corr,
        "sum_x2": sum_x2,
        "sse": sse,
    }
    return metrics


# -----------------------------
# IO helpers
# -----------------------------
def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_strs(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def make_save_model_dir(para: dict) -> str:
    basic_dir = os.path.join(para["save_dir"], para["results_subdir"])
    os.makedirs(basic_dir, exist_ok=True)
    save_model_dir = os.path.join(
        basic_dir,
        f"{para['model_name']}_C{para['n_latents']}_K{para['k']}"
    )
    os.makedirs(save_model_dir, exist_ok=True)
    return save_model_dir


def write_run_metrics_csv(save_model_dir: str, payload: dict):
    csv_path = os.path.join(save_model_dir, "metrics.csv")
    keys = list(payload.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(payload)


# -----------------------------
# Training loop
# -----------------------------
def autoencoders_methods_onlineloading(para: dict, save_model_dir: str):
    is_ddp = para["is_ddp"]
    rank = para["rank"]
    world_size = para["world_size"]
    device = para["device"]

    os.makedirs(save_model_dir, exist_ok=True)

    embedder = get_embedder_cached(para)
    d_activation = embedder.d_activation

    if is_rank0(is_ddp, rank):
        print(f"DDP={is_ddp}, rank={rank}/{world_size}, device={device}")
        print(f"d_activation={d_activation} (hidden={embedder.hidden_size} * layers={embedder.num_layers})")

    train_indices = torch.arange(8000)
    test_indices = torch.arange(2000) + 8000

    if is_ddp:
        train_sampler = DistributedSampler(train_indices, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        test_sampler = DistributedSampler(test_indices, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader_train = DataLoader(train_indices, batch_size=para["batch_size"], sampler=train_sampler, shuffle=False, pin_memory=True)
        dataloader_test = DataLoader(test_indices, batch_size=para["batch_size"], sampler=test_sampler, shuffle=False, pin_memory=True)
    else:
        train_sampler = None
        dataloader_train = DataLoader(train_indices, batch_size=para["batch_size"], shuffle=True)
        dataloader_test = DataLoader(test_indices, batch_size=para["batch_size"], shuffle=False)

    model = TopKSAE(
        d_activation=d_activation,
        n_features=para["n_latents"],
        sparsity=para["k"]
    ).to(device=device, dtype=torch.float32)

    if is_ddp:
        model = DDP(model, device_ids=[para["local_rank"]], output_device=para["local_rank"])

    optimizer = optim.AdamW(model.parameters(), lr=para["lr"], weight_decay=para["weight_decay"])
    num_epochs = para["epoch"]

    ckpt_path = os.path.join(save_model_dir, "ckpt.pth")
    loss_path = os.path.join(save_model_dir, "loss.pth")
    model_path = os.path.join(save_model_dir, "model.pth")
    metrics_path = os.path.join(save_model_dir, "metrics.json")

    loss_train_set, loss_test_set = [], []
    start_epoch = 0

    if os.path.exists(ckpt_path):
        if is_rank0(is_ddp, rank):
            print(f"[Resume] Loading checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        if is_ddp:
            model.module.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt["model"], strict=True)

        opt_state = ckpt.get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

        loss_train_set = ckpt.get("train", [])
        loss_test_set = ckpt.get("test", [])
        start_epoch = int(ckpt.get("epoch", len(loss_train_set) - 1)) + 1

    elif os.path.exists(model_path) and os.path.exists(loss_path):
        if is_rank0(is_ddp, rank):
            print(f"[Resume] Loading legacy: {model_path}, {loss_path}")
        try:
            state = torch.load(model_path, map_location=device, weights_only=False)
            results = torch.load(loss_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location=device)
            results = torch.load(loss_path, map_location="cpu")

        if is_ddp:
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)

        loss_train_set = results.get("train", [])
        loss_test_set = results.get("test", [])
        start_epoch = len(loss_train_set)

    if start_epoch < num_epochs:
        if is_ddp:
            dist.barrier()

        if is_rank0(is_ddp, rank):
            epoch_iter = tqdm(
                range(start_epoch, num_epochs),
                desc=f"SAE train L{para['n_latents']}-K{para['k']}-BS{para['batch_size']} ({para['model_name']})"
            )
        else:
            epoch_iter = range(start_epoch, num_epochs)

        for epoch in epoch_iter:
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train
            model.train()
            train_losses_local = []
            for x in dataloader_train:
                optimizer.zero_grad(set_to_none=True)
                loss_train = (model.module.loss(x, para) if is_ddp else model.loss(x, para))
                loss_train.backward()
                optimizer.step()
                train_losses_local.append(float(loss_train.item()))
            train_mean_local = float(np.mean(train_losses_local)) if len(train_losses_local) else 0.0

            # Test
            model.eval()
            test_losses_local = []
            with torch.no_grad():
                for x in dataloader_test:
                    loss_test = (model.module.loss(x, para) if is_ddp else model.loss(x, para))
                    test_losses_local.append(float(loss_test.item()))
            test_mean_local = float(np.mean(test_losses_local)) if len(test_losses_local) else 0.0

            train_mean, test_mean = allreduce_mean_2floats(is_ddp, device, train_mean_local, test_mean_local)

            if is_rank0(is_ddp, rank):
                loss_train_set.append(float(train_mean))
                loss_test_set.append(float(test_mean))

                state = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(state, model_path)
                torch.save({"train": loss_train_set, "test": loss_test_set}, loss_path)
                ckpt = {
                    "epoch": epoch,
                    "model": state,
                    "optimizer": optimizer.state_dict(),
                    "train": loss_train_set,
                    "test": loss_test_set,
                }
                torch.save(ckpt, ckpt_path)

                plt.figure(figsize=(10, 6))
                plt.plot(loss_train_set, label="Train Loss")
                plt.plot(loss_test_set, label="Test Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (mean over elements)")
                plt.title("Training / Test Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_model_dir, "loss.jpg"), bbox_inches="tight")
                plt.close()

            if is_ddp:
                dist.barrier()
    else:
        if is_rank0(is_ddp, rank):
            print(f"[Resume] start_epoch={start_epoch} >= num_epochs={num_epochs}, skip training and evaluate directly.")

    # Final reconstruction metrics
    recon_metrics = evaluate_reconstruction_metrics(model, dataloader_test, para)

    if is_rank0(is_ddp, rank):
        payload = {
            "model_name": para["model_name"],
            "dataset": para["dataset"],
            "batch_size": para["batch_size"],
            "epoch": para["epoch"],
            "n_latents": para["n_latents"],
            "k": para["k"],
            "lr": para["lr"],
            "weight_decay": para["weight_decay"],
            "max_length": para["max_length"],
            "hidden_size": embedder.hidden_size,
            "num_layers": embedder.num_layers,
            "d_activation": embedder.d_activation,
            "final_train_loss": float(loss_train_set[-1]) if len(loss_train_set) else None,
            "final_test_loss": float(loss_test_set[-1]) if len(loss_test_set) else None,
            **recon_metrics,
        }
        save_json(payload, metrics_path)
        write_run_metrics_csv(save_model_dir, payload)
        print(f"[Saved] metrics -> {metrics_path}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    if is_ddp:
        dist.barrier()


# -----------------------------
# Worker for mp.spawn (DDP)
# -----------------------------
def ddp_worker(local_rank: int, world_size: int, master_addr: str, master_port: str, para: dict, save_model_dir: str):
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    ddp_init_if_needed(
        is_ddp=True,
        local_rank=local_rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port
    )

    para = dict(para)
    para["is_ddp"] = True
    para["rank"] = int(os.environ["RANK"])
    para["world_size"] = world_size
    para["local_rank"] = local_rank
    para["device"] = torch.device("cuda", local_rank)

    autoencoders_methods_onlineloading(para, save_model_dir)
    ddp_cleanup(is_ddp=True)


# -----------------------------
# Summary mode for Experiment 1
# -----------------------------
def find_metrics_files(results_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(results_root, "*", "metrics.json")))


def summarize_experiment1(results_root: str, model_names: List[str], k_filter: int = None, out_name: str = "exp1_summary"):
    metrics_files = find_metrics_files(results_root)
    if len(metrics_files) == 0:
        raise FileNotFoundError(f"No metrics.json found under: {results_root}")

    records = []
    for fp in metrics_files:
        rec = load_json(fp)
        rec["run_dir"] = os.path.dirname(fp)
        if model_names and rec["model_name"] not in model_names:
            continue
        if k_filter is not None and int(rec["k"]) != int(k_filter):
            continue
        records.append(rec)

    if len(records) == 0:
        raise RuntimeError("No matching records after filtering.")

    summary_csv = os.path.join(results_root, f"{out_name}.csv")
    fieldnames = sorted(records[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    # Plot EV / NMSE vs n_latents per model
    plt.figure(figsize=(8, 5))
    for m in sorted(set(r["model_name"] for r in records)):
        rr = sorted([r for r in records if r["model_name"] == m], key=lambda x: int(x["n_latents"]))
        xs = [int(r["n_latents"]) for r in rr]
        ys = [float(r["explained_variance"]) for r in rr]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("n_latents")
    plt.ylabel("Explained Variance")
    plt.title("Experiment 1: EV vs n_latents")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, f"{out_name}_ev_vs_latents.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for m in sorted(set(r["model_name"] for r in records)):
        rr = sorted([r for r in records if r["model_name"] == m], key=lambda x: int(x["n_latents"]))
        xs = [int(r["n_latents"]) for r in rr]
        ys = [float(r["normalized_mse"]) for r in rr]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("n_latents")
    plt.ylabel("Normalized MSE")
    plt.title("Experiment 1: NMSE vs n_latents")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, f"{out_name}_nmse_vs_latents.png"), bbox_inches="tight")
    plt.close()

    # Find EV-matched combo across requested models
    groups = {m: sorted([r for r in records if r["model_name"] == m], key=lambda x: int(x["n_latents"])) for m in model_names}
    missing = [m for m, rr in groups.items() if len(rr) == 0]
    if missing:
        raise RuntimeError(f"Missing records for models: {missing}")

    all_combos = list(itertools.product(*[groups[m] for m in model_names]))
    scored = []
    for combo in all_combos:
        evs = [float(x["explained_variance"]) for x in combo]
        mean_evs = float(np.mean(evs))
        ev_range = float(max(evs) - min(evs))
        scored.append((ev_range, -mean_evs, combo))
    scored.sort(key=lambda x: (x[0], x[1]))

    top_rows = []
    for rank, (ev_range, neg_mean_ev, combo) in enumerate(scored[:20], start=1):
        row = {
            "rank": rank,
            "ev_range": ev_range,
            "mean_ev": -neg_mean_ev,
        }
        for item in combo:
            tag = item["model_name"]
            row[f"{tag}_n_latents"] = item["n_latents"]
            row[f"{tag}_k"] = item["k"]
            row[f"{tag}_ev"] = item["explained_variance"]
            row[f"{tag}_nmse"] = item["normalized_mse"]
            row[f"{tag}_run_dir"] = item["run_dir"]
        top_rows.append(row)

    matched_csv = os.path.join(results_root, f"{out_name}_matched_triplets.csv")
    matched_fieldnames = sorted(top_rows[0].keys()) if top_rows else ["rank", "ev_range", "mean_ev"]
    with open(matched_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matched_fieldnames)
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)

    best_payload = top_rows[0] if top_rows else {}
    save_json(best_payload, os.path.join(results_root, f"{out_name}_best_match.json"))

    print(f"[Saved] summary csv -> {summary_csv}")
    print(f"[Saved] matched triplets -> {matched_csv}")
    if best_payload:
        print("[Best EV-matched combo]")
        print(json.dumps(best_payload, indent=2, ensure_ascii=False))


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "summarize"])

    # DDP / hardware
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29601")

    # paths
    parser.add_argument("--data_dir", type=str, default="/data/hxt/P5_1/Dataset_text/")
    parser.add_argument("--model_dir", type=str, default="/data/hxt/P5_1/EleutherAI/")
    parser.add_argument("--save_dir", type=str, default="/data/hxt/P5_1/")
    parser.add_argument("--results_subdir", type=str, default="pythia_results_pretrainedSAE_exp1")
    parser.add_argument("--dataset", type=str, default="pile-10k")
    parser.add_argument("--model_name", type=str, default="pythia-410m")

    # training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=512)
    parser.add_argument("--n_latents", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_length", type=int, default=2048)

    # summarize mode
    parser.add_argument("--summary_models", type=str, default="pythia-70m,pythia-410m,pythia-1.4b")
    parser.add_argument("--summary_k", type=int, default=32)
    parser.add_argument("--summary_out_name", type=str, default="exp1_summary")

    args = parser.parse_args()

    results_root = os.path.join(args.save_dir, args.results_subdir)

    if args.mode == "summarize":
        summarize_experiment1(
            results_root=results_root,
            model_names=parse_csv_strs(args.summary_models),
            k_filter=args.summary_k,
            out_name=args.summary_out_name,
        )
        return

    gpu_list = parse_csv_ints(args.gpus)
    if len(gpu_list) == 0:
        gpu_list = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_list)

    para = {
        "code_dir": os.path.dirname(os.path.abspath(__file__)),
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "save_dir": args.save_dir,
        "results_subdir": args.results_subdir,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "n_latents": args.n_latents,
        "k": args.k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dataset": args.dataset,
        "max_length": args.max_length,
        # ddp fields
        "is_ddp": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "device": None,
    }

    save_model_dir = make_save_model_dir(para)

    if len(gpu_list) == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        para["device"] = device
        para["is_ddp"] = False
        para["rank"] = 0
        para["world_size"] = 1
        para["local_rank"] = 0

        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> Single GPU mode on {device}")
        autoencoders_methods_onlineloading(para, save_model_dir)
        return

    world_size = len(gpu_list)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> Auto DDP mode, world_size={world_size}")
    mp.spawn(
        ddp_worker,
        args=(world_size, args.master_addr, args.master_port, para, save_model_dir),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
