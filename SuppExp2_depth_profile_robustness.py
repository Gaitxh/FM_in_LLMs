# -*- coding: utf-8 -*-
"""
Experiment 2: Depth-profile hyperparameter robustness

Built from the user's original SAE training code and the Experiment 1 extension.
What this script adds beyond basic SAE training:
1) Sweeping over SAE hyperparameters: n_latents (M) and sparsity k (K_SAE)
2) Depth-profile analysis over multiple salient thresholds and cluster numbers
3) Active-module extraction from test-set activation frequency (AF > 0)
4) Per-setting clustering robustness metrics:
   - silhouette score (correlation distance)
   - Davies-Bouldin score
   - peak-layer ordering
   - optional ARI/NMI to the default threshold within the same run
   - template similarity to a chosen default reference run
5) Summary mode for aggregating all runs and generating heatmaps / csv tables

Paper-aligned implementation details:
- Active modules: modules with AF > 0 on the test split.
- Salient ANs: within-module z-normalized decoder weights > threshold.
- Depth profile v_m: layer-count vector built from salient AN counts per layer.
- Module grouping: hierarchical clustering over v_m using correlation distance.

Recommended usage:
  1) train several runs over M and K
  2) analyze each run over thresholds and cluster_k values
  3) summarize across all analyzed runs
"""

import os
import glob
import csv
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

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
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment


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
    模型常驻 GPU：进程内只加载一次 GPTNeoXModel，后续复用
    输出 token 级 embedding 拼接 [sum_tokens, d_activation]，float32
    """
    def __init__(self, para: dict):
        self.para = para
        self.device = para["device"]
        self.max_length = int(para.get("max_length", 2048))

        self.ds = self._load_text_dataset(para)

        self.model_path = os.path.join(para["model_dir"], para["model_name"])
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"模型目录不存在：{self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = AutoConfig.from_pretrained(self.model_path)
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_hidden_layers)
        self.d_activation = self.hidden_size * self.num_layers

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
                low_cpu_mem_usage=True,
            ).to(self.device)
            lm.eval()
            for p in lm.parameters():
                p.requires_grad_(False)
            self.lm = lm
        return self.lm

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
            dim=-1,
        )

        attn = inputs["attention_mask"].bool()
        batch_data = []
        for b in range(x.size(0)):
            xb = x[b, attn[b]]
            xb_fp32 = self.torch_zscore_fp32(xb.float(), dim=0)
            batch_data.append(xb_fp32.to(torch.float32))

        return torch.cat(batch_data, dim=0)


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
# Reconstruction metrics (kept for completeness)
# -----------------------------
def init_stats_tensor(device: torch.device) -> torch.Tensor:
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

    return {
        "num_elements": int(count),
        "mse": mse,
        "normalized_mse": nmse,
        "explained_variance": explained_variance,
        "pearson_corr": pearson_corr,
        "sum_x2": sum_x2,
        "sse": sse,
    }


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


def parse_csv_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


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


def make_analysis_dir(save_model_dir: str) -> str:
    out_dir = os.path.join(save_model_dir, "exp2_depth")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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
        sparsity=para["k"],
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
            epoch_iter = tqdm(range(start_epoch, num_epochs), desc=f"SAE train L{para['n_latents']}-K{para['k']}-BS{para['batch_size']} ({para['model_name']})")
        else:
            epoch_iter = range(start_epoch, num_epochs)

        for epoch in epoch_iter:
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_losses_local = []
            for x in dataloader_train:
                optimizer.zero_grad(set_to_none=True)
                loss_train = (model.module.loss(x, para) if is_ddp else model.loss(x, para))
                loss_train.backward()
                optimizer.step()
                train_losses_local.append(float(loss_train.item()))
            train_mean_local = float(np.mean(train_losses_local)) if len(train_losses_local) else 0.0

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
        master_port=master_port,
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
# Experiment 2 helpers
# -----------------------------
def make_test_dataloader(para: dict) -> DataLoader:
    test_indices = torch.arange(2000) + 8000
    return DataLoader(test_indices, batch_size=para["batch_size"], shuffle=False)


@torch.no_grad()
def extract_active_modules_and_depth_profiles(model: TopKSAE, para: dict, thresholds: List[float]) -> Dict[str, dict]:
    """
    Returns AF + active mask + v_m per threshold.
    """
    device = para["device"]
    embedder = get_embedder_cached(para)
    dataloader_test = make_test_dataloader(para)

    model.eval()
    n_features = model.n_features
    active_token_counts = torch.zeros(n_features, dtype=torch.long, device=device)
    total_tokens = 0

    for x_idx in tqdm(dataloader_test, desc="[Exp2] Collect AF", leave=False):
        x = embedder.embed_from_indices(x_idx)
        code = model.forward(x)
        active_token_counts += (code > 0).sum(dim=0).to(torch.long)
        total_tokens += int(code.shape[0])

    af = (active_token_counts.float() / max(total_tokens, 1)).detach().cpu().numpy()
    active_mask = (active_token_counts > 0).detach().cpu().numpy().astype(bool)
    active_indices = np.flatnonzero(active_mask)

    normed_dict = model.normalized_dict().detach().cpu().numpy()  # [M, d_activation]
    M, d_activation = normed_dict.shape
    L = embedder.num_layers
    H = embedder.hidden_size
    assert d_activation == L * H, f"d_activation mismatch: got {d_activation}, expect {L*H}"

    out = {
        "af": af,
        "active_mask": active_mask,
        "active_indices": active_indices,
        "total_tokens": total_tokens,
        "num_active_modules": int(active_mask.sum()),
        "num_layers": L,
        "hidden_size": H,
        "thresholds": {},
    }

    # z-normalize weights within each module across all AN dimensions
    mu = normed_dict.mean(axis=1, keepdims=True)
    sigma = normed_dict.std(axis=1, keepdims=True)
    z = (normed_dict - mu) / (sigma + 1e-6)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    for thr in thresholds:
        salient = z > float(thr)
        salient_active = salient[active_mask]
        vm = salient_active.reshape(salient_active.shape[0], L, H).sum(axis=-1).astype(np.float32)
        out["thresholds"][str(thr)] = {
            "vm": vm,
            "salient_counts": salient[active_mask].sum(axis=1).astype(np.int64),
        }

    return out


def corr_distance_matrix(vms: np.ndarray) -> np.ndarray:
    """
    Correlation distance on rows of vms, robust to zero-variance rows.
    Returns NxN matrix in [0, 2].
    """
    v = np.asarray(vms, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"vms must be 2D, got shape={v.shape}")
    n = v.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    v_center = v - v.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(v_center, axis=1, keepdims=True)
    safe = norms.squeeze(-1) > 1e-12
    v_unit = np.zeros_like(v_center)
    if safe.any():
        v_unit[safe] = v_center[safe] / norms[safe]
    corr = v_unit @ v_unit.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist



def cluster_vms_hierarchical(vms: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    n = vms.shape[0]
    if n == 0:
        raise ValueError("No active modules to cluster.")
    if n < n_clusters:
        raise ValueError(f"n_active={n} < n_clusters={n_clusters}")

    D = corr_distance_matrix(vms)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    order = leaves_list(Z)

    if len(np.unique(labels)) < 2:
        sil = float("nan")
        dbi = float("nan")
    else:
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        dbi = float(davies_bouldin_score(vms, labels))

    mean_profiles = []
    cluster_sizes = []
    peak_layers = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        cluster_sizes.append(int(len(idx)))
        if len(idx) == 0:
            mean_profile = np.zeros(vms.shape[1], dtype=np.float32)
            peak = -1
        else:
            mean_profile = vms[idx].mean(axis=0).astype(np.float32)
            peak = int(np.argmax(mean_profile))
        mean_profiles.append(mean_profile)
        peak_layers.append(peak)
    mean_profiles = np.stack(mean_profiles, axis=0)
    peak_layers = np.array(peak_layers, dtype=np.int64)

    # Order clusters by increasing peak layer: shallow -> middle -> deep style ordering.
    valid_peak = np.where(peak_layers >= 0, peak_layers, 10**9)
    cluster_order = np.argsort(valid_peak)
    relabeled = np.zeros_like(labels)
    for new_label, old_label in enumerate(cluster_order):
        relabeled[labels == old_label] = new_label
    ordered_mean_profiles = mean_profiles[cluster_order]
    ordered_peak_layers = peak_layers[cluster_order]
    ordered_cluster_sizes = [cluster_sizes[int(i)] for i in cluster_order]

    return {
        "distance_matrix": D.astype(np.float32),
        "linkage": Z.astype(np.float32),
        "labels": relabeled.astype(np.int64),
        "leaf_order": order.astype(np.int64),
        "silhouette": sil,
        "davies_bouldin": dbi,
        "mean_profiles": ordered_mean_profiles,
        "peak_layers": ordered_peak_layers.astype(np.int64),
        "cluster_sizes": np.array(ordered_cluster_sizes, dtype=np.int64),
    }



def profile_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between profile rows.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A_safe = A / np.maximum(A_norm, 1e-12)
    B_safe = B / np.maximum(B_norm, 1e-12)
    sim = A_safe @ B_safe.T
    return np.clip(sim, -1.0, 1.0)



def match_templates(sim: np.ndarray) -> Tuple[float, List[Tuple[int, int]], List[float]]:
    """
    Hungarian matching that maximizes mean similarity.
    Works for rectangular matrices by matching min(nA, nB) pairs.
    """
    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    vals = [float(sim[r, c]) for r, c in zip(row_ind, col_ind)]
    mean_sim = float(np.mean(vals)) if vals else float("nan")
    pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    return mean_sim, pairs, vals



def save_npy(path: str, arr: np.ndarray):
    np.save(path, arr)



def plot_module_layer_heatmap(vm: np.ndarray, labels: np.ndarray, leaf_order: np.ndarray, path: str, title: str):
    order = np.asarray(leaf_order, dtype=np.int64)
    vm_ord = vm[order]
    lab_ord = labels[order]

    plt.figure(figsize=(10, 7))
    plt.imshow(vm_ord, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Salient AN count")
    plt.xlabel("Layer")
    plt.ylabel("Active modules (ordered by hierarchical clustering)")
    plt.title(title)

    # Draw boundaries between clusters in leaf order.
    boundaries = []
    for i in range(1, len(lab_ord)):
        if lab_ord[i] != lab_ord[i - 1]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        plt.axhline(b, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



def plot_mean_profiles(mean_profiles: np.ndarray, peak_layers: np.ndarray, path: str, title: str):
    plt.figure(figsize=(8, 5))
    names = [f"Group{i+1}" for i in range(mean_profiles.shape[0])]
    for i in range(mean_profiles.shape[0]):
        plt.plot(mean_profiles[i], label=f"{names[i]} (peak={int(peak_layers[i])})")
    plt.xlabel("Layer")
    plt.ylabel("Mean salient AN count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



def plot_threshold_cluster_grid(entries: List[dict], thresholds: List[float], cluster_ks: List[int], metric_key: str, path: str, title: str):
    configs = sorted(list({e["config_name"] for e in entries}))
    n_rows = len(configs)
    n_cols = len(thresholds)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(figsize=(max(8, n_cols * 2.0), max(4, n_rows * 0.5 + 2.0)), nrows=1, ncols=1)
    heat = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i, cfg in enumerate(configs):
        for j, thr in enumerate(thresholds):
            subset = [e for e in entries if e["config_name"] == cfg and abs(float(e["threshold"]) - float(thr)) < 1e-9]
            if len(subset) == 0:
                continue
            # pick k=3 if present, otherwise best silhouette among all cluster_k
            k3 = [e for e in subset if int(e["cluster_k"]) == 3]
            if len(k3) > 0:
                heat[i, j] = float(k3[0].get(metric_key, np.nan))
            else:
                subset_sorted = sorted(subset, key=lambda x: x.get("silhouette", -1e9), reverse=True)
                heat[i, j] = float(subset_sorted[0].get(metric_key, np.nan))

    im = axes.imshow(heat, aspect="auto", interpolation="nearest")
    axes.set_xticks(range(n_cols))
    axes.set_xticklabels([str(t) for t in thresholds], rotation=45, ha="right")
    axes.set_yticks(range(n_rows))
    axes.set_yticklabels(configs)
    axes.set_xlabel("Salient threshold")
    axes.set_ylabel("SAE config")
    axes.set_title(title)
    fig.colorbar(im, ax=axes, label=metric_key)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



def load_model_weights_into_module(model: TopKSAE, save_model_dir: str, device: torch.device):
    model_path = os.path.join(save_model_dir, "model.pth")
    ckpt_path = os.path.join(save_model_dir, "ckpt.pth")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"]
    elif os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model checkpoint found under: {save_model_dir}")
    model.load_state_dict(state, strict=True)


# -----------------------------
# Experiment 2: analyze one run
# -----------------------------
def analyze_one_run(para: dict, save_model_dir: str, thresholds: List[float], cluster_ks: List[int], default_threshold: float, default_cluster_k: int):
    device = para["device"]
    embedder = get_embedder_cached(para)
    analysis_dir = make_analysis_dir(save_model_dir)

    model = TopKSAE(
        d_activation=embedder.d_activation,
        n_features=para["n_latents"],
        sparsity=para["k"],
    ).to(device=device, dtype=torch.float32)
    load_model_weights_into_module(model, save_model_dir, device)
    model.eval()

    extracted = extract_active_modules_and_depth_profiles(model, para, thresholds)

    save_npy(os.path.join(analysis_dir, "af.npy"), extracted["af"].astype(np.float32))
    save_npy(os.path.join(analysis_dir, "active_mask.npy"), extracted["active_mask"].astype(np.int64))
    save_npy(os.path.join(analysis_dir, "active_indices.npy"), extracted["active_indices"].astype(np.int64))

    summary_rows = []
    threshold_labels_store = {}
    default_labels_for_ari = {}  # cluster_k -> labels at default threshold within this run

    for thr in thresholds:
        thr_key = str(thr)
        vm = extracted["thresholds"][thr_key]["vm"]
        salient_counts = extracted["thresholds"][thr_key]["salient_counts"]
        save_npy(os.path.join(analysis_dir, f"vm_thr_{thr_key}.npy"), vm)
        save_npy(os.path.join(analysis_dir, f"salient_counts_thr_{thr_key}.npy"), salient_counts)

        if vm.shape[0] == 0:
            print(f"[Warn] No active modules for threshold={thr}. Skip clustering.")
            continue

        threshold_labels_store[thr_key] = {}

        for ck in cluster_ks:
            if vm.shape[0] < ck:
                print(f"[Warn] n_active={vm.shape[0]} < cluster_k={ck}; skip.")
                continue

            clus = cluster_vms_hierarchical(vm, ck)
            labels = clus["labels"]
            threshold_labels_store[thr_key][str(ck)] = labels.tolist()
            if abs(float(thr) - float(default_threshold)) < 1e-9:
                default_labels_for_ari[str(ck)] = labels.copy()

            save_npy(os.path.join(analysis_dir, f"labels_thr_{thr_key}_k_{ck}.npy"), labels)
            save_npy(os.path.join(analysis_dir, f"mean_profiles_thr_{thr_key}_k_{ck}.npy"), clus["mean_profiles"])
            save_npy(os.path.join(analysis_dir, f"distance_thr_{thr_key}_k_{ck}.npy"), clus["distance_matrix"])
            save_npy(os.path.join(analysis_dir, f"leaf_order_thr_{thr_key}_k_{ck}.npy"), clus["leaf_order"])

            ari_default = float("nan")
            nmi_default = float("nan")
            if str(ck) in default_labels_for_ari and len(default_labels_for_ari[str(ck)]) == len(labels):
                ari_default = float(adjusted_rand_score(default_labels_for_ari[str(ck)], labels))
                nmi_default = float(normalized_mutual_info_score(default_labels_for_ari[str(ck)], labels))

            row = {
                "model_name": para["model_name"],
                "dataset": para["dataset"],
                "n_latents": int(para["n_latents"]),
                "k": int(para["k"]),
                "config_name": f"C{para['n_latents']}_K{para['k']}",
                "threshold": float(thr),
                "cluster_k": int(ck),
                "num_active_modules": int(vm.shape[0]),
                "mean_saline_count": float(np.mean(salient_counts)) if len(salient_counts) else float("nan"),
                "median_saline_count": float(np.median(salient_counts)) if len(salient_counts) else float("nan"),
                "silhouette": float(clus["silhouette"]),
                "davies_bouldin": float(clus["davies_bouldin"]),
                "peak_layers": [int(x) for x in clus["peak_layers"].tolist()],
                "cluster_sizes": [int(x) for x in clus["cluster_sizes"].tolist()],
                "ari_to_default_threshold": ari_default,
                "nmi_to_default_threshold": nmi_default,
                "run_dir": save_model_dir,
                "analysis_dir": analysis_dir,
                "mean_profiles_path": os.path.join(analysis_dir, f"mean_profiles_thr_{thr_key}_k_{ck}.npy"),
            }
            summary_rows.append(row)

            # Plots only for the default k or small number of useful settings to avoid clutter.
            if int(ck) == int(default_cluster_k):
                plot_module_layer_heatmap(
                    vm=vm,
                    labels=labels,
                    leaf_order=clus["leaf_order"],
                    path=os.path.join(analysis_dir, f"heatmap_thr_{thr_key}_k_{ck}.png"),
                    title=f"{para['model_name']} C{para['n_latents']} K{para['k']} | thr={thr} | k={ck}",
                )
                plot_mean_profiles(
                    mean_profiles=clus["mean_profiles"],
                    peak_layers=clus["peak_layers"],
                    path=os.path.join(analysis_dir, f"mean_profiles_thr_{thr_key}_k_{ck}.png"),
                    title=f"Mean depth profiles | thr={thr} | k={ck}",
                )

    # Save raw summary for this run
    analysis_json = os.path.join(analysis_dir, "analysis.json")
    analysis_payload = {
        "model_name": para["model_name"],
        "n_latents": int(para["n_latents"]),
        "k": int(para["k"]),
        "thresholds": thresholds,
        "cluster_ks": cluster_ks,
        "default_threshold": float(default_threshold),
        "default_cluster_k": int(default_cluster_k),
        "num_active_modules": int(extracted["num_active_modules"]),
        "total_tokens": int(extracted["total_tokens"]),
        "summary_rows": summary_rows,
    }
    save_json(analysis_payload, analysis_json)

    # CSV for convenience
    if len(summary_rows) > 0:
        csv_path = os.path.join(analysis_dir, "analysis_summary.csv")
        fieldnames = [
            "model_name", "dataset", "n_latents", "k", "config_name", "threshold", "cluster_k",
            "num_active_modules", "mean_saline_count", "median_saline_count", "silhouette",
            "davies_bouldin", "peak_layers", "cluster_sizes", "ari_to_default_threshold",
            "nmi_to_default_threshold", "run_dir", "analysis_dir", "mean_profiles_path",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                flat = dict(row)
                flat["peak_layers"] = json.dumps(flat["peak_layers"], ensure_ascii=False)
                flat["cluster_sizes"] = json.dumps(flat["cluster_sizes"], ensure_ascii=False)
                writer.writerow(flat)

    print(f"[Saved] Exp2 analysis -> {analysis_json}")


# -----------------------------
# Experiment 2: summarize across runs
# -----------------------------
def find_exp2_analysis_files(results_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(results_root, "*", "exp2_depth", "analysis.json")))



def select_reference_entry(entries: List[dict], ref_model: str, ref_latents: int, ref_k: int, ref_threshold: float, ref_cluster_k: int) -> dict:
    cand = [
        e for e in entries
        if e["model_name"] == ref_model
        and int(e["n_latents"]) == int(ref_latents)
        and int(e["k"]) == int(ref_k)
        and abs(float(e["threshold"]) - float(ref_threshold)) < 1e-9
        and int(e["cluster_k"]) == int(ref_cluster_k)
    ]
    if len(cand) == 0:
        raise RuntimeError(
            f"Reference entry not found: model={ref_model}, C={ref_latents}, K={ref_k}, threshold={ref_threshold}, cluster_k={ref_cluster_k}"
        )
    return cand[0]



def summarize_experiment2(
    results_root: str,
    model_names: List[str],
    thresholds: List[float],
    cluster_ks: List[int],
    ref_model: str,
    ref_latents: int,
    ref_k: int,
    ref_threshold: float,
    ref_cluster_k: int,
    out_name: str = "exp2_summary",
):
    analysis_files = find_exp2_analysis_files(results_root)
    if len(analysis_files) == 0:
        raise FileNotFoundError(f"No exp2 analysis.json found under: {results_root}")

    entries: List[dict] = []
    for fp in analysis_files:
        payload = load_json(fp)
        for row in payload.get("summary_rows", []):
            if model_names and row["model_name"] not in model_names:
                continue
            entries.append(row)

    if len(entries) == 0:
        raise RuntimeError("No matching exp2 analysis entries after filtering.")

    ref_entry = select_reference_entry(entries, ref_model, ref_latents, ref_k, ref_threshold, ref_cluster_k)
    ref_profiles = np.load(ref_entry["mean_profiles_path"])

    # Add template similarity to reference mean profiles.
    for e in entries:
        cur_profiles = np.load(e["mean_profiles_path"])
        sim = profile_similarity_matrix(ref_profiles, cur_profiles)
        mean_sim, pairs, vals = match_templates(sim)
        e["template_similarity_to_ref"] = mean_sim
        e["template_match_pairs"] = pairs
        e["template_match_values"] = vals

    # Save full csv
    summary_csv = os.path.join(results_root, f"{out_name}.csv")
    fieldnames = [
        "model_name", "dataset", "n_latents", "k", "config_name", "threshold", "cluster_k",
        "num_active_modules", "mean_saline_count", "median_saline_count", "silhouette", "davies_bouldin",
        "peak_layers", "cluster_sizes", "ari_to_default_threshold", "nmi_to_default_threshold",
        "template_similarity_to_ref", "template_match_pairs", "template_match_values",
        "run_dir", "analysis_dir", "mean_profiles_path",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            flat = dict(e)
            flat["peak_layers"] = json.dumps(flat["peak_layers"], ensure_ascii=False)
            flat["cluster_sizes"] = json.dumps(flat["cluster_sizes"], ensure_ascii=False)
            flat["template_match_pairs"] = json.dumps(flat["template_match_pairs"], ensure_ascii=False)
            flat["template_match_values"] = json.dumps(flat["template_match_values"], ensure_ascii=False)
            writer.writerow(flat)

    # Heatmaps using k=3 by default preference
    plot_threshold_cluster_grid(
        entries=entries,
        thresholds=thresholds,
        cluster_ks=cluster_ks,
        metric_key="template_similarity_to_ref",
        path=os.path.join(results_root, f"{out_name}_template_similarity_k3pref.png"),
        title="Experiment 2: template similarity to reference (prefer k=3)",
    )
    plot_threshold_cluster_grid(
        entries=entries,
        thresholds=thresholds,
        cluster_ks=cluster_ks,
        metric_key="silhouette",
        path=os.path.join(results_root, f"{out_name}_silhouette_k3pref.png"),
        title="Experiment 2: silhouette (prefer k=3)",
    )

    # Reference-focused line plot across cluster numbers.
    ref_subset = [
        e for e in entries
        if e["model_name"] == ref_model and int(e["n_latents"]) == int(ref_latents) and int(e["k"]) == int(ref_k)
    ]
    if len(ref_subset) > 0:
        plt.figure(figsize=(8, 5))
        for thr in thresholds:
            rr = [e for e in ref_subset if abs(float(e["threshold"]) - float(thr)) < 1e-9]
            rr = sorted(rr, key=lambda x: int(x["cluster_k"]))
            if len(rr) == 0:
                continue
            plt.plot([int(x["cluster_k"]) for x in rr], [float(x["silhouette"]) for x in rr], marker="o", label=f"thr={thr}")
        plt.xlabel("cluster_k")
        plt.ylabel("silhouette")
        plt.title(f"Reference config silhouette across cluster_k ({ref_model}, C{ref_latents}, K{ref_k})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_root, f"{out_name}_ref_silhouette_curve.png"), bbox_inches="tight")
        plt.close()

    ref_json = os.path.join(results_root, f"{out_name}_reference.json")
    save_json({
        "ref_model": ref_model,
        "ref_latents": ref_latents,
        "ref_k": ref_k,
        "ref_threshold": ref_threshold,
        "ref_cluster_k": ref_cluster_k,
        "ref_mean_profiles_path": ref_entry["mean_profiles_path"],
    }, ref_json)

    print(f"[Saved] Exp2 summary -> {summary_csv}")
    print(f"[Saved] Exp2 reference -> {ref_json}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "analyze", "summarize"])

    # common paths
    parser.add_argument("--data_dir", type=str, default="/data/hxt/P5_1/Dataset_text/")
    parser.add_argument("--model_dir", type=str, default="/data/hxt/P5_1/EleutherAI/")
    parser.add_argument("--save_dir", type=str, default="/data/hxt/P5_1/")
    parser.add_argument("--results_subdir", type=str, default="pythia_results_pretrainedSAE_exp2")

    # train/analyze target run
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--dataset", type=str, default="pile-10k")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=512)
    parser.add_argument("--n_latents", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)

    # gpu / ddp
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29601")

    # experiment2 analyze
    parser.add_argument("--thresholds", type=str, default="1.0,1.65,2.0,2.5")
    parser.add_argument("--cluster_ks", type=str, default="2,3,4,5")
    parser.add_argument("--default_threshold", type=float, default=1.65)
    parser.add_argument("--default_cluster_k", type=int, default=3)

    # summarize mode
    parser.add_argument("--summary_models", type=str, default="pythia-410m")
    parser.add_argument("--summary_out_name", type=str, default="exp2_summary")
    parser.add_argument("--ref_model", type=str, default="pythia-410m")
    parser.add_argument("--ref_latents", type=int, default=1024)
    parser.add_argument("--ref_k", type=int, default=32)
    parser.add_argument("--ref_threshold", type=float, default=1.65)
    parser.add_argument("--ref_cluster_k", type=int, default=3)

    args = parser.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    cluster_ks = parse_csv_ints(args.cluster_ks)
    results_root = os.path.join(args.save_dir, args.results_subdir)

    if args.mode == "summarize":
        summarize_experiment2(
            results_root=results_root,
            model_names=parse_csv_strs(args.summary_models),
            thresholds=thresholds,
            cluster_ks=cluster_ks,
            ref_model=args.ref_model,
            ref_latents=args.ref_latents,
            ref_k=args.ref_k,
            ref_threshold=args.ref_threshold,
            ref_cluster_k=args.ref_cluster_k,
            out_name=args.summary_out_name,
        )
        return

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    para = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "save_dir": args.save_dir,
        "results_subdir": args.results_subdir,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "n_latents": args.n_latents,
        "k": args.k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "is_ddp": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "device": None,
    }
    save_model_dir = make_save_model_dir(para)

    if args.mode == "analyze":
        # analyze mode intentionally runs on a single GPU / process to simplify evaluation I/O.
        if not torch.cuda.is_available():
            raise RuntimeError("Analyze mode currently expects CUDA.")
        para["device"] = torch.device("cuda", 0)
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> Analyze mode on {para['device']}")
        analyze_one_run(
            para=para,
            save_model_dir=save_model_dir,
            thresholds=thresholds,
            cluster_ks=cluster_ks,
            default_threshold=args.default_threshold,
            default_cluster_k=args.default_cluster_k,
        )
        return

    # train mode
    if len(gpus) == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for train mode.")
        device = torch.device("cuda", 0)
        para["device"] = device
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> Single GPU mode on {device}")
        autoencoders_methods_onlineloading(para, save_model_dir)
        return

    world_size = len(gpus)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> Auto DDP mode, world_size={world_size}")
    mp.spawn(
        ddp_worker,
        args=(world_size, args.master_addr, args.master_port, para, save_model_dir),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
