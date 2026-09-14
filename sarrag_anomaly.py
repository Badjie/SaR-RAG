"""
SaR-RAG: Selective and Refined Retrieval-Augmented Anomaly Memory with
Knowledge Graph Integration for Anomaly Detection in Cryptocurrency
Transaction Networks.

This script extends the MGGRU temporal graph pipeline (mglstmrag.py) with the
five-stage SaR-RAG architecture:

    Transaction Data
        -> Knowledge Graph Construction        (Sec. 3)
        -> KG Representation Learning (KGNN)   (Sec. 4)
        -> Selective Retrieval                 (Sec. 5)
        -> Memory Refinement                   (Sec. 6)
        -> Anomaly Prediction                  (Sec. 7)

Every concept from the source specification is expressed in anomaly-detection
terms: the memory bank is an anomaly memory, retrieved cases are anomaly
evidence, and the output head produces an anomaly probability.

Datasets
--------
  soc-sign-bitcoinalpha.csv    columns: from_address, to_address, value, timestamp
  0to999999_ERC20Transaction.csv
                               columns: blockNumber, timestamp, transactionHash,
                                        tokenAddress, from, to, fromIsContract,
                                        toIsContract, value

Usage
-----
  python sarrag_anomaly.py --dataset bitcoinalpha
  python sarrag_anomaly.py --dataset erc20 --epochs 60
  python sarrag_anomaly.py --dataset bitcoinalpha --ablation no_kg
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Knowledge graph schema (Sec. 3.1 / 3.2)
#
# The specification lists ten entity categories. Six of them have an observable
# counterpart in the two transaction logs used here; four (Device, IP Address,
# Location, and real User identity) have no observable counterpart in on-chain
# or rating data and are therefore NOT instantiated. Claiming them would put the
# manuscript out of step with the code.
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    "WALLET": 0,          # cryptocurrency address (Sec. 3.1 Wallet)
    "CONTRACT": 1,        # smart contract address (Sec. 3.1 Smart Contract)
    "TOKEN": 2,           # ERC20 token / value band  (Sec. 3.1 Exchange proxy)
    "HUB": 3,             # exchange-like high-throughput address (Sec. 3.1 Exchange)
    "ANOMALY_EVENT": 4,   # confirmed anomalous community (Sec. 3.1 Fraud Event)
    "PATTERN": 5,         # anomaly archetype           (Sec. 3.1 Attack Type)
}

RELATION_TYPES = {
    "transfers": 0,          # (Wallet_1, transfers, Wallet_2)
    "interacts_with": 1,     # (Wallet, interacts, Contract)
    "uses_token": 2,         # (Wallet, uses, Token/ValueBand)
    "routes_through": 3,     # (Wallet, routes_through, Hub)
    "exhibits": 4,           # (Wallet, exhibits, Pattern)
    "belongs_to": 5,         # (Wallet, belongs_to, AnomalyEvent)
    "co_counterparty": 6,    # (Wallet, co_counterparty, Wallet) -- multi-hop link
    "associated_with": 7,    # (AnomalyEvent, associated_with, Pattern)
}

NUM_RELATIONS = len(RELATION_TYPES)
NUM_ENTITY_TYPES = len(ENTITY_TYPES)

PATTERN_NAMES = ["dispersion", "collection", "relay", "burst"]


# ===========================================================================
# 1. Data loading
# ===========================================================================

def load_dataset(name: str, path: str) -> Tuple[pd.DataFrame, Dict]:
    """Return a frame with a unified schema plus dataset metadata."""
    df = pd.read_csv(path)

    if name == "bitcoinotc":
        df = df.rename(columns={"from_address": "src", "to_address": "dst"})
        df["src"] = df["src"].astype(str)
        df["dst"] = df["dst"].astype(str)
        # `value` in Bitcoin-Alpha is a trust rating in [-10, 10]; a negative
        # rating is a direct distrust signal and is kept as a risk prior.
        df["rating"] = df["value"].astype(float)
        df["amount"] = df["value"].abs().astype(float)
        df["token"] = "band_" + pd.cut(
            df["rating"], bins=[-10.1, -1, 0, 2, 5, 10],
            labels=["strong_neg", "neg", "low_pos", "mid_pos", "high_pos"],
        ).astype(str)
        df["src_is_contract"] = 0
        df["dst_is_contract"] = 0
        meta = {"slice_days": 31, "has_contracts": False, "has_rating": True}

    elif name == "erc20":
        df = df.rename(columns={"from": "src", "to": "dst",
                                "tokenAddress": "token",
                                "fromIsContract": "src_is_contract",
                                "toIsContract": "dst_is_contract"})
        df["src"] = df["src"].astype(str)
        df["dst"] = df["dst"].astype(str)
        df["amount"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0).astype(float)
        df["rating"] = 0.0
        df["src_is_contract"] = pd.to_numeric(df["src_is_contract"], errors="coerce").fillna(0).astype(int)
        df["dst_is_contract"] = pd.to_numeric(df["dst_is_contract"], errors="coerce").fillna(0).astype(int)
        meta = {"slice_days": 7, "has_contracts": True, "has_rating": False}

    else:
        raise ValueError(f"unknown dataset '{name}'")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "src", "dst"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("dt").reset_index(drop=True)
    return df[["src", "dst", "amount", "rating", "token", "src_is_contract",
               "dst_is_contract", "timestamp", "dt"]], meta


def temporal_slices(df: pd.DataFrame, slice_days: int, min_edges: int = 20) -> pd.DataFrame:
    """Assign each transaction to a temporal slice, merging sparse tails."""
    df = df.copy()
    df["time_slice"] = (df["dt"] - df["dt"].min()).dt.days // slice_days
    counts = df["time_slice"].value_counts()
    sparse = counts[counts < min_edges].index.tolist()
    if sparse:
        df.loc[df["time_slice"].isin(sparse), "time_slice"] = int(df["time_slice"].max()) + 1
    df["time_slice"] = df["time_slice"].astype(int)
    return df


# ===========================================================================
# 2. Transaction graph and weak anomaly labelling
# ===========================================================================

def create_graph(slice_df: pd.DataFrame) -> nx.DiGraph:
    """Directed multigraph collapsed to a DiGraph with aggregated edge state."""
    G = nx.DiGraph()
    G.add_nodes_from(set(slice_df["src"]) | set(slice_df["dst"]))
    for row in slice_df.itertuples(index=False):
        if G.has_edge(row.src, row.dst):
            e = G[row.src][row.dst]
            e["count"] += 1
            e["amount"] += float(row.amount)
            e["rating"] = min(e["rating"], float(row.rating))
            e["t_first"] = min(e["t_first"], int(row.timestamp))
            e["t_last"] = max(e["t_last"], int(row.timestamp))
            e["tokens"].add(row.token)
        else:
            G.add_edge(row.src, row.dst,
                       weight=int(row.timestamp),
                       count=1,
                       amount=float(row.amount),
                       rating=float(row.rating),
                       t_first=int(row.timestamp),
                       t_last=int(row.timestamp),
                       tokens={row.token},
                       src_is_contract=int(row.src_is_contract),
                       dst_is_contract=int(row.dst_is_contract))
    return G


def calculate_anomaly_and_antianomaly_scores(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    """Centrality- and degree-based anomaly / anti-anomaly scores (unchanged)."""
    try:
        base_anomaly = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        base_anomaly = nx.degree_centrality(G)
    base_antianomaly = nx.out_degree_centrality(G)

    in_deg, out_deg = dict(G.in_degree()), dict(G.out_degree())
    max_in = max(in_deg.values()) or 1
    max_out = max(out_deg.values()) or 1
    in_norm = {n: v / max_in for n, v in in_deg.items()}
    out_norm = {n: v / max_out for n, v in out_deg.items()}

    anomaly = {n: base_anomaly.get(n, 0.0) + out_norm.get(n, 0.0) for n in G}
    antianomaly = {n: base_antianomaly.get(n, 0.0) + in_norm.get(n, 0.0) for n in G}
    return anomaly, antianomaly


def label_nodes(anomaly_scores: Dict, antianomaly_scores: Dict,
                percentile: float = 80.0) -> Dict:
    """Adaptive percentile labelling; top (100-percentile)% flagged abnormal."""
    a_thr = np.percentile(list(anomaly_scores.values()), percentile)
    p_thr = np.percentile(list(antianomaly_scores.values()), percentile)

    labels = {}
    for node in anomaly_scores:
        coll = "collection_abnormal" if anomaly_scores[node] > a_thr else "collection_normal"
        pay = "payment_normal" if antianomaly_scores[node] > p_thr else "payment_abnormal"
        labels[node] = (coll, pay)

    n_abn = sum(1 for v in labels.values() if v[0] == "collection_abnormal")
    if n_abn == 0 or n_abn == len(labels):
        keys = list(labels)
        labels[keys[0]] = ("collection_abnormal", labels[keys[0]][1])
        if len(keys) > 1:
            labels[keys[-1]] = ("collection_normal", labels[keys[-1]][1])
    return labels


def annotate_edges(G: nx.DiGraph, labels: Dict) -> None:
    """Write collection_* / payment_* markers onto edges.

    FIX relative to the original pipeline: `count_edges` tested for these keys
    in the edge attribute dictionary, but no code ever wrote them, so the four
    reachability counts P1-P4 were identically zero for every node. Half of the
    eight-dimensional feature vector carried no signal.
    """
    for u, v, data in G.edges(data=True):
        recv = labels.get(v, ("collection_normal", "payment_abnormal"))[0]
        send = labels.get(u, ("collection_normal", "payment_abnormal"))[1]
        data[recv] = True
        data["payment_normal" if send == "payment_normal" else "payment_abnormal"] = True


# ===========================================================================
# 3. Node features
# ===========================================================================

def reachability_subgraph(G: nx.DiGraph, node, max_depth: int, max_nodes: int) -> nx.DiGraph:
    sub = nx.DiGraph()
    sub.add_node(node)
    frontier = {node}
    for _ in range(max_depth):
        nxt = set()
        for u in frontier:
            for v in G.successors(u):
                if v not in sub:
                    sub.add_edge(u, v, **G[u][v])
                    nxt.add(v)
                    if sub.number_of_nodes() >= max_nodes:
                        return sub
        if not nxt:
            break
        frontier = nxt
    return sub


def build_reachability_networks(G: nx.DiGraph, max_depth: int, max_nodes: int) -> Dict:
    """Compute every node's reachability subgraph once.

    FIX relative to the original pipeline: `extract_features` rebuilt the full
    set of reachability subgraphs inside its per-node loop, making feature
    extraction O(N^2) in the number of nodes.
    """
    return {n: reachability_subgraph(G, n, max_depth, max_nodes) for n in G}


def count_labelled_edges(sub: nx.DiGraph, key: str) -> int:
    return sum(1 for _, _, d in sub.edges(data=True) if key in d)


NODE_FEATURE_DIM = 16


def extract_node_features(G: nx.DiGraph, reach: Dict) -> Tuple[np.ndarray, Dict]:
    """Sixteen behavioural / structural descriptors per address."""
    nodes = list(G.nodes())
    t_all = [d["t_last"] for _, _, d in G.edges(data=True)]
    t_span_global = (max(t_all) - min(t_all)) if len(t_all) > 1 else 1.0
    t_span_global = max(t_span_global, 1.0)

    feats, stats = [], {}
    for n in nodes:
        sub = reach[n]
        p1 = count_labelled_edges(sub, "collection_normal")
        p2 = count_labelled_edges(sub, "collection_abnormal")
        p3 = count_labelled_edges(sub, "payment_normal")
        p4 = count_labelled_edges(sub, "payment_abnormal")

        out_edges = list(G.out_edges(n, data=True))
        in_edges = list(G.in_edges(n, data=True))
        out_val = sum(d["amount"] for _, _, d in out_edges)
        in_val = sum(d["amount"] for _, _, d in in_edges)
        denom = out_val + in_val + 1e-9

        succ = set(G.successors(n))
        pred = set(G.predecessors(n))
        recip = len(succ & pred) / (len(succ | pred) + 1e-9)

        times = [d["t_last"] for _, _, d in out_edges + in_edges]
        span = (max(times) - min(times)) if len(times) > 1 else 0.0
        burst = len(times) / (span / 3600.0 + 1.0)
        neg_frac = (sum(1 for _, _, d in out_edges + in_edges if d["rating"] < 0)
                    / (len(out_edges) + len(in_edges) + 1e-9))

        f = [
            float(p1), float(p2), float(p3), float(p4),
            float(len(succ)),
            float(sub.number_of_nodes()),
            float(G.in_degree(n)),
            float(G.out_degree(n)),
            math.log1p(in_val),
            math.log1p(out_val),
            (out_val - in_val) / denom,
            float(len(pred)),
            float(len(succ)),
            span / t_span_global,
            math.log1p(burst),
            neg_frac,
        ]
        feats.append(f)
        stats[n] = {"in_val": in_val, "out_val": out_val, "neg_frac": neg_frac,
                    "in_deg": G.in_degree(n), "out_deg": G.out_degree(n),
                    "burst": burst}

    X = np.asarray(feats, dtype=np.float32)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sd, stats


# ===========================================================================
# 4. Knowledge graph construction (Sec. 3)
# ===========================================================================

def assign_patterns(G: nx.DiGraph, stats: Dict) -> Dict:
    """Map each address to one anomaly archetype (the Attack Type analogue)."""
    out_deg = np.array([stats[n]["out_deg"] for n in G])
    in_deg = np.array([stats[n]["in_deg"] for n in G])
    burst = np.array([stats[n]["burst"] for n in G])
    hi_out = np.percentile(out_deg, 85) if len(out_deg) else 0
    hi_in = np.percentile(in_deg, 85) if len(in_deg) else 0
    hi_burst = np.percentile(burst, 90) if len(burst) else 0

    pattern = {}
    for n in G:
        s = stats[n]
        if s["burst"] > hi_burst and hi_burst > 0:
            pattern[n] = "burst"
        elif s["out_deg"] > hi_out and s["in_deg"] > hi_in:
            pattern[n] = "relay"
        elif s["out_deg"] > hi_out:
            pattern[n] = "dispersion"
        else:
            pattern[n] = "collection"
    return pattern


def build_knowledge_graph(G: nx.DiGraph, labels: Dict, stats: Dict,
                          node_to_idx: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Materialise G_K = (V_K, E_K, R_K).

    Wallet entities occupy indices 0..N-1 and are index-aligned with the
    transaction graph, so the KGNN output rows for wallets can be read off
    directly. Auxiliary entities are appended after them.

    Returns (kg_x, kg_edge_index, kg_edge_type, num_kg_nodes).
    """
    n_wallets = len(node_to_idx)
    kg_index: Dict[Tuple[str, str], int] = {}
    kg_type: List[int] = []
    cursor = n_wallets

    def entity(kind: str, key: str) -> int:
        nonlocal cursor
        k = (kind, key)
        if k not in kg_index:
            kg_index[k] = cursor
            kg_type.append(ENTITY_TYPES[kind])
            cursor += 1
        return kg_index[k]

    wallet_types = [ENTITY_TYPES["WALLET"]] * n_wallets
    edges: List[Tuple[int, int]] = []
    etypes: List[int] = []

    def link(s: int, d: int, rel: str) -> None:
        edges.append((s, d))
        etypes.append(RELATION_TYPES[rel])

    # --- contracts, hubs, patterns, tokens ---------------------------------
    contract_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("src_is_contract"):
            contract_nodes.add(u)
        if d.get("dst_is_contract"):
            contract_nodes.add(v)
    for n in contract_nodes:
        wallet_types[node_to_idx[n]] = ENTITY_TYPES["CONTRACT"]

    deg_total = np.array([stats[n]["in_deg"] + stats[n]["out_deg"] for n in G])
    hub_thr = np.percentile(deg_total, 97) if len(deg_total) else np.inf
    hub_nodes = {n for n in G if stats[n]["in_deg"] + stats[n]["out_deg"] >= hub_thr}

    patterns = assign_patterns(G, stats)
    pattern_ids = {p: entity("PATTERN", p) for p in PATTERN_NAMES}

    # (Wallet_1, transfers, Wallet_2) and (Wallet, interacts, Contract)
    for u, v, d in G.edges(data=True):
        su, sv = node_to_idx[u], node_to_idx[v]
        link(su, sv, "transfers")
        if v in contract_nodes:
            link(su, sv, "interacts_with")
        for tok in d["tokens"]:
            tid = entity("TOKEN", str(tok))
            link(su, tid, "uses_token")
            link(sv, tid, "uses_token")
        if v in hub_nodes:
            link(su, entity("HUB", v), "routes_through")
        if u in hub_nodes:
            link(sv, entity("HUB", u), "routes_through")

    # (Wallet, exhibits, Pattern)
    for n in G:
        link(node_to_idx[n], pattern_ids[patterns[n]], "exhibits")

    # --- anomaly events: connected components of the abnormal sub-network ---
    abnormal = {n for n in G if labels.get(n, ("collection_normal",))[0] == "collection_abnormal"}
    if abnormal:
        sub = G.subgraph(abnormal).to_undirected()
        for cid, comp in enumerate(nx.connected_components(sub)):
            ev = entity("ANOMALY_EVENT", f"event_{cid}")
            for n in comp:
                link(node_to_idx[n], ev, "belongs_to")
            # (AnomalyEvent, associated_with, Pattern)
            dominant = defaultdict(int)
            for n in comp:
                dominant[patterns[n]] += 1
            link(ev, pattern_ids[max(dominant, key=dominant.get)], "associated_with")

    # --- co_counterparty: wallets sharing a counterparty (multi-hop link) ---
    shared = defaultdict(list)
    for n in G:
        for v in G.successors(n):
            shared[v].append(node_to_idx[n])
    for _, senders in shared.items():
        if 2 <= len(senders) <= 40:
            for i in range(len(senders)):
                for j in range(i + 1, len(senders)):
                    link(senders[i], senders[j], "co_counterparty")
                    link(senders[j], senders[i], "co_counterparty")

    num_kg_nodes = cursor
    type_ids = np.array(wallet_types + kg_type, dtype=np.int64)
    kg_x = np.zeros((num_kg_nodes, NUM_ENTITY_TYPES), dtype=np.float32)
    kg_x[np.arange(num_kg_nodes), type_ids] = 1.0

    if edges:
        kg_edge_index = np.asarray(edges, dtype=np.int64).T
        kg_edge_type = np.asarray(etypes, dtype=np.int64)
    else:
        kg_edge_index = np.zeros((2, 0), dtype=np.int64)
        kg_edge_type = np.zeros((0,), dtype=np.int64)

    return kg_x, kg_edge_index, kg_edge_type, num_kg_nodes


# ===========================================================================
# 5. PyG data assembly
# ===========================================================================

class AnomalyGraphData(Data):
    """Data object carrying both the transaction graph and its knowledge graph."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "kg_edge_index":
            return int(self.num_kg_nodes)
        if key == "wallet_kg_index":
            return int(self.num_kg_nodes)
        if key in ("kg_edge_type", "edge_time", "edge_risk"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def create_edge_labels(G, labels, edge_index, node_to_idx, rule: str = "and") -> torch.Tensor:
    """Propagate node-level weak labels onto transactions.

    rule='or'   a transaction is anomalous if either endpoint is abnormal.
                This is the original rule. With 20% of addresses flagged, it
                marks 70-85% of transactions anomalous on both datasets, so a
                constant all-positive predictor already reaches F1 ~ 0.85 and
                the headline scores mostly reflect the base rate.
    rule='and'  both endpoints abnormal. Yields a minority positive class and
                a metric range that is actually informative. Default here.
    rule='dst'  the receiving address is abnormal (collection-side view).
    """
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    out = []
    for i in range(edge_index.size(1)):
        u = idx_to_node[int(edge_index[0, i])]
        v = idx_to_node[int(edge_index[1, i])]
        su = labels.get(u, ("collection_normal",))[0] == "collection_abnormal"
        sv = labels.get(v, ("collection_normal",))[0] == "collection_abnormal"
        if rule == "or":
            out.append(float(su or sv))
        elif rule == "and":
            out.append(float(su and sv))
        else:
            out.append(float(sv))
    if out and all(x == 0.0 for x in out):
        out[0] = 1.0
    elif out and all(x == 1.0 for x in out):
        out[0] = 0.0
    return torch.tensor(out, dtype=torch.float)


def build_slice_data(G: nx.DiGraph, labels: Dict, slice_id: int,
                     anomaly_scores: Dict, max_depth: int,
                     max_reach_nodes: int,
                     edge_rule: str = "and") -> Optional[AnomalyGraphData]:
    if G.number_of_edges() == 0:
        return None

    annotate_edges(G, labels)
    node_to_idx = {n: i for i, n in enumerate(G.nodes())}
    reach = build_reachability_networks(G, max_depth, max_reach_nodes)
    X, stats = extract_node_features(G, reach)

    ei = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()],
                      dtype=torch.long).t().contiguous()
    y = create_edge_labels(G, labels, ei, node_to_idx, rule=edge_rule)

    edge_time = torch.tensor([G[u][v]["t_last"] for u, v in G.edges()], dtype=torch.float)

    # Risk prior per transaction (feeds S_risk in Sec. 5 and F_i in Sec. 6.2).
    a_vals = np.array(list(anomaly_scores.values()), dtype=np.float32)
    a_lo, a_hi = float(a_vals.min()), float(a_vals.max() + 1e-9)
    risk = []
    for u, v in G.edges():
        base = max((anomaly_scores.get(u, 0.0) - a_lo) / (a_hi - a_lo),
                   (anomaly_scores.get(v, 0.0) - a_lo) / (a_hi - a_lo))
        neg = 1.0 if G[u][v]["rating"] < 0 else 0.0
        risk.append(min(1.0, 0.75 * base + 0.25 * neg))
    edge_risk = torch.tensor(risk, dtype=torch.float)

    kg_x, kg_ei, kg_et, n_kg = build_knowledge_graph(G, labels, stats, node_to_idx)

    return AnomalyGraphData(
        x=torch.from_numpy(X),
        edge_index=ei,
        y=y,
        edge_time=edge_time,
        edge_risk=edge_risk,
        kg_x=torch.from_numpy(kg_x),
        kg_edge_index=torch.from_numpy(kg_ei),
        kg_edge_type=torch.from_numpy(kg_et),
        num_kg_nodes=torch.tensor(n_kg),
        wallet_kg_index=torch.arange(len(node_to_idx), dtype=torch.long),
        slice_id=torch.tensor(slice_id),
    )


# ===========================================================================
# 6. Encoders
# ===========================================================================

class MGGRU(nn.Module):
    """Multi-graph gated recurrent encoder over the transaction network."""

    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        # concat=False keeps the output width at hidden_dim across all heads.
        gat = lambda: GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.gat_hidden = gat()
        self.gat_update_x, self.gat_update_h = gat(), gat()
        self.gat_reset_x, self.gat_reset_h = gat(), gat()
        self.gat_candidate_x, self.gat_candidate_h = gat(), gat()

        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing_info(self, h_v, h_N_v):
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        r = torch.zeros_like(h_v)
        return h_v + ((gamma + 1) * r + beta) - h_N_v

    def forward(self, x, edge_index, h=None):
        xt = self.input_transform(x)
        if h is None:
            h = torch.zeros(xt.size(0), self.hidden_dim, device=x.device)
        h_N = self.gat_hidden(h, edge_index)
        h_N = h_N + self.predict_missing_info(h, h_N)
        r = torch.sigmoid(self.gat_reset_x(xt, edge_index) + self.gat_reset_h(h_N, edge_index))
        z = torch.sigmoid(self.gat_update_x(xt, edge_index) + self.gat_update_h(h_N, edge_index))
        h_tilde = torch.tanh(self.gat_candidate_x(xt, edge_index)
                             + r * self.gat_candidate_h(h_N, edge_index))
        return (1 - z) * h + z * h_tilde


class RelationalGATLayer(nn.Module):
    """Attention-weighted relational convolution (Sec. 4).

        h_i^{(l+1)} = sigma( sum_r sum_{j in N_i^r} alpha_ij^r W_r h_j^{(l)} )

    One GATConv per relation type supplies both W_r and alpha_ij^r; the outputs
    are summed over relations and passed through a nonlinearity.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, heads: int = 1):
        super().__init__()
        self.num_relations = num_relations
        self.rel_convs = nn.ModuleList(
            [GATConv(in_dim, out_dim, heads=heads, concat=False, add_self_loops=False)
             for _ in range(num_relations)]
        )
        self.self_loop = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        out = self.self_loop(x)
        for r in range(self.num_relations):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            out = out + self.rel_convs[r](x, edge_index[:, mask])
        return F.elu(out)


class KGNN(nn.Module):
    """Knowledge graph representation learner producing Z_K = KGNN(G_K)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 num_relations: int = NUM_RELATIONS):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [RelationalGATLayer(dims[i], dims[i + 1], num_relations)
             for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, kg_x, kg_edge_index, kg_edge_type):
        h = kg_x
        for layer in self.layers:
            h = layer(h, kg_edge_index, kg_edge_type)
        return self.norm(h)


# ===========================================================================
# 7. SaR-RAG: selective retrieval and refined anomaly memory (Secs. 5-6)
# ===========================================================================

class RefinedAnomalyMemory:
    """Dynamic anomaly memory with redundancy removal, reliability filtering,
    and exponential-moving-average consolidation (Sec. 6).

    Each memory item stores a semantic vector (transaction behaviour), a
    structural vector (KG context), an observation time, a risk score, and a
    confidence score. Only anomalous transactions are admitted.
    """

    def __init__(self, max_size: int = 2000, epsilon: float = 0.08,
                 merge_threshold: float = 0.97, eta: float = 0.3,
                 conf_alpha: float = 0.5, conf_beta: float = 0.3,
                 conf_gamma: float = 0.2, conf_floor: float = 0.15):
        self.max_size = max_size
        self.epsilon = epsilon              # redundancy radius, Sec. 6.1
        self.merge_threshold = merge_threshold
        self.eta = eta                      # update rate, Sec. 6.3
        self.alpha, self.beta, self.gamma = conf_alpha, conf_beta, conf_gamma
        self.conf_floor = conf_floor
        self.reset()

    def reset(self):
        self.sem: Optional[torch.Tensor] = None
        self.kg: Optional[torch.Tensor] = None
        self.time: Optional[torch.Tensor] = None
        self.risk: Optional[torch.Tensor] = None
        self.conf: Optional[torch.Tensor] = None
        self.slice_id: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self.sem is None else int(self.sem.size(0))

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    # -- Sec. 6.2: C_i = alpha*F_i + beta*R_i + gamma*T_i --------------------
    def _confidence(self, risk: torch.Tensor, relevance: torch.Tensor,
                    time: torch.Tensor, t_ref: float, tau: float) -> torch.Tensor:
        temporal = torch.exp(-(t_ref - time).clamp(min=0) / tau)
        return self.alpha * risk + self.beta * relevance + self.gamma * temporal

    def add(self, sem, kg, time, risk, slice_id, t_ref: float, tau: float):
        """Insert anomaly cases, merging near-duplicates by EMA (Sec. 6.3)."""
        if sem.numel() == 0:
            return
        sem = F.normalize(sem.detach(), dim=1)
        kg = F.normalize(kg.detach(), dim=1)
        time, risk = time.detach(), risk.detach()
        relevance = risk.clone()  # bootstrap R_i before any retrieval feedback
        conf = self._confidence(risk, relevance, time, t_ref, tau)

        if self.is_empty:
            self.sem, self.kg = sem, kg
            self.time, self.risk, self.conf = time, risk, conf
            self.slice_id = slice_id.detach()
            self._prune()
            return

        sims = sem @ self.sem.t()                      # [n_new, n_mem]
        best, idx = sims.max(dim=1)
        merge = best >= self.merge_threshold
        keep = ~merge

        if bool(merge.any()):
            tgt = idx[merge]
            eta = self.eta
            # M_{t+1} = (1 - eta) M_t + eta M_new
            self.sem[tgt] = F.normalize((1 - eta) * self.sem[tgt] + eta * sem[merge], dim=1)
            self.kg[tgt] = F.normalize((1 - eta) * self.kg[tgt] + eta * kg[merge], dim=1)
            self.time[tgt] = torch.maximum(self.time[tgt], time[merge])
            self.risk[tgt] = (1 - eta) * self.risk[tgt] + eta * risk[merge]
            self.conf[tgt] = (1 - eta) * self.conf[tgt] + eta * conf[merge]

        if bool(keep.any()):
            self.sem = torch.cat([self.sem, sem[keep]])
            self.kg = torch.cat([self.kg, kg[keep]])
            self.time = torch.cat([self.time, time[keep]])
            self.risk = torch.cat([self.risk, risk[keep]])
            self.conf = torch.cat([self.conf, conf[keep]])
            self.slice_id = torch.cat([self.slice_id, slice_id.detach()[keep]])
        self._prune()

    def _prune(self):
        """Reliability filtering (Sec. 6.2) followed by capacity enforcement."""
        if self.is_empty:
            return
        keep = self.conf >= self.conf_floor
        if int(keep.sum()) >= 16:
            self._index(keep.nonzero(as_tuple=True)[0])
        if len(self) > self.max_size:
            order = torch.argsort(self.conf, descending=True)[: self.max_size]
            self._index(order)

    def _index(self, idx: torch.Tensor):
        self.sem, self.kg = self.sem[idx], self.kg[idx]
        self.time, self.risk = self.time[idx], self.risk[idx]
        self.conf, self.slice_id = self.conf[idx], self.slice_id[idx]

    def stats(self) -> str:
        if self.is_empty:
            return "memory: empty"
        return (f"memory: {len(self)} cases | mean risk {float(self.risk.mean()):.3f} "
                f"| mean confidence {float(self.conf.mean()):.3f}")


class SaRRAG(nn.Module):
    """Selective retrieval + refinement, returning an anomaly evidence vector.

    Retrieval score (Sec. 5):
        S_i = l1*S_sem + l2*S_struct + l3*S_temp + l4*S_risk

    Refinement (Sec. 6): greedy selection of the retrieved candidate pool that
    maximises Q(M) - gamma*R(M), dropping any case whose distance
    D_ij = 1 - cos(m_i, m_j) falls below epsilon.
    """

    def __init__(self, sem_dim: int, kg_dim: int, evidence_dim: int,
                 top_k: int = 8, pool_mult: int = 4, tau_days: float = 30.0,
                 redundancy_eps: float = 0.08, redundancy_gamma: float = 0.5,
                 learn_lambda: bool = True, use_kg: bool = True,
                 use_selective: bool = True, use_refinement: bool = True):
        super().__init__()
        self.top_k = top_k
        self.pool_size = top_k * pool_mult
        self.tau = tau_days * 86400.0
        self.eps = redundancy_eps
        self.gamma = redundancy_gamma
        self.use_kg = use_kg
        self.use_selective = use_selective
        self.use_refinement = use_refinement

        # lambda_1..lambda_4, kept on a simplex via softmax
        init = torch.tensor([1.2, 0.9, 0.4, 0.5])
        self.lambda_logits = nn.Parameter(init, requires_grad=learn_lambda)

        self.evidence_proj = nn.Sequential(
            nn.Linear(sem_dim + kg_dim + 4, evidence_dim),
            nn.ReLU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.evidence_dim = evidence_dim

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.softmax(self.lambda_logits, dim=0)

    def _score(self, q_sem, q_kg, q_time, memory: RefinedAnomalyMemory):
        lam = self.lambdas
        s_sem = q_sem @ memory.sem.t()                                   # S_sem
        if self.use_kg:
            s_struct = q_kg @ memory.kg.t()                              # S_struct
        else:
            s_struct = torch.zeros_like(s_sem)
        dt = (q_time.unsqueeze(1) - memory.time.unsqueeze(0)).abs()
        s_temp = torch.exp(-dt / self.tau)                               # S_temp
        s_risk = memory.risk.unsqueeze(0).expand_as(s_sem)               # S_risk

        if not self.use_selective:
            return s_sem, s_sem, s_struct, s_temp, s_risk
        total = (lam[0] * s_sem + lam[1] * s_struct
                 + lam[2] * s_temp + lam[3] * s_risk)
        return total, s_sem, s_struct, s_temp, s_risk

    def _refine(self, cand_idx, cand_score, memory):
        """Greedy Q(M) - gamma*R(M) maximisation over the candidate pool."""
        n_q, pool = cand_idx.shape
        k = min(self.top_k, pool)
        sel_idx = torch.zeros(n_q, k, dtype=torch.long, device=cand_idx.device)
        sel_w = torch.zeros(n_q, k, device=cand_score.device)

        pool_vecs = memory.sem[cand_idx]                        # [n_q, pool, d]
        pool_conf = memory.conf[cand_idx]                       # [n_q, pool]
        cross = torch.bmm(pool_vecs, pool_vecs.transpose(1, 2))  # cosine (unit norm)
        dist = 1.0 - cross                                       # D_ij

        quality = cand_score * pool_conf                         # Q contribution
        chosen_mask = torch.zeros(n_q, pool, dtype=torch.bool, device=cand_idx.device)
        redundancy = torch.zeros(n_q, pool, device=cand_score.device)

        for step in range(k):
            obj = quality - self.gamma * redundancy
            obj = obj.masked_fill(chosen_mask, float("-inf"))
            if step > 0:
                # Suppress candidate j when some already-selected case i sits
                # within the redundancy radius: D_ij = 1 - cos(m_i, m_j) < eps.
                # dim 1 indexes the selected case, dim 2 the candidate, so the
                # reduction must run over dim 1 to land back on candidates.
                too_close = (dist < self.eps) & chosen_mask.unsqueeze(-1)
                obj = obj.masked_fill(too_close.any(dim=1), float("-inf"))
            best = obj.argmax(dim=1)
            rows = torch.arange(n_q, device=cand_idx.device)
            # if every candidate was suppressed, fall back to the top-scoring one
            dead = torch.isinf(obj.max(dim=1).values)
            if bool(dead.any()):
                best[dead] = cand_score[dead].argmax(dim=1)
            chosen_mask[rows, best] = True
            sel_idx[:, step] = cand_idx[rows, best]
            sel_w[:, step] = cand_score[rows, best].clamp(min=0)
            redundancy = torch.maximum(redundancy, cross[rows, best, :])
        return sel_idx, sel_w

    def forward(self, q_sem, q_kg, q_time, memory: RefinedAnomalyMemory,
                exclude_slice: Optional[torch.Tensor] = None,
                chunk: int = 512):
        n = q_sem.size(0)
        if memory.is_empty:
            zeros = torch.zeros(n, self.evidence_dim, device=q_sem.device)
            return zeros, torch.zeros(n, 4, device=q_sem.device)

        q_sem_n = F.normalize(q_sem, dim=1)
        q_kg_n = F.normalize(q_kg, dim=1)
        evid, diag = [], []

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            qs, qk, qt = q_sem_n[start:end], q_kg_n[start:end], q_time[start:end]
            total, s_sem, _, _, _ = self._score(qs, qk, qt, memory)

            if exclude_slice is not None:
                # block retrieval of cases originating from the querying slice
                blocked = exclude_slice[start:end].unsqueeze(1) == memory.slice_id.unsqueeze(0)
                total = total.masked_fill(blocked, float("-inf"))
                if bool(torch.isinf(total).all(dim=1).any()):
                    total = torch.where(torch.isinf(total).all(dim=1, keepdim=True),
                                        s_sem, total)

            pool = min(self.pool_size, total.size(1))
            cand_score, cand_idx = torch.topk(total, pool, dim=1)      # M_k = TopK(S_i)

            if self.use_refinement and pool > self.top_k:
                sel_idx, sel_w = self._refine(cand_idx, cand_score, memory)
            else:
                k = min(self.top_k, pool)
                sel_idx, sel_w = cand_idx[:, :k], cand_score[:, :k].clamp(min=0)

            w = torch.softmax(sel_w, dim=1).unsqueeze(-1)
            ev_sem = (memory.sem[sel_idx] * w).sum(1)
            ev_kg = (memory.kg[sel_idx] * w).sum(1)
            r_sel = memory.risk[sel_idx]
            c_sel = memory.conf[sel_idx]
            summary = torch.stack([
                sel_w.max(dim=1).values,
                sel_w.mean(dim=1),
                r_sel.mean(dim=1),
                c_sel.mean(dim=1),
            ], dim=1)
            evid.append(self.evidence_proj(torch.cat([ev_sem, ev_kg, summary], dim=1)))
            diag.append(summary)

        return torch.cat(evid, 0), torch.cat(diag, 0)


# ===========================================================================
# 8. Full detector (Sec. 7)
# ===========================================================================

class SaRRAGAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, kg_dim: int = 32,
                 evidence_dim: int = 32, fusion_dim: int = 64, num_layers: int = 2,
                 top_k: int = 8, tau_days: float = 30.0, ablation: str = "full"):
        super().__init__()
        self.ablation = ablation
        self.num_layers = num_layers
        self.use_kg = ablation != "no_kg"
        self.use_memory = ablation != "no_memory"

        self.mggru = MGGRU(input_dim, hidden_dim)
        self.kgnn = KGNN(NUM_ENTITY_TYPES, kg_dim, num_layers=2)

        self.sarrag = SaRRAG(
            sem_dim=2 * hidden_dim, kg_dim=2 * kg_dim, evidence_dim=evidence_dim,
            top_k=top_k, tau_days=tau_days,
            use_kg=self.use_kg,
            use_selective=ablation != "no_selective",
            use_refinement=ablation != "no_refinement",
        )

        # H_F = W_f [ H_G || H_K || M ]
        fuse_in = 2 * hidden_dim + (2 * kg_dim if self.use_kg else 0) \
            + (evidence_dim if self.use_memory else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fusion_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(fusion_dim, 1)  # P(y) = sigma(W_c H_F + b)

    def encode(self, data):
        """Return per-transaction semantic (H_G) and structural (H_K) embeddings."""
        h = None
        for _ in range(self.num_layers):
            h = self.mggru(data.x, data.edge_index, h)
        src, dst = data.edge_index[0], data.edge_index[1]
        h_g = torch.cat([h[src], h[dst]], dim=1)

        if self.use_kg:
            z = self.kgnn(data.kg_x, data.kg_edge_index, data.kg_edge_type)
            z_wallet = z[data.wallet_kg_index]
            h_k = torch.cat([z_wallet[src], z_wallet[dst]], dim=1)
        else:
            h_k = torch.zeros(h_g.size(0), 0, device=h_g.device)
        return h_g, h_k

    def forward(self, data, memory: RefinedAnomalyMemory,
                exclude_slice: Optional[torch.Tensor] = None):
        h_g, h_k = self.encode(data)

        parts = [h_g]
        if self.use_kg:
            parts.append(h_k)

        if self.use_memory:
            kg_query = h_k if self.use_kg else h_g
            evidence, diag = self.sarrag(h_g, kg_query, data.edge_time, memory,
                                         exclude_slice=exclude_slice)
            parts.append(evidence)
        else:
            diag = torch.zeros(h_g.size(0), 4, device=h_g.device)

        h_f = self.fusion(torch.cat(parts, dim=1))
        logits = self.classifier(h_f).squeeze(-1)
        return logits, diag


# ===========================================================================
# 9. Training and evaluation
# ===========================================================================

def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-weighted objective; pos_weight = N_neg / N_pos within the batch."""
    n_pos = float(targets.sum())
    n_neg = float(targets.numel() - n_pos)
    pw = torch.tensor(max(n_neg / max(n_pos, 1.0), 1.0), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


@torch.no_grad()
def rebuild_memory(model: SaRRAGAnomalyDetector, memory: RefinedAnomalyMemory,
                   loader, tau_seconds: float):
    """Re-encode the training slices and repopulate the anomaly memory.

    The memory index is refreshed once per epoch because the encoders change
    during training; a memory populated with stale embeddings drifts away from
    the query space it is supposed to serve. Only training slices contribute,
    so no test-time information enters the memory.
    """
    model.eval()
    memory.reset()
    t_ref = 0.0
    for data in loader:
        if data.edge_time.numel():
            t_ref = max(t_ref, float(data.edge_time.max()))
    for data in loader:
        data = data.to(DEVICE)
        h_g, h_k = model.encode(data)
        mask = data.y > 0.5
        if not bool(mask.any()):
            continue
        kg_vec = h_k if model.use_kg else h_g
        sid = data.slice_id[data.batch[data.edge_index[0]]] if hasattr(data, "batch") \
            else data.slice_id.expand(h_g.size(0))
        memory.add(h_g[mask], kg_vec[mask], data.edge_time[mask],
                   data.edge_risk[mask], sid[mask], t_ref, tau_seconds)
    model.train()


def train_model(model, train_loader, memory, memory_loader=None, epochs=60, lr=5e-3,
                refresh_every=1, tau_days=30.0, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tau_seconds = tau_days * 86400.0
    # Memory consolidation (Sec. 6.3) is a streaming update, so the resulting
    # bank depends on insertion order. Refreshing from an unshuffled loader
    # keeps the index reproducible and identical to the one used at test time.
    memory_loader = memory_loader or train_loader

    for epoch in range(epochs):
        if model.use_memory and epoch % refresh_every == 0:
            rebuild_memory(model, memory, memory_loader, tau_seconds)

        total, nb = 0.0, 0
        for data in train_loader:
            data = data.to(DEVICE)
            opt.zero_grad()
            excl = None
            if model.use_memory and not memory.is_empty:
                excl = data.slice_id[data.batch[data.edge_index[0]]]
            logits, _ = model(data, memory, exclude_slice=excl)
            loss = weighted_bce_loss(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.detach().item()
            nb += 1
        sched.step()

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            lam = model.sarrag.lambdas.detach().cpu().numpy()
            print(f"  epoch {epoch+1:3d}/{epochs}  loss {total/max(nb,1):.4f}  "
                  f"lambda=[{lam[0]:.2f} {lam[1]:.2f} {lam[2]:.2f} {lam[3]:.2f}]  "
                  f"{memory.stats()}")
    return model


@torch.no_grad()
def evaluate_model(model, loader, memory, tag="BOTC", outdir="figures", plots=True):
    model.eval()
    probs, labels, diags = [], [], []
    for data in loader:
        data = data.to(DEVICE)
        logits, diag = model(data, memory)          # no slice exclusion at test time
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(data.y.cpu().numpy())
        diags.append(diag.cpu().numpy())

    p = np.concatenate(probs)
    y = np.concatenate(labels)
    d = np.concatenate(diags)
    binary = (p > 0.5).astype(int)

    auc = roc_auc_score(y, p)
    prec = precision_score(y, binary, zero_division=0)
    rec = recall_score(y, binary, zero_division=0)
    f1 = f1_score(y, binary, zero_division=0)

    pv, rv, thr = precision_recall_curve(y, p)
    f1s = 2 * pv * rv / (pv + rv + 1e-12)
    best = int(np.nanargmax(f1s))
    best_thr = float(thr[min(best, len(thr) - 1)])

    print(f"\n  AUC={auc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  best-threshold F1={f1s[best]:.4f} at t={best_thr:.3f} "
          f"(P={pv[best]:.4f}, R={rv[best]:.4f})")
    print(f"  retrieved-evidence diagnostics: mean top-1 score {d[:,0].mean():.3f}, "
          f"mean retrieved risk {d[:,2].mean():.3f}, "
          f"mean retrieved confidence {d[:,3].mean():.3f}")

    if plots:
        os.makedirs(outdir, exist_ok=True)
        cm = confusion_matrix(y, binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Anomalous"],
                    yticklabels=["Normal", "Anomalous"], annot_kws={"size": 18})
        plt.xlabel("Predicted", fontsize=18)
        plt.ylabel("True", fontsize=18)
        plt.title(f"Confusion Matrix -- SaR-RAG ({tag})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{outdir}/confusion_{tag}.png", dpi=150)
        plt.close()

        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"SaR-RAG (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate", fontsize=16)
        plt.ylabel("True Positive Rate", fontsize=16)
        plt.title(f"ROC Curve -- SaR-RAG ({tag})", fontsize=16)
        plt.legend(loc="lower right", fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/roc_{tag}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(rv, pv, color="purple", lw=2)
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.title(f"Precision-Recall Curve -- SaR-RAG ({tag})", fontsize=16)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/pr_{tag}.png", dpi=150)
        plt.close()
        print(f"  figures written to {outdir}/")

    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1,
            "best_f1": float(f1s[best]), "best_threshold": best_thr}


# ===========================================================================
# 10. Pipeline
# ===========================================================================

def build_dataset(name: str, path: str, max_depth: int, max_reach_nodes: int,
                  edge_rule: str = "and", label_percentile: float = 80.0,
                  verbose: bool = True):
    df, meta = load_dataset(name, path)
    df = temporal_slices(df, meta["slice_days"])
    if verbose:
        print(f"loaded {name}: {len(df)} transactions, "
              f"{df['time_slice'].nunique()} temporal slices, "
              f"edge rule '{edge_rule}'")

    data_list = []
    for sid in sorted(df["time_slice"].unique()):
        sl = df[df["time_slice"] == sid]
        G = create_graph(sl)
        if G.number_of_edges() == 0:
            continue
        a_scores, anti = calculate_anomaly_and_antianomaly_scores(G)
        labels = label_nodes(a_scores, anti, percentile=label_percentile)
        d = build_slice_data(G, labels, int(sid), a_scores, max_depth,
                             max_reach_nodes, edge_rule=edge_rule)
        if d is not None:
            data_list.append(d)
            if verbose:
                print(f"  slice {sid:>3}: {G.number_of_nodes():>5} wallets, "
                      f"{G.number_of_edges():>6} transactions, "
                      f"{int(d.num_kg_nodes)} KG entities, "
                      f"{d.kg_edge_index.size(1)} KG triples, "
                      f"anomaly rate {float(d.y.mean()):.3f}")
    if verbose and data_list:
        rate = float(torch.cat([d.y for d in data_list]).mean())
        print(f"  overall anomaly rate: {rate:.4f}")
    return data_list


def main():
    ap = argparse.ArgumentParser(description="SaR-RAG anomaly detection")
    ap.add_argument("--dataset", default="bitcoinotc", choices=["bitcoinotc", "erc20"])
    ap.add_argument("--data-dir", default="F:/Next_paper_2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--kg-dim", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-reach-nodes", type=int, default=200)
    ap.add_argument("--tau-days", type=float, default=30.0)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--edge-rule", default="and", choices=["and", "or", "dst"],
                    help="how node-level weak labels propagate to transactions; "
                         "'or' reproduces the original rule but leaves 70-85%% "
                         "of transactions positive")
    ap.add_argument("--label-percentile", type=float, default=80.0,
                    help="addresses above this anomaly-score percentile are flagged")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--ablation", default="full",
                    choices=["full", "no_kg", "no_selective", "no_refinement", "no_memory"],
                    help="full | no_kg (drop KGNN + S_struct) | no_selective "
                         "(S_sem only) | no_refinement (plain TopK) | no_memory "
                         "(no retrieval at all)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = {"bitcoinotc": "soc-sign-bitcoinotc.csv",
             "erc20": "combined_data_sar_rag.csv"}
    path = os.path.join(args.data_dir, files[args.dataset])
    tag = "BOTC" if args.dataset == "bitcoinalpha" else "ERC20"

    print("=" * 74)
    print(f"SaR-RAG  |  dataset={args.dataset}  ablation={args.ablation}  device={DEVICE}")
    print("=" * 74)

    t0 = time.time()
    data_list = build_dataset(args.dataset, path, args.max_depth, args.max_reach_nodes,
                              edge_rule=args.edge_rule,
                              label_percentile=args.label_percentile)
    print(f"graph construction took {time.time()-t0:.1f}s\n")

    train_data, test_data = train_test_split(
        data_list, test_size=args.test_size, shuffle=False)
    y_test = torch.cat([d.y for d in test_data])
    base_rate = float(y_test.mean())
    print(f"train slices: {len(train_data)}  |  test slices: {len(test_data)}")
    print(f"test anomaly rate: {base_rate:.4f}  "
          f"(all-positive baseline F1 = {2*base_rate/(1+base_rate):.4f})")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    memory_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    model = SaRRAGAnomalyDetector(
        input_dim=NODE_FEATURE_DIM, hidden_dim=args.hidden_dim, kg_dim=args.kg_dim,
        top_k=args.top_k, tau_days=args.tau_days, ablation=args.ablation).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_par:,}\n")

    memory = RefinedAnomalyMemory(max_size=2000)
    train_model(model, train_loader, memory, memory_loader=memory_loader,
                epochs=args.epochs, lr=args.lr, tau_days=args.tau_days)

    if model.use_memory:
        rebuild_memory(model, memory, memory_loader, args.tau_days * 86400.0)
        print(f"\nfinal {memory.stats()}")

    print("\n--- test-set evaluation ---")
    res = evaluate_model(model, test_loader, memory, tag=tag,
                         outdir=args.outdir, plots=not args.no_plots)
    print("\n" + "=" * 74)
    return model, res


if __name__ == "__main__":
    main()
