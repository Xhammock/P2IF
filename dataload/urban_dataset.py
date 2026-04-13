import json
import math
import os
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


def _load_adjacency_ids(path: str) -> List[int]:
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _standardize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + eps)


class UrbanWeekdayDataset(data.Dataset):
    """
    单图数据集：包含空间邻接图与 OD 功能图，以及节点多模态特征。
    """

    def __init__(
        self,
        data_root: str = "data/weekday",
        feature_file: str = "features_normalized_weekday.csv",
        spatial_adj: str = "Spatial_Adjacency/adjacency_matrix.csv",
        spatial_ids: str = "Spatial_Adjacency/adjacency_ids.txt",
        od_matrix: str = "OD_Aajacency/Flow_matrix_weekday.csv",
        od_edgelist: Optional[str] = None,
        top_k_od: Optional[int] = None,
        dims: Optional[Dict[str, int]] = None,
        use_street: bool = False,
        street_dim: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.device = device or torch.device("cpu")
        self.top_k_od = top_k_od
        self.use_street = use_street
        self.street_dim = street_dim if use_street else 0
        self.dims = dims or {
            "poi": 155,
            "res": 28,
            "vis": 28,
            "street": self.street_dim,
        }

        feature_path = os.path.join(data_root, feature_file)
        spatial_adj_path = os.path.join(data_root, spatial_adj)
        spatial_ids_path = os.path.join(data_root, spatial_ids)
        od_matrix_path = os.path.join(data_root, od_matrix)
        od_edge_path = os.path.join(
            data_root, od_edgelist) if od_edgelist else None

        self.node_ids = _load_adjacency_ids(spatial_ids_path)
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        self.features = self._load_features(feature_path)
        self.g_spatial = self._build_spatial_graph(spatial_adj_path)
        self.g_od = self._build_od_graph(od_matrix_path, od_edge_path)

    def _load_features(self, feature_path: str) -> torch.Tensor:
        df = pd.read_csv(feature_path)
        node_ids = df.iloc[:, 0].astype(int).tolist()
        assert node_ids == self.node_ids, "feature 节点顺序与 adjacency_ids 不一致"

        poi_dim = self.dims["poi"]
        res_dim = self.dims["res"]
        vis_dim = self.dims["vis"]
        street_dim = self.dims.get("street", 0)

        poi_cols = df.columns[1: 1 + poi_dim]
        res_cols = df.columns[1 + poi_dim: 1 + poi_dim + res_dim]
        vis_cols = df.columns[1 + poi_dim +
                              res_dim: 1 + poi_dim + res_dim + vis_dim]

        poi_feat = df[poi_cols].to_numpy(dtype=np.float32)
        res_feat = df[res_cols].to_numpy(dtype=np.float32)
        vis_feat = df[vis_cols].to_numpy(dtype=np.float32)

        if self.use_street and street_dim > 0:
            street_cols = df.columns[
                1 + poi_dim + res_dim + vis_dim: 1 + poi_dim + res_dim + vis_dim + street_dim
            ]
            street_feat = df[street_cols].to_numpy(dtype=np.float32)
        else:
            street_feat = np.zeros((len(df), street_dim), dtype=np.float32)

        feats = np.concatenate(
            [poi_feat, res_feat, vis_feat, street_feat], axis=1)
        return torch.from_numpy(feats)

    def _build_spatial_graph(self, adj_path: str) -> dgl.DGLGraph:
        # CSV 第一列是节点 id，跳过首列只读邻接矩阵数值
        adj = np.loadtxt(
            adj_path,
            delimiter=",",
            skiprows=1,
            usecols=range(1, len(self.node_ids) + 1),
        )
        assert adj.shape[0] == adj.shape[1] == len(
            self.node_ids), "空间邻接矩阵尺寸不匹配"

        src, dst = np.nonzero(adj > 0)
        # 构建无向图：补齐反向边
        src_all = np.concatenate([src, dst])
        dst_all = np.concatenate([dst, src])

        g = dgl.graph((src_all, dst_all), num_nodes=len(self.node_ids))
        g.ndata["feat"] = self.features.clone()
        return g

    def _build_od_graph(self, matrix_path: str, edge_path: Optional[str]) -> dgl.DGLGraph:
        if edge_path and os.path.exists(edge_path):
            # 三列格式：src_id, dst_id, weight
            edge_df = pd.read_csv(edge_path)
            assert edge_df.shape[1] >= 3, "od edge list 需要三列"
            src_ids = edge_df.iloc[:, 0].astype(int).to_numpy()
            dst_ids = edge_df.iloc[:, 1].astype(int).to_numpy()
            weights = edge_df.iloc[:, 2].to_numpy(dtype=np.float32)
        else:
            # OD 矩阵文件没有额外的 id 行/列，直接读取全部数值
            mat = np.loadtxt(matrix_path, delimiter=",", skiprows=0)
            assert mat.shape[0] == mat.shape[1] == len(
                self.node_ids), "OD 矩阵尺寸不匹配"
            weights = []
            src_ids = []
            dst_ids = []
            for i in range(mat.shape[0]):
                row = mat[i]
                nz_idx = np.nonzero(row > 0)[0]
                if self.top_k_od is not None and self.top_k_od > 0 and nz_idx.size > self.top_k_od:
                    topk_idx = np.argpartition(
                        row[nz_idx], -self.top_k_od)[-self.top_k_od:]
                    nz_idx = nz_idx[topk_idx]
                for j in nz_idx:
                    src_ids.append(self.node_ids[j])  # 目标 j
                    dst_ids.append(self.node_ids[i])  # 源 i
                    weights.append(row[j])

            src_ids = np.array(src_ids, dtype=int)
            dst_ids = np.array(dst_ids, dtype=int)
            weights = np.array(weights, dtype=np.float32)

        # 将节点 id 映射为索引，并反向存边：dst=源，src=目的地，方便在 dst 聚合
        src_idx = np.vectorize(self.id_to_idx.get)(src_ids)
        dst_idx = np.vectorize(self.id_to_idx.get)(dst_ids)

        # 保存原始权重（用于 Region2Vec loss）
        raw_weights = np.array(weights, dtype=np.float32)
        norm_weights = _standardize(np.log1p(weights))

        g = dgl.graph((src_idx, dst_idx), num_nodes=len(self.node_ids))
        g.ndata["feat"] = self.features.clone()
        g.edata["flow"] = torch.from_numpy(norm_weights).unsqueeze(-1).float()
        g.edata["flow_raw"] = torch.from_numpy(raw_weights).unsqueeze(-1).float()  # 原始 flow 值
        return g

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        # 单图任务直接返回两张图与特征
        return {
            "g_spatial": self.g_spatial,
            "g_od": self.g_od,
            "feat": self.features,
        }


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)
