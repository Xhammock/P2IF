from model.nets.model_Region2Vec import UrbanModel
from model.nets.model_aug import UrbanModelAug
from model.nets.model_GAT import UrbanModelGAT
from model.nets.urban_model_spatial_only import UrbanModelSpatialOnly
from model.nets.model_without_res import UrbanModelAugFullQuery
from model.nets.model_without_vis import UrbanModelAugWithoutVis
from model.nets.model_without_cl import UrbanModelAugWithoutCL
from model.nets.model_without_interaction import UrbanModelAugWithoutInteraction
from model.nets.model_ReMVC import UrbanModelReMVC
from model.nets.model_HREP import UrbanModelHREP
from dataload.urban_dataset import UrbanWeekdayDataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime

import matplotlib

# 使用无界面后端，方便服务器保存图片
matplotlib.use("Agg")


def _collate_single(batch):
    # 数据集只有一张图，直接返回首个元素，避免默认 collate 处理 DGLGraph 报错
    return batch[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/urban.json")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    dims = cfg["data"]["dims"]
    dataset = UrbanWeekdayDataset(
        data_root=cfg["data"]["root"],
        feature_file=cfg["data"]["feature_file"],
        spatial_adj=cfg["data"]["spatial_adj"],
        spatial_ids=cfg["data"]["spatial_ids"],
        od_matrix=cfg["data"]["od_matrix"],
        od_edgelist=cfg["data"].get("od_edgelist"),
        top_k_od=cfg["data"].get("top_k_od"),
        dims=dims,
        use_street=cfg["data"].get("use_street", False),
        street_dim=dims.get("street", 0),
        device=device,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=_collate_single)

    # 根据配置文件自动选择模型类型
    model_config = cfg["model"]

    # 检测模型类型：
    # 1. 优先检查是否有model_type字段（显式指定）
    # 2. 检查是否有use_res参数（spatial_only模型特有）
    # 3. 检查是否有gat_layers（GAT模型）
    # 4. 检查视图增强参数（UrbanModelAug）
    model_type = model_config.get("model_type", None)
    has_use_res = "use_res" in model_config
    has_gat_layers = "gat_layers" in model_config
    has_aug_params = any(key in model_config for key in [
        "feat_drop_ratio", "edge_drop_ratio", "tau", "od_mask_topk"])

    if model_type == "full_query":
        print("[模型类型] W/o Res：使用完整embedding查询 (UrbanModelAugFullQuery)")
        # 消融实验参数
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # 打印消融实验配置
        if not use_poi or not use_vis or not use_street:
            print("[消融实验配置]")
            print(f"  use_poi: {use_poi}")
            print(f"  use_vis: {use_vis}")
            print(f"  use_street: {use_street}")
        print("[注意] 使用完整的 embedding (h_spatial) 作为查询向量，而不是 res 子空间")

        model = UrbanModelAugFullQuery(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # 视图增强参数
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            # 消融实验参数
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "remvc":
        print("[模型类型] ReMVC：多视图对比学习 (UrbanModelReMVC)")
        model = UrbanModelReMVC(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config.get("sage_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            proj_dim=model_config.get("proj_dim", 128),
            tau=model_config.get("tau", 0.1),
            use_street=model_config.get("use_street", cfg["data"].get("use_street", True)),
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "hrep":
        print("[模型类型] HREP：关系感知嵌入 (UrbanModelHREP)")
        model = UrbanModelHREP(
            dims=dims,
            embedding_size=model_config.get("embedding_size", 144),
            gcn_layers=model_config.get("gcn_layers", 3),
            dropout=model_config.get("dropout", 0.1),
            importance_k=model_config.get("importance_k", 10),
            proj_dim=model_config.get("proj_dim", 128),
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "without_vis":
        print("[模型类型] W/o Vis：去除访客画像信息 (UrbanModelAugWithoutVis)")
        use_poi = model_config.get("use_poi", True)
        use_street = model_config.get("use_street", True)
        model = UrbanModelAugWithoutVis(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            use_poi=use_poi,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "without_cl":
        print("[模型类型] W/o CL：去除空间感知对比约束 (UrbanModelAugWithoutCL)")
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)
        model = UrbanModelAugWithoutCL(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "without_interaction":
        print("[模型类型] W/o Interaction：去除分模态交互机制 (UrbanModelAugWithoutInteraction)")
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)
        model = UrbanModelAugWithoutInteraction(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "spatial_only" or (has_use_res and not has_gat_layers):
        print("[模型类型] 使用仅空间GraphSAGE模型 (UrbanModelSpatialOnly)")
        # 消融实验参数
        use_poi = model_config.get("use_poi", True)
        use_res = model_config.get("use_res", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # 打印消融实验配置
        print("[消融实验配置]")
        print(f"  use_poi: {use_poi}")
        print(f"  use_res: {use_res}")
        print(f"  use_vis: {use_vis}")
        print(f"  use_street: {use_street}")

        model = UrbanModelSpatialOnly(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            # 视图增强参数
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            # 消融实验参数
            use_poi=use_poi,
            use_res=use_res,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True  # 使用视图增强
        use_spatial_only = True  # 标记为spatial_only模型
    elif has_gat_layers:
        print("[模型类型] 使用 GAT 模型 (UrbanModelGAT)")
        model = UrbanModelGAT(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            gat_layers=model_config.get("gat_layers", 1),
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # 视图增强参数
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
        ).to(device)
        use_aug_model = True  # GAT模型使用视图增强
        use_spatial_only = False
    elif has_aug_params:
        print("[模型类型] 使用视图增强模型 (UrbanModelAug)")
        # 消融实验参数
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # 打印消融实验配置
        if not use_poi or not use_vis or not use_street:
            print("[消融实验配置]")
            print(f"  use_poi: {use_poi}")
            print(f"  use_vis: {use_vis}")
            print(f"  use_street: {use_street}")

        model = UrbanModelAug(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # 视图增强参数
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            # 消融实验参数
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    else:
        print("[模型类型] 使用 Region2Vec 模型 (UrbanModel)")
        model = UrbanModel(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # Region2Vec-style loss 参数
            spatial_lambda=model_config.get("spatial_lambda", 0.1),
            loss_eps=model_config.get("loss_eps", 1e-15),
            hops_threshold=model_config.get("hops_threshold", 5),
            hops_matrix_path=cfg["data"].get("hops_matrix_path") or model_config.get(
                "hops_matrix_path"),  # 优先从 data 配置读取
            loss_type=model_config.get("loss_type", "divreg"),
        ).to(device)
        use_aug_model = False
        use_spatial_only = False

    optimizer = torch.optim.Adam(model.parameters(
    ), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    # 学习率调度器
    scheduler_config = cfg["optim"].get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "ReduceLROnPlateau")
    scheduler = None

    if scheduler_type == "ReduceLROnPlateau":
        # 当loss不再下降时降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get("factor", 0.5),  # 学习率衰减因子
            patience=scheduler_config.get("patience", 10),  # 容忍多少个epoch无改善
            min_lr=scheduler_config.get("min_lr", 1e-6),  # 最小学习率
            verbose=True  # 打印学习率变化
        )
        print(f"[学习率调度器] ReduceLROnPlateau: factor={scheduler_config.get('factor', 0.5)}, "
              f"patience={scheduler_config.get('patience', 10)}, min_lr={scheduler_config.get('min_lr', 1e-6)}")
    elif scheduler_type == "StepLR":
        # 每隔固定epoch降低学习率
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 50),
            gamma=scheduler_config.get("gamma", 0.5)
        )
        print(f"[学习率调度器] StepLR: step_size={scheduler_config.get('step_size', 50)}, "
              f"gamma={scheduler_config.get('gamma', 0.5)}")
    elif scheduler_type == "CosineAnnealingLR":
        # 余弦退火
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"],
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
        print(f"[学习率调度器] CosineAnnealingLR: T_max={cfg['train']['epochs']}, "
              f"min_lr={scheduler_config.get('min_lr', 1e-6)}")
    elif scheduler_type == "ExponentialLR":
        # 指数衰减
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.95)
        )
        print(
            f"[学习率调度器] ExponentialLR: gamma={scheduler_config.get('gamma', 0.95)}")
    elif scheduler_type == "None" or scheduler_type is None:
        print("[学习率调度器] 未启用学习率衰减")
    else:
        print(f"[警告] 未知的学习率调度器类型: {scheduler_type}，将不使用学习率衰减")

    # 创建基于时间戳的输出目录
    base_log_dir = cfg["train"]["out_dir"]  # "checkpoints"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # "checkpoints/train_20241218_143025"
    log_dir = os.path.join(base_log_dir, f"train_{timestamp}")

    # 确保 checkpoints 目录存在
    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 保存配置文件副本到输出目录
    config_copy_path = os.path.join(log_dir, "config.json")
    with open(config_copy_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[训练开始] 输出目录: {log_dir}")
    print(f"[配置已保存] {config_copy_path}")

    ckpt_path = os.path.join(log_dir, "urban_model.pt")
    best_ckpt_path = os.path.join(log_dir, "urban_model_best.pt")
    emb_path = os.path.join(log_dir, "best_embeddings.npz")
    curve_path = os.path.join(log_dir, "loss_curve.csv")
    curve_png = os.path.join(log_dir, "loss_curve.png")

    history = []
    best_loss = float("inf")

    # 早停参数
    patience = cfg["train"].get("patience", None)  # None 表示不使用早停
    patience_counter = 0
    best_epoch = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in batch.items()}
            loss, info = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = info.get("loss", loss.item())

        # 更新学习率调度器（在记录历史之前更新，这样记录的是更新后的学习率）
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(epoch_loss)  # ReduceLROnPlateau 需要传入 loss
            else:
                scheduler.step()  # 其他调度器不需要参数
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[学习率更新] {old_lr:.2e} -> {new_lr:.2e}")

        # 获取当前学习率（更新后的）
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "loss_pos": info.get("loss_pos", 0.0),
            "loss_neg": info.get("loss_neg", 0.0),
            "loss_spatial": info.get("loss_spatial", 0.0),
            "lr": current_lr,
        })

        # 打印详细的损失信息（兼容两种模型）
        loss_pos = info.get('loss_pos', 0.0)
        loss_neg = info.get('loss_neg', 0.0)
        loss_spatial = info.get('loss_spatial', 0.0)
        num_pos = info.get('num_pos_pairs', 0)
        num_neg = info.get('num_neg_pairs', 0)
        num_hops = info.get('num_hops_pairs', 0)

        # 检查是否有改善
        improved = False
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            patience_counter = 0
            improved = True
            # 保存最佳模型
            torch.save(model.state_dict(), best_ckpt_path)
            if use_aug_model:
                # 视图增强模型：只显示主要损失
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"lr={current_lr:.2e} | "
                      f"✓ 最佳模型已更新 (best={best_loss:.4f})")
            else:
                # Region2Vec模型：显示详细损失
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"pos={loss_pos:.2f}(n={num_pos}) | "
                      f"neg={loss_neg:.2f}(n={num_neg}) | "
                      f"spatial={loss_spatial:.4f}(n={num_hops}) | "
                      f"lr={current_lr:.2e} | "
                      f"✓ 最佳模型已更新 (best={best_loss:.4f})")
        else:
            patience_counter += 1
            if use_aug_model:
                # 视图增强模型：只显示主要损失
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"lr={current_lr:.2e} | "
                      f"无改善 ({patience_counter}/{patience if patience else 'N/A'})")
            else:
                # Region2Vec模型：显示详细损失
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"pos={loss_pos:.2f}(n={num_pos}) | "
                      f"neg={loss_neg:.2f}(n={num_neg}) | "
                      f"spatial={loss_spatial:.4f}(n={num_hops}) | "
                      f"lr={current_lr:.2e} | "
                      f"无改善 ({patience_counter}/{patience if patience else 'N/A'})")

        # 定期保存最新检查点（覆盖）
        if (epoch + 1) % cfg["train"].get("save_every", 10) == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': epoch_loss,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'current_lr': current_lr,
            }, ckpt_path)
            print(f"[检查点已保存] {ckpt_path}")

        # 早停检查
        if patience is not None and patience_counter >= patience:
            print(f"\n[早停触发] 已连续 {patience} 个 epoch 无改善，停止训练")
            print(f"  最佳 epoch: {best_epoch}, 最佳损失: {best_loss:.4f}")
            break

    # 保存损失曲线，便于后续可视化
    with open(curve_path, "w") as f:
        f.write("epoch,loss,loss_pos,loss_neg,loss_spatial,lr\n")
        for rec in history:
            f.write(f"{rec['epoch']},{rec['loss']},{rec.get('loss_pos', 0.0)},{rec.get('loss_neg', 0.0)},{rec.get('loss_spatial', 0.0)},{rec.get('lr', cfg['optim']['lr'])}\n")

    # 直接绘制并保存损失曲线
    epochs = [rec["epoch"] for rec in history]
    losses = [rec["loss"] for rec in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(curve_png, dpi=200)
    plt.close()

    # 导出最佳节点嵌入，便于聚类/2D/3D 可视化
    model.eval()
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    with torch.no_grad():
        batch = dataset[0]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)
                 else v for k, v in batch.items()}
        encode_result = model.encode(batch)

        # 处理不同模型的encode返回值
        if use_spatial_only:
            # UrbanModelSpatialOnly只返回z
            z = encode_result
            np.savez(
                emb_path,
                node_ids=np.array(dataset.node_ids, dtype=np.int64),
                z=z.cpu().numpy(),
            )
        else:
            # UrbanModel/UrbanModelAug/UrbanModelGAT返回h_spatial, h_od, z
            h_spatial, h_od, z = encode_result
            np.savez(
                emb_path,
                node_ids=np.array(dataset.node_ids, dtype=np.int64),
                h_spatial=h_spatial.cpu().numpy(),
                h_od=h_od.cpu().numpy(),
                z=z.cpu().numpy(),
            )
    print(f"\n[训练完成]")
    print(f"  总 epoch 数: {len(history)}")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  输出目录: {log_dir}")
    print(f"  最佳模型: {best_ckpt_path}")
    print(f"  嵌入文件: {emb_path}")
    print(f"  损失曲线: {curve_path}")
    print(f"  损失图表: {curve_png}")


if __name__ == "__main__":
    main()
