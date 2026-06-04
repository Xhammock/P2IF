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

# Non-interactive backend for saving figures on headless servers
matplotlib.use("Agg")


def _collate_single(batch):
    # Single-graph dataset: return first sample to avoid default collate errors on DGLGraph
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

    # Select model type from config automatically
    model_config = cfg["model"]

    # Model type detection:
    # 1. Prefer explicit model_type field
    # 2. Check use_res (spatial_only-specific)
    # 3. Check gat_layers (GAT model)
    # 4. Check view-augmentation params (UrbanModelAug)
    model_type = model_config.get("model_type", None)
    has_use_res = "use_res" in model_config
    has_gat_layers = "gat_layers" in model_config
    has_aug_params = any(key in model_config for key in [
        "feat_drop_ratio", "edge_drop_ratio", "tau", "od_mask_topk"])

    if model_type == "full_query":
        print("[Model] W/o Res: full embedding query (UrbanModelAugFullQuery)")
        # Ablation flags
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # Log ablation config
        if not use_poi or not use_vis or not use_street:
            print("[Ablation config]")
            print(f"  use_poi: {use_poi}")
            print(f"  use_vis: {use_vis}")
            print(f"  use_street: {use_street}")
        print("[Note] Using full embedding (h_spatial) as query, not res subspace")

        model = UrbanModelAugFullQuery(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # View augmentation
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            # Ablation flags
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    elif model_type == "remvc":
        print("[Model] ReMVC: multi-view contrastive learning (UrbanModelReMVC)")
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
        print("[Model] HREP: relation-aware embeddings (UrbanModelHREP)")
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
        print("[Model] W/o Vis: visitor profile removed (UrbanModelAugWithoutVis)")
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
        print("[Model] W/o CL: spatial contrastive constraint removed (UrbanModelAugWithoutCL)")
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
        print("[Model] W/o Interaction: per-modality interaction removed (UrbanModelAugWithoutInteraction)")
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
        print("[Model] Spatial-only GraphSAGE (UrbanModelSpatialOnly)")
        # Ablation flags
        use_poi = model_config.get("use_poi", True)
        use_res = model_config.get("use_res", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # Log ablation config
        print("[Ablation config]")
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
            # View augmentation
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            # Ablation flags
            use_poi=use_poi,
            use_res=use_res,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True  # view augmentation enabled
        use_spatial_only = True  # spatial_only model
    elif has_gat_layers:
        print("[Model] GAT (UrbanModelGAT)")
        model = UrbanModelGAT(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            gat_layers=model_config.get("gat_layers", 1),
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # View augmentation
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
        ).to(device)
        use_aug_model = True  # GAT uses view augmentation
        use_spatial_only = False
    elif has_aug_params:
        print("[Model] View-augmented (UrbanModelAug)")
        # Ablation flags
        use_poi = model_config.get("use_poi", True)
        use_vis = model_config.get("use_vis", True)
        use_street = model_config.get("use_street", True)

        # Log ablation config
        if not use_poi or not use_vis or not use_street:
            print("[Ablation config]")
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
            # View augmentation
            feat_drop_ratio=model_config.get("feat_drop_ratio", 0.1),
            edge_drop_ratio=model_config.get("edge_drop_ratio", 0.075),
            noise_std=model_config.get("noise_std", 0.01),
            tau=model_config.get("tau", 0.1),
            od_mask_topk=model_config.get("od_mask_topk", 200),
            # Ablation flags
            use_poi=use_poi,
            use_vis=use_vis,
            use_street=use_street,
        ).to(device)
        use_aug_model = True
        use_spatial_only = False
    else:
        print("[Model] Region2Vec (UrbanModel)")
        model = UrbanModel(
            dims=dims,
            hidden_dim=model_config["hidden_dim"],
            sage_layers=model_config["sage_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            proj_dim=model_config["proj_dim"],
            loss_weight=model_config.get("loss_weight"),
            # Region2Vec-style loss
            spatial_lambda=model_config.get("spatial_lambda", 0.1),
            loss_eps=model_config.get("loss_eps", 1e-15),
            hops_threshold=model_config.get("hops_threshold", 5),
            hops_matrix_path=cfg["data"].get("hops_matrix_path") or model_config.get(
                "hops_matrix_path"),  # prefer data config
            loss_type=model_config.get("loss_type", "divreg"),
        ).to(device)
        use_aug_model = False
        use_spatial_only = False

    optimizer = torch.optim.Adam(model.parameters(
    ), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    # Learning rate scheduler
    scheduler_config = cfg["optim"].get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "ReduceLROnPlateau")
    scheduler = None

    if scheduler_type == "ReduceLROnPlateau":
        # Reduce LR when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get("factor", 0.5),  # LR decay factor
            patience=scheduler_config.get("patience", 10),  # epochs without improvement
            min_lr=scheduler_config.get("min_lr", 1e-6),  # minimum LR
            verbose=True  # log LR changes
        )
        print(f"[LR scheduler] ReduceLROnPlateau: factor={scheduler_config.get('factor', 0.5)}, "
              f"patience={scheduler_config.get('patience', 10)}, min_lr={scheduler_config.get('min_lr', 1e-6)}")
    elif scheduler_type == "StepLR":
        # Step decay every fixed number of epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 50),
            gamma=scheduler_config.get("gamma", 0.5)
        )
        print(f"[LR scheduler] StepLR: step_size={scheduler_config.get('step_size', 50)}, "
              f"gamma={scheduler_config.get('gamma', 0.5)}")
    elif scheduler_type == "CosineAnnealingLR":
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"],
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
        print(f"[LR scheduler] CosineAnnealingLR: T_max={cfg['train']['epochs']}, "
              f"min_lr={scheduler_config.get('min_lr', 1e-6)}")
    elif scheduler_type == "ExponentialLR":
        # Exponential decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.95)
        )
        print(
            f"[LR scheduler] ExponentialLR: gamma={scheduler_config.get('gamma', 0.95)}")
    elif scheduler_type == "None" or scheduler_type is None:
        print("[LR scheduler] LR decay disabled")
    else:
        print(f"[Warning] Unknown LR scheduler type: {scheduler_type}; no decay will be applied")

    # Timestamped output directory
    base_log_dir = cfg["train"]["out_dir"]  # "checkpoints"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # "checkpoints/train_20241218_143025"
    log_dir = os.path.join(base_log_dir, f"train_{timestamp}")

    # Ensure checkpoints directory exists
    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config copy to output directory
    config_copy_path = os.path.join(log_dir, "config.json")
    with open(config_copy_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Training] Output directory: {log_dir}")
    print(f"[Config saved] {config_copy_path}")

    ckpt_path = os.path.join(log_dir, "urban_model.pt")
    best_ckpt_path = os.path.join(log_dir, "urban_model_best.pt")
    emb_path = os.path.join(log_dir, "best_embeddings.npz")
    curve_path = os.path.join(log_dir, "loss_curve.csv")
    curve_png = os.path.join(log_dir, "loss_curve.png")

    history = []
    best_loss = float("inf")

    # Early stopping
    patience = cfg["train"].get("patience", None)  # None disables early stopping
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

        # Update LR scheduler before logging (records post-update LR)
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(epoch_loss)  # ReduceLROnPlateau requires loss
            else:
                scheduler.step()  # other schedulers need no argument
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[LR update] {old_lr:.2e} -> {new_lr:.2e}")

        # Current LR (after scheduler step)
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "loss_pos": info.get("loss_pos", 0.0),
            "loss_neg": info.get("loss_neg", 0.0),
            "loss_spatial": info.get("loss_spatial", 0.0),
            "lr": current_lr,
        })

        # Detailed loss logging (both model families)
        loss_pos = info.get('loss_pos', 0.0)
        loss_neg = info.get('loss_neg', 0.0)
        loss_spatial = info.get('loss_spatial', 0.0)
        num_pos = info.get('num_pos_pairs', 0)
        num_neg = info.get('num_neg_pairs', 0)
        num_hops = info.get('num_hops_pairs', 0)

        # Check for improvement
        improved = False
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            patience_counter = 0
            improved = True
            # Save best checkpoint
            torch.save(model.state_dict(), best_ckpt_path)
            if use_aug_model:
                # View-augmented models: main loss only
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"lr={current_lr:.2e} | "
                      f"best checkpoint updated (best={best_loss:.4f})")
            else:
                # Region2Vec: detailed losses
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"pos={loss_pos:.2f}(n={num_pos}) | "
                      f"neg={loss_neg:.2f}(n={num_neg}) | "
                      f"spatial={loss_spatial:.4f}(n={num_hops}) | "
                      f"lr={current_lr:.2e} | "
                      f"best checkpoint updated (best={best_loss:.4f})")
        else:
            patience_counter += 1
            if use_aug_model:
                # View-augmented models: main loss only
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"lr={current_lr:.2e} | "
                      f"no improvement ({patience_counter}/{patience if patience else 'N/A'})")
            else:
                # Region2Vec: detailed losses
                print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | "
                      f"pos={loss_pos:.2f}(n={num_pos}) | "
                      f"neg={loss_neg:.2f}(n={num_neg}) | "
                      f"spatial={loss_spatial:.4f}(n={num_hops}) | "
                      f"lr={current_lr:.2e} | "
                      f"no improvement ({patience_counter}/{patience if patience else 'N/A'})")

        # Periodic latest checkpoint (overwrite)
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
            print(f"[Checkpoint saved] {ckpt_path}")

        # Early stopping
        if patience is not None and patience_counter >= patience:
            print(f"\n[Early stop] No improvement for {patience} epochs; stopping training")
            print(f"  Best epoch: {best_epoch}, best loss: {best_loss:.4f}")
            break

    # Save loss curve for downstream plotting
    with open(curve_path, "w") as f:
        f.write("epoch,loss,loss_pos,loss_neg,loss_spatial,lr\n")
        for rec in history:
            f.write(f"{rec['epoch']},{rec['loss']},{rec.get('loss_pos', 0.0)},{rec.get('loss_neg', 0.0)},{rec.get('loss_spatial', 0.0)},{rec.get('lr', cfg['optim']['lr'])}\n")

    # Plot and save loss curve
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

    # Export best node embeddings for clustering / 2D / 3D visualization
    model.eval()
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    with torch.no_grad():
        batch = dataset[0]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)
                 else v for k, v in batch.items()}
        encode_result = model.encode(batch)

        # Handle different encode() return shapes
        if use_spatial_only:
            # UrbanModelSpatialOnly returns z only
            z = encode_result
            np.savez(
                emb_path,
                node_ids=np.array(dataset.node_ids, dtype=np.int64),
                z=z.cpu().numpy(),
            )
        else:
            # UrbanModel / UrbanModelAug / UrbanModelGAT return h_spatial, h_od, z
            h_spatial, h_od, z = encode_result
            np.savez(
                emb_path,
                node_ids=np.array(dataset.node_ids, dtype=np.int64),
                h_spatial=h_spatial.cpu().numpy(),
                h_od=h_od.cpu().numpy(),
                z=z.cpu().numpy(),
            )
    print(f"\n[Training complete]")
    print(f"  Total epochs: {len(history)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Output directory: {log_dir}")
    print(f"  Best model: {best_ckpt_path}")
    print(f"  Embeddings: {emb_path}")
    print(f"  Loss curve (CSV): {curve_path}")
    print(f"  Loss curve (PNG): {curve_png}")


if __name__ == "__main__":
    main()
