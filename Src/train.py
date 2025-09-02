from datetime import datetime
import os
import argparse
import numpy as np
from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

from dataloader import get_dataloaders, set_seed
from model import InteractionPredictionModel_NoAttention, InteractionPredictionModel


def log_init():
    return f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_write(logfile, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {msg}\n")


def train_model(train_loader, val_loader, model, optimizer, pos_weight=1.0, epochs=10, patience=3, seed=42, model_save_path="best_model.pth"):
    logfile = log_init()

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_auc = 0.0
    patience_counter = 0

    log_write(logfile, "🚀 Starting training...")
    for epoch in range(epochs):
        log_write(logfile, f"\n🔁 Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        all_y = []
        all_logits = []

        for i, (inputs, labels) in enumerate(train_loader):
                (cd, cp, cs, pd, pp, ps) = inputs
                cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
                labels = labels.to(device).float()

                logits = model(cd, cp, cs, pd, pp, ps)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                all_y.append(labels.detach().cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                raise

        all_y = np.concatenate(all_y).ravel()
        all_probs = expit(np.concatenate(all_logits).ravel())
        train_auc = roc_auc_score(all_y, all_probs)
        train_acc = accuracy_score(all_y, (all_probs > 0.5).astype(int))
        avg_train_loss = running_loss / max(1, len(train_loader))
        log_write(logfile, f"✅ Train | Loss: {avg_train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f}")

        # ---------------- Val ----------------
        model.eval()
        v_loss = 0.0
        vy, vlogits = [], []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                (cd, cp, cs, pd, pp, ps) = inputs
                cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
                labels = labels.to(device).float()

                logits = model(cd, cp, cs, pd, pp, ps)
                loss = criterion(logits, labels)

                v_loss += loss.item()
                vy.append(labels.cpu().numpy())
                vlogits.append(logits.cpu().numpy())

        vy = np.concatenate(vy).ravel()
        vprobs = expit(np.concatenate(vlogits).ravel())
        val_auc = roc_auc_score(vy, vprobs)
        val_acc = accuracy_score(vy, (vprobs > 0.5).astype(int))
        avg_val_loss = v_loss / max(1, len(val_loader))

        log_write(logfile, f"🧪 Val   | Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        # checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            log_write(logfile, f"💾 New best (AUC={val_auc:.4f}). Saved to {model_save_path}")
        else:
            patience_counter += 1
            log_write(logfile, f"⚠️  No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            log_write(logfile, f"⏹️  Early stopping at epoch {epoch+1}")
            break

    log_write(logfile, f"🎉 Training completed. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease_json', default='../Data/disease_list.json')
    parser.add_argument('--phenotype_json', default='../Data/phenotype.json')
    parser.add_argument('--positive_json', default='../Data/pos_datasets.json')
    parser.add_argument('--negative_json', default='../Data/neg_datasets.json')
    parser.add_argument('--model_out', default='model_no_attention.pth')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model instead of mean-pool')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--pos_weight', type=float, default=3.0, help='Positive class weight for BCEWithLogitsLoss')

    args = parser.parse_args()

    # Load data
    loaders = get_dataloaders(
        disease_json_path=args.disease_json,
        phenotype_json_path=args.phenotype_json,
        positive_json_path=args.positive_json,
        negative_json_path=args.negative_json,
        text_json_path=args.text_json,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=args.seed,
    )

    enc = loaders['encoders']

    # Build model with correct vocab sizes from encoders
    num_diseases = len(enc['disease'].classes_)
    num_phenotypes = len(enc['phenotype'].classes_)
    num_sub = len(enc['subcellular'].classes_)

    if args.use_attention:
        model = InteractionPredictionModel(
            disease_embedding_dim=64,
            phenotype_embedding_dim=32,
            subcellular_embedding_dim=32,
            num_diseases=num_diseases,
            num_phenotypes=num_phenotypes,
            num_subcellular_locations=num_sub,
            hidden_dim1=256,
            hidden_dim2=128,
            dropout_rate=0.1,
        )
    else:
        model = InteractionPredictionModel_NoAttention(
            disease_embedding_dim=64,
            phenotype_embedding_dim=32,
            subcellular_embedding_dim=32,
            num_diseases=num_diseases,
            num_phenotypes=num_phenotypes,
            num_subcellular_locations=num_sub,
            hidden_dim1=256,
            hidden_dim2=128,
            dropout_rate=0.1,
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_model(
        loaders['train'],
        loaders['val'],
        model,
        optimizer,
        pos_weight=args.pos_weight,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        model_save_path=args.model_out,
    )
