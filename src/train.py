import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from .model import InteractionPredictionModel
from .data_loader import load_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, 
                epochs=10, patience=3, seed=42, model_save_path="best_model.pth"):
    set_seed(seed)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) 

    best_auc_roc = 0.0
    patience_counter = 0
    early_stop = False

    for epoch in range(epochs):
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            (compound_diseases, compound_phenotypes, compound_subcellular_locations,
             protein_diseases, protein_phenotypes, protein_subcellular_locations) = inputs

            compound_diseases = compound_diseases.to(device)
            compound_phenotypes = compound_phenotypes.to(device)
            compound_subcellular_locations = compound_subcellular_locations.to(device)
            protein_diseases = protein_diseases.to(device)
            protein_phenotypes = protein_phenotypes.to(device)
            protein_subcellular_locations = protein_subcellular_locations.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(compound_diseases, compound_phenotypes, compound_subcellular_locations,
                            protein_diseases, protein_phenotypes, protein_subcellular_locations)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_train_labels.append(labels.cpu().detach().numpy())
            all_train_outputs.append(outputs.cpu().detach().numpy())

        all_train_labels = np.concatenate(all_train_labels).ravel()
        all_train_outputs = np.concatenate(all_train_outputs).ravel()
        train_auc_roc = roc_auc_score(all_train_labels, all_train_outputs)
        train_predictions = (all_train_outputs > 0.6).astype(int)
        train_accuracy = accuracy_score(all_train_labels, train_predictions)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_dataloader):.4f}, Training AUC-ROC: {train_auc_roc:.4f}, Training Accuracy: {train_accuracy:.4f}')

        model.eval()
        val_loss = 0.0  
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                (compound_diseases, compound_phenotypes, compound_subcellular_locations,
                 protein_diseases, protein_phenotypes, protein_subcellular_locations) = inputs

                compound_diseases = compound_diseases.to(device)
                compound_phenotypes = compound_phenotypes.to(device)
                compound_subcellular_locations = compound_subcellular_locations.to(device)
                protein_diseases = protein_diseases.to(device)
                protein_phenotypes = protein_phenotypes.to(device)
                protein_subcellular_locations = protein_subcellular_locations.to(device)

                labels = labels.to(device)

                outputs = model(compound_diseases, compound_phenotypes, compound_subcellular_locations,
                                protein_diseases, protein_phenotypes, protein_subcellular_locations)

                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                all_val_labels.append(labels.cpu().detach().numpy())
                all_val_outputs.append(outputs.cpu().detach().numpy())

        all_val_labels = np.concatenate(all_val_labels).ravel()
        all_val_outputs = np.concatenate(all_val_outputs).ravel()
        val_auc_roc = roc_auc_score(all_val_labels, all_val_outputs)
        val_predictions = (all_val_outputs > 0.5).astype(int)
        val_accuracy = accuracy_score(all_val_labels, val_predictions)

        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation AUC-ROC: {val_auc_roc:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_auc_roc > best_auc_roc:
            print(f"New best model found at epoch {epoch+1}, saving model...")
            best_auc_roc = val_auc_roc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered. Best Validation AUC-ROC: {best_auc_roc:.4f}')
            early_stop = True
            break

        if early_stop:
            break

    print(f"Training completed. Best model saved with Validation AUC-ROC: {best_auc_roc:.4f}")


if __name__ == "__main__":

    data_dir = "data"


    train_dataloader, val_dataloader, text_dataloader, text_compound_names, text_protein_names, disease_encoder, phenotype_encoder, subcellular_location_encoder = load_data(data_dir)


    model = InteractionPredictionModel(
        disease_embedding_dim=64,
        phenotype_embedding_dim=32,
        subcellular_embedding_dim=16,
        num_diseases=13660,
        num_phenotypes=17300,
        num_subcellular_locations=30,
        hidden_dim1=256,
        hidden_dim2=128,
        dropout_rate=0.5
    )


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_model(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        epochs=20, 
        patience=10,
        model_save_path="best_model.pth"
    )