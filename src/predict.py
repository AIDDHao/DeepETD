import torch
import json
import pandas as pd
from .model import InteractionPredictionModel  
from .data_loader import load_data 
import torch.nn as nn
import torch.optim as optim

def predict_and_save_results(model, dataloader, protein_names, compound_names, output_file="predictions.json"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_scores = [] 
    all_protein_names = []
    all_compound_names = []

    protein_idx = 0  
    compound_idx = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            (compound_diseases, compound_phenotypes, compound_subcellular_locations,
             protein_diseases, protein_phenotypes, protein_subcellular_locations) = inputs

            compound_diseases = compound_diseases.to(device)
            compound_phenotypes = compound_phenotypes.to(device)
            compound_subcellular_locations = compound_subcellular_locations.to(device)
            protein_diseases = protein_diseases.to(device)
            protein_phenotypes = protein_phenotypes.to(device)
            protein_subcellular_locations = protein_subcellular_locations.to(device)

            outputs = model(compound_diseases, compound_phenotypes, compound_subcellular_locations,
                            protein_diseases, protein_phenotypes, protein_subcellular_locations)

            scores = outputs.cpu().detach().numpy().ravel()

            batch_size = len(scores)
            batch_protein_names = protein_names[protein_idx:protein_idx + batch_size]
            batch_compound_names = compound_names[compound_idx:compound_idx + batch_size]

            all_scores.extend(scores)
            all_protein_names.extend(batch_protein_names)
            all_compound_names.extend(batch_compound_names)

            protein_idx += batch_size
            compound_idx += batch_size

    df = pd.DataFrame({
        'Protein Name': all_protein_names,
        'Compound Name': all_compound_names,
        'Prediction Score': all_scores
    })

    result = {}
    for compound in df['Compound Name'].unique():
        compound_df = df[df['Compound Name'] == compound]
        top_30 = compound_df.sort_values(by='Prediction Score', ascending=False).head(30)
        result[compound] = {
            'Protein Names': top_30['Protein Name'].tolist(),
            'Prediction Scores': top_30['Prediction Score'].tolist()
        }

    with open(output_file, 'w') as output_json:
        json.dump(result, output_json, indent=4)

    print(f"Predictions saved to {output_file}")

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

    model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))

    predict_and_save_results(
        model=model,
        dataloader=text_dataloader,
        protein_names=text_protein_names,
        compound_names=text_compound_names,
        output_file="predictions.json"
    )
