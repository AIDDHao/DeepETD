import argparse
import json
import pandas as pd
import numpy as np
import torch
from dataloader import get_dataloaders, extract_names_from_text_json, set_seed
from model import InteractionPredictionModel_NoAttention, InteractionPredictionModel


def predict(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            (cd, cp, cs, pd, pp, ps) = inputs
            cd, cp, cs, pd, pp, ps = cd.to(device), cp.to(device), cs.to(device), pd.to(device), pp.to(device), ps.to(device)
            logits = model(cd, cp, cs, pd, pp, ps)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            scores.extend(probs.tolist())
    return np.array(scores)


def save_topk_per_compound(scores, protein_names, compound_names, out_json, topk=20):
    df = pd.DataFrame({
        'Protein Name': protein_names,
        'Compound Name': compound_names,
        'Prediction Score': scores
    })
    result = {}
    for compound in df['Compound Name'].unique():
        cdf = df[df['Compound Name'] == compound].sort_values('Prediction Score', ascending=False).head(topk)
        result[compound] = {
            'Protein Names': cdf['Protein Name'].tolist(),
            'Prediction Scores': [float(x) for x in cdf['Prediction Score'].tolist()],
            'Score Type': 'sigmoid_probability'
        }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ Predictions saved to {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease_json', default='../Data/disease_list.json')
    parser.add_argument('--phenotype_json', default='../Data/phenotype.json')
    parser.add_argument('--positive_json', default='../Data/pos_datasets.json')
    parser.add_argument('--negative_json', default='../Data/neg_datasets.json')
    parser.add_argument('--predict_json', default='../Data/predict_datasets.json')

    parser.add_argument('--checkpoint', default='model_no_attention.pth')
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--out', default='predictions.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Build loaders just to get the text dataloader and encoder sizes
    loaded = get_dataloaders(
        disease_json_path=args.disease_json,
        phenotype_json_path=args.phenotype_json,
        positive_json_path=args.positive_json,
        negative_json_path=args.negative_json,
        text_json_path=args.text_json,
        batch_size=16,
        val_split=0.2,
        seed=args.seed,
    )

    enc = loaded['encoders']
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

    # Load weights
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state)

    # Names (aligned with text samples order)
    protein_names, compound_names = extract_names_from_text_json(args.text_json)

    # Predict on text dataloader
    scores = predict(model, loaded['text'])

    # Basic safety if lengths mismatch due to batching/filters
    n = min(len(scores), len(protein_names), len(compound_names))
    save_topk_per_compound(scores[:n], protein_names[:n], compound_names[:n], args.out, topk=20)

