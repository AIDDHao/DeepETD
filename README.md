# DeepETD
This repository contains a deep learning model for predicting interactions between Endogenous Metabolite and target proteins.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AIDDHao/DeepETD
   cd DeepETD
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### How to run

```bash
# 1) Train
python train.py \
  --disease_json ../Data/disease_list.json \
  --phenotype_json ../Data/phenotype.json \
  --positive_json ../Data/pos_datasets.json \
  --negative_json ../Data/neg_datasets..json \
  --predict_json ../Data/predict_datasets.json \
  --model_out best_model.pth \
  --epochs 20 --patience 10 --pos_weight 3.0


# 2) Predict
python predict.py \
  --disease_json ../Data/disease_list.json \
  --phenotype_json ../Data/phenotype.json \
  --positive_json ../Data/pos_datasets.json \
  --negative_json ../Data/neg_datasets..json \
  --predict_json ../Data/predict_datasets.json \
  --checkpoint best_model.pth \
  --out predictions.json
```

**Notes**

* The model returns logits; we apply `BCEWithLogitsLoss` during training and `sigmoid` only for metrics and prediction.
* Vocab sizes are taken from the fitted label encoders to avoid mismatches (e.g., subcellular locations length).
* `predict.py` groups and exports top-20 proteins per compound as you specified.
* If any sample has an empty list for a modality, we fallback to index 0 for that encoder to keep tensor shapes valid. Adjust if you prefer a dedicated `<UNK>` token.
