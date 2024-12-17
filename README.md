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

## Usage
### Training
```bash
python src/train.py
```
### Prediction
```bash
python src/predict.py 
```

## Directory Structure
- `src/`: Core source code.
  - `data_loader.py`: Code for data preprocessing and loading.
  - `model.py`: Model definition.
  - `train.py`: Training script.
  - `predict.py`: Prediction script.
- `data/`: Input and output data files.
- `test/`: Unit and integration tests.
- `scripts/`: Utility scripts.

## Requirements
Specify required dependencies in `requirements.txt`.
![image](https://github.com/user-attachments/assets/0e791e1a-2989-4681-a07d-aae2d64af04a)
