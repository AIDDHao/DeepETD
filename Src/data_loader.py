import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_SUBCELLULAR_LOCATIONS = [
    "Nucleus", "Cytoplasm", "Mitochondria", "Endoplasmic Reticulum",
    "Golgi Apparatus", "Lysosome", "Plasma Membrane", "Nuclear Membrane",
    "Peroxisome", "Nucleolus", "Cytoskeleton", "Vacuole", "Chloroplast",
    "Plasmid", "Ribosome", "Flagellum", "Microvilli", "Vesicle",
    "Thylakoid", "Centrosome", "Synaptic Vesicle", "Endosome",
    "Nuclear Pore Complex"
]

# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------
# Encoders
# -----------------------------

def _normalize_list(xs):
    return [x.lower().strip() for x in xs]


def build_label_encoders(disease_json_path: str,
                         phenotype_json_path: str,
                         subcellular_locations=None):
    """Build LabelEncoders for diseases, phenotypes, and subcellular locations.
    """
    with open(disease_json_path, 'r', encoding='utf-8') as f:
        all_diseases = json.load(f)
    with open(phenotype_json_path, 'r', encoding='utf-8') as f:
        all_phenotypes = json.load(f)

    if subcellular_locations is None:
        subcellular_locations = DEFAULT_SUBCELLULAR_LOCATIONS

    disease_encoder = LabelEncoder().fit(_normalize_list(all_diseases))
    phenotype_encoder = LabelEncoder().fit(_normalize_list(all_phenotypes))
    subcellular_location_encoder = LabelEncoder().fit(_normalize_list(subcellular_locations))

    return disease_encoder, phenotype_encoder, subcellular_location_encoder

# -----------------------------
# Dataset
# -----------------------------

class InteractionDataset(Dataset):
    def __init__(self, samples, disease_encoder, phenotype_encoder, subcellular_location_encoder):
        self.samples = samples
        self.disease_encoder = disease_encoder
        self.phenotype_encoder = phenotype_encoder
        self.subcellular_location_encoder = subcellular_location_encoder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]

        # Defensively handle empty lists
        def safe_transform(encoder, xs):
            xs = _normalize_list(xs)
            if len(xs) == 0:
                # If a field is empty, duplicate the first class index (0) to keep shape [1]
                return torch.LongTensor([0])
            return torch.LongTensor(encoder.transform(xs))

        compound_diseases = safe_transform(self.disease_encoder, sample.get('compound_diseases', []))
        compound_phenotypes = safe_transform(self.phenotype_encoder, sample.get('compound_phenotypes', []))
        compound_subcellular_locations = safe_transform(self.subcellular_location_encoder, sample.get('compound_subcellular_locations', []))

        protein_diseases = safe_transform(self.disease_encoder, sample.get('protein_diseases', []))
        protein_phenotypes = safe_transform(self.phenotype_encoder, sample.get('protein_phenotypes', []))
        protein_subcellular_locations = safe_transform(self.subcellular_location_encoder, sample.get('protein_subcellular_locations', []))

        return (
            compound_diseases,
            compound_phenotypes,
            compound_subcellular_locations,
            protein_diseases,
            protein_phenotypes,
            protein_subcellular_locations,
        ), torch.LongTensor([label])

# -----------------------------
# Samplers & Dataloaders
# -----------------------------

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_samples(positive_json_path: str,
                 negative_json_path: str,
                 text_json_path: str):
    positive_data = _load_json(positive_json_path)
    negative_data = _load_json(negative_json_path)
    text_data = _load_json(text_json_path)

    # label: 1 for positives, 0 for negatives; text samples also 0 (labels unused at inference)
    positive_samples = [(entry, 1) for entry in positive_data]
    negative_samples = [(entry, 0) for entry in negative_data]
    text_samples = [(entry, 0) for entry in text_data]

    all_samples = positive_samples + negative_samples
    return positive_samples, negative_samples, text_samples, all_samples


def get_dataloaders(
    disease_json_path: str,
    phenotype_json_path: str,
    positive_json_path: str,
    negative_json_path: str,
    text_json_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
):
    set_seed(seed)

    disease_encoder, phenotype_encoder, subcellular_location_encoder = build_label_encoders(
        disease_json_path, phenotype_json_path
    )

    _, _, text_samples, all_samples = make_samples(
        positive_json_path, negative_json_path, text_json_path
    )

    train_samples, val_samples = train_test_split(all_samples, test_size=val_split, random_state=seed)

    train_dataset = InteractionDataset(train_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)
    val_dataset = InteractionDataset(val_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)
    text_dataset = InteractionDataset(text_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'text': text_loader,
        'encoders': {
            'disease': disease_encoder,
            'phenotype': phenotype_encoder,
            'subcellular': subcellular_location_encoder,
        }
    }


def extract_names_from_text_json(text_json_path: str):
    """Return two lists aligned with text samples: protein_names, compound_names.
    Fallback to generated names if keys are missing.
    """
    data = _load_json(text_json_path)
    protein_names = []
    compound_names = []
    for i, entry in enumerate(data):
        protein_names.append(entry.get('protein_name', f'protein_{i}'))
        compound_names.append(entry.get('compound_name', f'compound_{i}'))
    return protein_names, compound_names





