import json
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import os

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

        compound_diseases = self.disease_encoder.transform(sample['compound_diseases'])
        compound_phenotypes = self.phenotype_encoder.transform(sample['compound_phenotypes'])
        compound_subcellular_locations = self.subcellular_location_encoder.transform(sample['compound_subcellular_locations'])

        protein_diseases = self.disease_encoder.transform(sample['protein_diseases'])
        protein_phenotypes = self.phenotype_encoder.transform(sample['protein_phenotypes'])
        protein_subcellular_locations = self.subcellular_location_encoder.transform(sample['protein_subcellular_locations'])
        return (torch.LongTensor(compound_diseases), 
                torch.LongTensor(compound_phenotypes), 
                torch.LongTensor(compound_subcellular_locations),
                torch.LongTensor(protein_diseases), 
                torch.LongTensor(protein_phenotypes), 
                torch.LongTensor(protein_subcellular_locations)), torch.LongTensor([label])

def load_data(data_dir):
    # Load necessary files
    with open(os.path.join(data_dir, 'new_phenotype.json'), 'r', encoding='utf-8') as f:
        all_phenotypes = json.load(f)
    with open(os.path.join(data_dir, 'disease_list.json'), 'r', encoding='utf-8') as f:
        all_diseases = json.load(f)

    all_subcellular_locations = [
        "Nucleus", "Cytoplasm", "Mitochondria", "Endoplasmic Reticulum", "Golgi Apparatus", 
        "Lysosome", "Plasma Membrane", "Nuclear Membrane", "Peroxisome", "Nucleolus", 
        "Cytoskeleton", "Vacuole", "Chloroplast", "Plasmid", "Ribosome", "Flagellum", 
        "Microvilli", "Vesicle", "Thylakoid", "Centrosome", "Synaptic Vesicle", 
        "Endosome", "Nuclear Pore Complex"
    ]

    # Encoders
    disease_encoder = LabelEncoder().fit(all_diseases)
    phenotype_encoder = LabelEncoder().fit(all_phenotypes)
    subcellular_location_encoder = LabelEncoder().fit(all_subcellular_locations)

    # Load samples
    with open(os.path.join(data_dir, 'test_samples_dataset.json'), 'r') as f:
        text_data = json.load(f)
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    positive_data = dataset[:870]
    negative_data = dataset[870:]

    positive_samples = [(entry, 1) for entry in positive_data]
    negative_samples = [(entry, 0) for entry in negative_data]
    text_samples = [(entry, 0) for entry in text_data]

    all_samples = positive_samples + negative_samples
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)

    train_dataset = InteractionDataset(train_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)
    val_dataset = InteractionDataset(val_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)
    text_dataset = InteractionDataset(text_samples, disease_encoder, phenotype_encoder, subcellular_location_encoder)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    text_dataloader = DataLoader(text_dataset, batch_size=16, shuffle=False)

    text_compound_names = [sample['compound'] for sample, _ in text_samples]
    text_protein_names = [sample['protein'] for sample, _ in text_samples]

    return train_dataloader, val_dataloader, text_dataloader, text_compound_names, text_protein_names, disease_encoder, phenotype_encoder, subcellular_location_encoder
