import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_input = x * attn_weights
        return weighted_input.sum(dim=1)

class InteractionPredictionModel(nn.Module):
    def __init__(self, 
                 disease_embedding_dim=32, 
                 phenotype_embedding_dim=16, 
                 subcellular_embedding_dim=16, 
                 num_diseases=13000, 
                 num_phenotypes=14000, 
                 num_subcellular_locations=30, 
                 hidden_dim1=128, 
                 hidden_dim2=64, 
                 dropout_rate=0.3):
        super(InteractionPredictionModel, self).__init__()

        self.disease_embedding = nn.Embedding(num_diseases, disease_embedding_dim)
        self.phenotype_embedding = nn.Embedding(num_phenotypes, phenotype_embedding_dim)
        self.subcellular_embedding = nn.Embedding(num_subcellular_locations, subcellular_embedding_dim)

        self.disease_attention = AttentionLayer(disease_embedding_dim, disease_embedding_dim // 2)
        self.phenotype_attention = AttentionLayer(phenotype_embedding_dim, phenotype_embedding_dim // 2)
        self.subcellular_attention = AttentionLayer(subcellular_embedding_dim, subcellular_embedding_dim // 2)

        input_dim = (disease_embedding_dim // 2 + phenotype_embedding_dim // 2 + subcellular_embedding_dim // 2) * 2
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, compound_diseases, compound_phenotypes, compound_subcellular_locations,
                protein_diseases, protein_phenotypes, protein_subcellular_locations):

        compound_disease_emb = self.disease_embedding(compound_diseases)
        compound_phenotype_emb = self.phenotype_embedding(compound_phenotypes)
        compound_subcellular_emb = self.subcellular_embedding(compound_subcellular_locations)

        compound_disease_att = self.disease_attention(compound_disease_emb)
        compound_phenotype_att = self.phenotype_attention(compound_phenotype_emb)
        compound_subcellular_att = self.subcellular_attention(compound_subcellular_emb)

        protein_disease_emb = self.disease_embedding(protein_diseases)
        protein_phenotype_emb = self.phenotype_embedding(protein_phenotypes)
        protein_subcellular_emb = self.subcellular_embedding(protein_subcellular_locations)

        protein_disease_att = self.disease_attention(protein_disease_emb)
        protein_phenotype_att = self.phenotype_attention(protein_phenotype_emb)
        protein_subcellular_att = self.subcellular_attention(protein_subcellular_emb)

        compound_features = torch.cat([compound_disease_att, compound_phenotype_att, compound_subcellular_att], dim=1)
        protein_features = torch.cat([protein_disease_att, protein_phenotype_att, protein_subcellular_att], dim=1)

        combined_features = torch.cat([compound_features, protein_features], dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return self.sigmoid(x)
