import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, L, D]
        attn_weights = self.attention(x)                # [B, L, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = x * attn_weights                     # [B, L, D]
        return weighted.sum(dim=1)                      # [B, D]


class InteractionPredictionModel(nn.Module):
    def __init__(self,
                 disease_embedding_dim=32,
                 phenotype_embedding_dim=16,
                 subcellular_embedding_dim=16,
                 num_diseases=13752,
                 num_phenotypes=17393,
                 num_subcellular_locations=30,
                 hidden_dim1=128,
                 hidden_dim2=64,
                 dropout_rate=0.3):
        super().__init__()

        # Embeddings
        self.disease_embedding = nn.Embedding(num_diseases, disease_embedding_dim)
        self.phenotype_embedding = nn.Embedding(num_phenotypes, phenotype_embedding_dim)
        self.subcellular_embedding = nn.Embedding(num_subcellular_locations, subcellular_embedding_dim)

        # Attentions (output dims equal respective embedding dims)
        self.disease_attention = AttentionLayer(disease_embedding_dim, max(2, disease_embedding_dim // 2))
        self.phenotype_attention = AttentionLayer(phenotype_embedding_dim, max(2, phenotype_embedding_dim // 2))
        self.subcellular_attention = AttentionLayer(subcellular_embedding_dim, max(2, subcellular_embedding_dim // 2))

        # Input: 3 modalities for compound + 3 for protein
        input_dim = (disease_embedding_dim + phenotype_embedding_dim + subcellular_embedding_dim) * 2

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, compound_diseases, compound_phenotypes, compound_subcellular_locations,
                protein_diseases, protein_phenotypes, protein_subcellular_locations):
        # Compound branches
        c_dis = self.disease_attention(self.disease_embedding(compound_diseases))
        c_phe = self.phenotype_attention(self.phenotype_embedding(compound_phenotypes))
        c_sub = self.subcellular_attention(self.subcellular_embedding(compound_subcellular_locations))

        # Protein branches
        p_dis = self.disease_attention(self.disease_embedding(protein_diseases))
        p_phe = self.phenotype_attention(self.phenotype_embedding(protein_phenotypes))
        p_sub = self.subcellular_attention(self.subcellular_embedding(protein_subcellular_locations))

        compound_feat = torch.cat([c_dis, c_phe, c_sub], dim=1)
        protein_feat = torch.cat([p_dis, p_phe, p_sub], dim=1)
        x = torch.cat([compound_feat, protein_feat], dim=1)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # logits
        return x


class InteractionPredictionModel_NoAttention(nn.Module):
    def __init__(self,
                 disease_embedding_dim=32,
                 phenotype_embedding_dim=16,
                 subcellular_embedding_dim=16,
                 num_diseases=13752,
                 num_phenotypes=17393,
                 num_subcellular_locations=30,
                 hidden_dim1=128,
                 hidden_dim2=64,
                 dropout_rate=0.1):
        super().__init__()

        self.disease_embedding = nn.Embedding(num_diseases, disease_embedding_dim)
        self.phenotype_embedding = nn.Embedding(num_phenotypes, phenotype_embedding_dim)
        self.subcellular_embedding = nn.Embedding(num_subcellular_locations, subcellular_embedding_dim)

        # Mean-pool per modality
        input_dim = (disease_embedding_dim + phenotype_embedding_dim + subcellular_embedding_dim) * 2
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, compound_diseases, compound_phenotypes, compound_subcellular_locations,
                protein_diseases, protein_phenotypes, protein_subcellular_locations):
        # Compound
        c_dis = self.disease_embedding(compound_diseases).mean(dim=1)
        c_phe = self.phenotype_embedding(compound_phenotypes).mean(dim=1)
        c_sub = self.subcellular_embedding(compound_subcellular_locations).mean(dim=1)

        # Protein
        p_dis = self.disease_embedding(protein_diseases).mean(dim=1)
        p_phe = self.phenotype_embedding(protein_phenotypes).mean(dim=1)
        p_sub = self.subcellular_embedding(protein_subcellular_locations).mean(dim=1)

        compound_feat = torch.cat([c_dis, c_phe, c_sub], dim=1)
        protein_feat = torch.cat([p_dis, p_phe, p_sub], dim=1)
        x = torch.cat([compound_feat, protein_feat], dim=1)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # logits
        return x
