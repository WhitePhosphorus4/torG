import torch

ATOM_FEATURE_DICT = 3

class AtomEncoder(torch.nn.Module):
    """
    Output:
        原子表征
        atom representations
    """

    def __init__(self, emb_dim):
        """Atom Encoder Module"""

        super(AtomEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.atom_embedding = torch.nn.Embedding(
            num_embeddings=len(ATOM_FEATURE_DICT),
            embedding_dim=self.emb_dim,
            padding_idx=0
        )

    def forward(self, atom_feature):
        """
        Input:
            atom_feature:
                atom features
        Output:
            atom_embedding:
                atom embedding
        """

        atom_embedding = self.atom_embedding(atom_feature)
        return atom_embedding