import torch
import torch.nn as nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_isbns, embedding_dim):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.isbn_embedding = nn.Embedding(num_embeddings=num_isbns, embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, users, isbns):
        user_embeds = self.user_embedding(users.long())
        isbn_embeds = self.isbn_embedding(isbns.long())
        embeds = torch.cat([user_embeds, isbn_embeds], dim=1)
        x = torch.relu(self.fc1(embeds.view(embeds.size(0), -1)))
        x = self.fc2(x)
        return x
