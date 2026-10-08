import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from match_recommender import MatchPredictorGCNTransformer, MatchRecommender

def generate_synthetic_data(num_users=200):
    print(f"Generating {num_users} synthetic users for training...")
    genders = ['Male', 'Female', 'Non-binary', 'Other']
    goals = ['Friendship', 'Casual Dating', 'Serious Relationship', 'Long-Term Relationship', 'Marriage']
    comm = ['Text Chat', 'Voice', 'Video Call', 'Sign Language']
    interests_pool = ["Cricket", "Music", "Movies", "Reading", "Technology", "Travelling", "Art", "Sports"]
    
    users = []
    for i in range(num_users):
        u = {
            '_id': str(i),
            'age': random.randint(18, 65),
            'gender': random.choice(genders),
            'relationship_goal': random.choice(goals),
            'communication_preferences': random.sample(comm, random.randint(1, 3)),
            'interests': random.sample(interests_pool, random.randint(2, 6)),
            'matching_preferences': {
                'min_age': random.randint(18, 30),
                'max_age': random.randint(35, 65),
                'preferred_genders': [random.choice(genders), 'Any']
            }
        }
        users.append(u)
    return users

def train_model():
    print("Initializing GCN + Transformer Encoder matching model training...")
    
    # 1. Generate synthetic data
    users = generate_synthetic_data(num_users=300)
    recommender = MatchRecommender(model_path="dummy") # Just to use feature extraction
    
    features = np.array([recommender._extract_features(u) for u in users])
    
    # 2. Graph Construction
    N = len(users)
    adj = np.eye(N)
    for i in range(N):
        for j in range(i+1, N):
            sim = recommender._calculate_edge_weight(users[i], users[j])
            if sim > 0.1:
                adj[i, j] = sim
                adj[j, i] = sim
                
    d_sum = np.sum(adj, axis=1)
    d_sum[d_sum == 0] = 1e-10
    D = np.diag(d_sum ** -0.5)
    adj_norm = np.dot(np.dot(D, adj), D)
    
    x_t = torch.FloatTensor(features)
    adj_t = torch.FloatTensor(adj_norm)
    
    # 3. Generate Labels (Supervised pairs)
    # For synthetic training, we'll label high edge similarity as highly compatible (1.0), low as (0.0)
    print("Generating labels based on synthetic graph...")
    pairs_a, pairs_b, labels = [], [], []
    for i in range(N):
        for j in range(i+1, N):
            # Sample pairs to avoid O(N^2) explosion
            if random.random() < 0.05:
                pairs_a.append(i)
                pairs_b.append(j)
                score = adj[i, j] # raw similarity
                labels.append(1.0 if score > 1.0 else (score/1.0 if score > 0 else 0.0))
                
    idx_a = torch.tensor(pairs_a, dtype=torch.long)
    idx_b = torch.tensor(pairs_b, dtype=torch.long)
    y_t = torch.FloatTensor(labels)
    
    # 4. Train Model
    model = MatchPredictorGCNTransformer(num_features=62, hidden_dim=64, num_heads=4, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    epochs = 50
    print(f"Training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x_t, adj_t, idx_a, idx_b)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    # 5. Save Model
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'matching_gcn_transformer.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
