import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

# ==========================================
# 1. GRAPH CONVOLUTION
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


# ==========================================
# 2. GCN + TRANSFORMER ENCODER MODEL
# ==========================================
class MatchPredictorGCNTransformer(nn.Module):
    def __init__(self, num_features=62, hidden_dim=64, num_heads=4, num_layers=2):
        super(MatchPredictorGCNTransformer, self).__init__()
        
        # GCN
        self.gcn1 = GraphConvolution(num_features, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj, user_idx_a, user_idx_b):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, 0.2, training=self.training)
        emb = self.gcn2(x, adj)
        
        emb_a = emb[user_idx_a]
        emb_b = emb[user_idx_b]
        
        seq = torch.stack([emb_a, emb_b], dim=1) 
        seq_out = self.transformer(seq)
        
        t_emb_a = seq_out[:, 0, :]
        t_emb_b = seq_out[:, 1, :]
        
        cat_emb = torch.cat([t_emb_a, t_emb_b], dim=1)
        out = F.relu(self.fc1(cat_emb))
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze()


# ==========================================
# 3. INFERENCE PIPELINE
# ==========================================
class MatchRecommender:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'matching_gcn_transformer.pt')
        else:
            self.model_path = model_path
            
        self.model = None
        self.num_features = 62
        
        self.all_interests = [
            "Cricket", "Football", "Basketball", "Badminton", "Tennis", "Swimming", "Running", "Gym", "Yoga", "Cycling",
            "Movies", "Music", "Singing", "Dancing", "Photography", "Gaming", "Reading", "Watching TV", "Anime",
            "Drawing", "Painting", "Writing", "Cooking", "Baking", "Crafts", "Design",
            "Travelling", "Hiking", "Nature", "Adventure", "Food", "Fashion", "Shopping",
            "Technology", "Programming", "AI & Machine Learning", "Science", "Education", "Learning Languages",
            "Pets", "Volunteering", "Community Activities", "Meditation", "Spirituality"
        ]
        
        if os.path.exists(self.model_path):
            try:
                self.model = MatchPredictorGCNTransformer(num_features=self.num_features)
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"[MatchRecommender] Loaded AI model from {self.model_path}")
            except Exception as e:
                print(f"[MatchRecommender] Error loading model: {e}")
                self.model = None
        else:
            print(f"[MatchRecommender] Model not found at {self.model_path}. Using fallback.")

    def _extract_features(self, user):
        f = np.zeros(self.num_features, dtype=np.float32)
        idx = 0
        
        # 1. Age (1)
        age = float(user.get('age', 25))
        f[idx] = min(age / 100.0, 1.0); idx += 1
        
        # 2. Gender (5)
        gender = user.get('gender', 'Prefer not to say')
        genders = ['Male', 'Female', 'Non-binary', 'Other', 'Prefer not to say']
        if gender in genders: f[idx + genders.index(gender)] = 1.0
        idx += 5
        
        # 3. Relationship Goal (6)
        goal = user.get('relationship_goal', 'Friendship')
        goals = ['Friendship', 'Casual Dating', 'Serious Relationship', 'Long-Term Relationship', 'Marriage', 'Open to Options']
        if goal in goals: f[idx + goals.index(goal)] = 1.0
        idx += 6
        
        # 4. Comm Prefs (5)
        prefs = user.get('communication_preferences', [])
        c_opts = ['Text Chat', 'Sign Language', 'Voice', 'Video Call', 'Mixed / Multiple']
        for p in prefs:
            if p in c_opts: f[idx + c_opts.index(p)] = 1.0
        idx += 5
        
        # 5. Matching Prefs: Age (2)
        m_prefs = user.get('matching_preferences', {})
        f[idx] = min(float(m_prefs.get('min_age', 18))/100.0, 1.0); idx += 1
        f[idx] = min(float(m_prefs.get('max_age', 99))/100.0, 1.0); idx += 1
        
        # 6. Matching Prefs: Gender (5)
        p_genders = m_prefs.get('preferred_genders', ['Any'])
        for g in p_genders:
            if g in genders: 
                f[idx + genders.index(g)] = 1.0
            elif g == 'Any':
                f[idx:idx+5] = 1.0
        idx += 5
        
        # 7. Interests (~40+ dims, we cap at 37 for remaining space up to 62, 62 - 25 = 37)
        # Actually len(all_interests) = 44. Let's dynamically adjust num_features.
        # Wait, I set num_features=62. 1+5+6+5+2+5 = 24. 62 - 24 = 38. 
        # I'll just hash interests into 38 buckets to keep it fixed size safely.
        u_ints = user.get('interests', [])
        for ui in u_ints:
            bucket = hash(ui) % 38
            f[idx + bucket] = 1.0
            
        return f

    def _calculate_edge_weight(self, u1, u2):
        score = 0.0
        # Common interests
        i1 = set(u1.get('interests', []))
        i2 = set(u2.get('interests', []))
        if i1 and i2:
            score += len(i1.intersection(i2)) / max(len(i1), len(i2), 1)
            
        # Same goal
        if u1.get('relationship_goal') == u2.get('relationship_goal'):
            score += 0.5
            
        # Comm prefs
        c1 = set(u1.get('communication_preferences', []))
        c2 = set(u2.get('communication_preferences', []))
        if c1.intersection(c2):
            score += 0.5
            
        return min(score, 1.0)

    def rank_candidates(self, current_user, candidates):
        if not candidates:
            return []
            
        if not self.model:
            return self.baseline_rank(current_user, candidates)

        users = [current_user] + candidates
        N = len(users)
        
        features = np.array([self._extract_features(u) for u in users])
        
        # Meaningful Graph Construction
        adj = np.eye(N)
        for i in range(N):
            for j in range(i+1, N):
                sim = self._calculate_edge_weight(users[i], users[j])
                if sim > 0.1: # Only add edge if there's some similarity
                    adj[i, j] = sim
                    adj[j, i] = sim
                    
        # Degree normalization for GCN
        d_sum = np.sum(adj, axis=1)
        d_sum[d_sum == 0] = 1e-10
        D = np.diag(d_sum ** -0.5)
        adj_norm = np.dot(np.dot(D, adj), D)
        
        x_t = torch.FloatTensor(features)
        adj_t = torch.FloatTensor(adj_norm)
        
        idx_a = torch.zeros(N-1, dtype=torch.long)
        idx_b = torch.arange(1, N, dtype=torch.long)
        
        with torch.no_grad():
            scores = self.model(x_t, adj_t, idx_a, idx_b)
            
        if (N-1) == 1:
            scores = [scores.item()]
        else:
            scores = scores.tolist()
            
        for i, candidate in enumerate(candidates):
            candidate['_match_score'] = int(scores[i] * 100)
            
        candidates.sort(key=lambda c: c['_match_score'], reverse=True)
        return candidates

    def baseline_rank(self, current_user, candidates):
        my_interests = set(current_user.get('interests', []))
        for c in candidates:
            their_interests = set(c.get('interests', []))
            overlap = len(my_interests.intersection(their_interests))
            c['_match_score'] = 50 + min(overlap * 10, 50)
        candidates.sort(key=lambda c: c['_match_score'], reverse=True)
        return candidates
