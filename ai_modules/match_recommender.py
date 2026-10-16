import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
# 3. INFERENCE PIPELINE & EXPLAINABLE RECOMMENDER
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

    # ------------------------------------------------------------------
    # Data Normalization Helpers (Supports Dicts, UserModels, Legacy keys)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_field(user, key, default=None):
        if user is None:
            return default
        if isinstance(user, dict):
            val = user.get(key)
        else:
            val = getattr(user, key, None)
        return val if val is not None else default

    @staticmethod
    def _normalize_comm_token(token):
        if not token or not isinstance(token, str):
            return None
        t = token.strip().lower()
        if 'sign' in t:
            return 'sign_language'
        if 'text' in t or 'chat' in t:
            return 'text'
        if 'voice' in t or 'speech' in t or 'audio' in t:
            return 'voice'
        if 'video' in t:
            return 'video_call'
        if 'mixed' in t or 'multiple' in t:
            return 'mixed'
        return t

    @classmethod
    def _extract_comm_prefs(cls, user):
        """Extract and normalize all communication preferences from user dict or model."""
        raw_items = []
        # Check communication_preferences
        cp = cls._extract_field(user, 'communication_preferences')
        if isinstance(cp, (list, tuple, set)):
            raw_items.extend(cp)
        elif isinstance(cp, str) and cp.strip():
            raw_items.append(cp)

        # Check comm_prefs
        cp2 = cls._extract_field(user, 'comm_prefs')
        if isinstance(cp2, (list, tuple, set)):
            raw_items.extend(cp2)
        elif isinstance(cp2, str) and cp2.strip():
            raw_items.append(cp2)

        # Check legacy comm_pref
        cp3 = cls._extract_field(user, 'comm_pref')
        if isinstance(cp3, str) and cp3.strip():
            raw_items.append(cp3)

        normalized = set()
        for item in raw_items:
            norm = cls._normalize_comm_token(item)
            if norm:
                normalized.add(norm)
        return normalized

    @classmethod
    def _extract_interests(cls, user):
        """Extract normalized interest tokens as a lowercase set."""
        raw_ints = cls._extract_field(user, 'interests', [])
        if not isinstance(raw_ints, (list, tuple, set)):
            return set()
        return set(str(i).strip().lower() for i in raw_ints if i and isinstance(i, str) and str(i).strip())

    @classmethod
    def _extract_relationship_goal(cls, user):
        """Extract relationship goal string. Returns None if missing or empty."""
        raw = cls._extract_field(user, 'relationship_goal')
        if raw and isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned and cleaned.lower() not in {'not specified', 'none'}:
                return cleaned.lower()
        return None

    @classmethod
    def _extract_matching_preferences(cls, user):
        """Extract age and gender preference constraints."""
        mp = cls._extract_field(user, 'matching_preferences')
        if not isinstance(mp, dict):
            mp = {}
        min_age = mp.get('min_age') or cls._extract_field(user, 'pref_min_age')
        max_age = mp.get('max_age') or cls._extract_field(user, 'pref_max_age')
        pref_genders = mp.get('preferred_genders') or cls._extract_field(user, 'pref_gender')
        if isinstance(pref_genders, str):
            pref_genders = [pref_genders]
        elif not isinstance(pref_genders, (list, tuple, set)):
            pref_genders = []
        return {
            'min_age': min_age,
            'max_age': max_age,
            'preferred_genders': [str(g).strip().lower() for g in pref_genders if g]
        }

    # ------------------------------------------------------------------
    # Explainable Compatibility Calculation (0 to 100)
    # ------------------------------------------------------------------
    @classmethod
    def compute_compatibility_score(cls, user_a, user_b):
        """
        Calculate an explainable, normalized compatibility score (0 - 100).
        - Shared interests using Jaccard similarity (up to 45 pts)
        - Communication preference compatibility (up to 25 pts)
        - Relationship goal compatibility only when both exist (up to 15 pts)
        - Age & gender preferences where required data exists (up to 15 pts)
        - Missing data NEVER awards points.
        """
        if not user_a or not user_b:
            return 0

        # 1. Interests Jaccard Similarity (45 pts max)
        ints_a = cls._extract_interests(user_a)
        ints_b = cls._extract_interests(user_b)
        union_ints = ints_a | ints_b
        if ints_a and ints_b and union_ints:
            jaccard = len(ints_a & ints_b) / len(union_ints)
            interest_points = jaccard * 45.0
        else:
            interest_points = 0.0

        # 2. Communication Preference Compatibility (25 pts max)
        comm_a = cls._extract_comm_prefs(user_a)
        comm_b = cls._extract_comm_prefs(user_b)
        if comm_a and comm_b:
            shared_comm = comm_a & comm_b
            if shared_comm:
                # 12.5 pts for sharing at least one channel + up to 12.5 pts for full overlap
                overlap_ratio = len(shared_comm) / len(comm_a | comm_b)
                comm_points = 12.5 + (12.5 * overlap_ratio)
            else:
                comm_points = 0.0
        else:
            comm_points = 0.0

        # 3. Relationship Goal Compatibility (15 pts max)
        # ONLY awarded when BOTH users have a valid, non-empty relationship goal
        goal_a = cls._extract_relationship_goal(user_a)
        goal_b = cls._extract_relationship_goal(user_b)
        if goal_a and goal_b:
            if goal_a == goal_b:
                goal_points = 15.0
            else:
                goal_points = 0.0
        else:
            goal_points = 0.0

        # 4. Age & Gender Preferences (15 pts max: 7.5 age, 7.5 gender)
        pref_points = 0.0
        prefs_a = cls._extract_matching_preferences(user_a)
        raw_b_age = cls._extract_field(user_b, 'age')
        if raw_b_age is not None and prefs_a.get('min_age') is not None and prefs_a.get('max_age') is not None:
            try:
                b_age = float(raw_b_age)
                min_a = float(prefs_a['min_age'])
                max_a = float(prefs_a['max_age'])
                if min_a <= b_age <= max_a:
                    pref_points += 7.5
            except (ValueError, TypeError):
                pass

        raw_b_gender = cls._extract_field(user_b, 'gender')
        if raw_b_gender and prefs_a.get('preferred_genders'):
            b_gen = str(raw_b_gender).strip().lower()
            p_gens = prefs_a['preferred_genders']
            if 'any' in p_gens or b_gen in p_gens:
                pref_points += 7.5

        total = interest_points + comm_points + goal_points + pref_points
        return int(round(max(0.0, min(100.0, total))))

    # ------------------------------------------------------------------
    # Graph Edge Weight Calculation (0.0 to 1.0)
    # ------------------------------------------------------------------
    def _calculate_edge_weight(self, u1, u2):
        """
        Calculate normalized graph edge weight between 0.0 and 1.0.
        Only valid, non-empty fields contribute to edge weight.
        """
        weight = 0.0

        # 1. Interests Jaccard similarity (up to 0.50)
        i1 = self._extract_interests(u1)
        i2 = self._extract_interests(u2)
        if i1 and i2 and (i1 | i2):
            jaccard = len(i1 & i2) / len(i1 | i2)
            weight += 0.50 * jaccard

        # 2. Relationship goal compatibility (up to 0.25) - ONLY when both are valid and non-empty
        g1 = self._extract_relationship_goal(u1)
        g2 = self._extract_relationship_goal(u2)
        if g1 and g2 and g1 == g2:
            weight += 0.25

        # 3. Communication preferences compatibility (up to 0.25) - ONLY when both exist
        c1 = self._extract_comm_prefs(u1)
        c2 = self._extract_comm_prefs(u2)
        if c1 and c2 and (c1 | c2):
            comm_overlap = len(c1 & c2) / len(c1 | c2)
            weight += 0.25 * comm_overlap

        return float(min(max(weight, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Feature Extraction (Safe against None, missing keys, malformed data)
    # ------------------------------------------------------------------
    def _extract_features(self, user):
        f = np.zeros(self.num_features, dtype=np.float32)
        idx = 0

        # 1. Age (1) - safe parse
        raw_age = self._extract_field(user, 'age')
        try:
            age = float(raw_age) if raw_age is not None else 0.0
        except (ValueError, TypeError):
            age = 0.0
        f[idx] = min(max(age, 0.0) / 100.0, 1.0)
        idx += 1

        # 2. Gender (5)
        raw_gender = self._extract_field(user, 'gender')
        gender = str(raw_gender).strip() if raw_gender else 'Prefer not to say'
        genders = ['Male', 'Female', 'Non-binary', 'Other', 'Prefer not to say']
        if gender in genders:
            f[idx + genders.index(gender)] = 1.0
        idx += 5

        # 3. Relationship Goal (6)
        raw_goal = self._extract_field(user, 'relationship_goal')
        goal = str(raw_goal).strip() if raw_goal else ''
        goals = ['Friendship', 'Casual Dating', 'Serious Relationship', 'Long-Term Relationship', 'Marriage', 'Open to Options']
        for g_idx, g_opt in enumerate(goals):
            if goal and goal.lower() == g_opt.lower():
                f[idx + g_idx] = 1.0
        idx += 6

        # 4. Comm Prefs (5)
        prefs = self._extract_comm_prefs(user)
        c_opts = ['text', 'sign_language', 'voice', 'video_call', 'mixed']
        for p in prefs:
            if p in c_opts:
                f[idx + c_opts.index(p)] = 1.0
        idx += 5

        # 5. Matching Prefs: Age (2)
        mp = self._extract_matching_preferences(user)
        try:
            min_age = float(mp['min_age']) if mp.get('min_age') is not None else 0.0
        except (ValueError, TypeError):
            min_age = 0.0
        try:
            max_age = float(mp['max_age']) if mp.get('max_age') is not None else 0.0
        except (ValueError, TypeError):
            max_age = 0.0
        f[idx] = min(max(min_age, 0.0) / 100.0, 1.0); idx += 1
        f[idx] = min(max(max_age, 0.0) / 100.0, 1.0); idx += 1

        # 6. Matching Prefs: Gender (5)
        p_genders = mp.get('preferred_genders', [])
        for g in p_genders:
            if g.lower() == 'any':
                f[idx:idx + 5] = 1.0
            else:
                for gen_idx, gen_opt in enumerate(genders):
                    if g.lower() == gen_opt.lower():
                        f[idx + gen_idx] = 1.0
        idx += 5

        # 7. Interests (38 buckets safely hashed)
        u_ints = self._extract_interests(user)
        for ui in u_ints:
            bucket = abs(hash(ui)) % 38
            f[idx + bucket] = 1.0

        return f

    # ------------------------------------------------------------------
    # Ranking & Inference (With calibration verification)
    # ------------------------------------------------------------------
    def rank_candidates(self, current_user, candidates):
        if not candidates:
            return []

        # 1. Always compute explainable compatibility scores
        explainable_scores = [self.compute_compatibility_score(current_user, c) for c in candidates]

        # 2. If model is not loaded, use explainable scores directly
        if not self.model:
            for i, c in enumerate(candidates):
                c['_match_score'] = explainable_scores[i]
            candidates.sort(key=lambda c: c.get('_match_score', 0), reverse=True)
            return candidates

        # 3. Model is loaded: run inference safely with validation
        try:
            users = [current_user] + candidates
            N = len(users)

            features = np.array([self._extract_features(u) for u in users])

            adj = np.eye(N)
            for i in range(N):
                for j in range(i + 1, N):
                    sim = self._calculate_edge_weight(users[i], users[j])
                    if sim > 0.05:
                        adj[i, j] = sim
                        adj[j, i] = sim

            d_sum = np.sum(adj, axis=1)
            d_sum[d_sum == 0] = 1e-10
            D = np.diag(d_sum ** -0.5)
            adj_norm = np.dot(np.dot(D, adj), D)

            x_t = torch.FloatTensor(features)
            adj_t = torch.FloatTensor(adj_norm)

            idx_a = torch.zeros(N - 1, dtype=torch.long)
            idx_b = torch.arange(1, N, dtype=torch.long)

            with torch.no_grad():
                scores = self.model(x_t, adj_t, idx_a, idx_b)

            if (N - 1) == 1:
                raw_scores = [scores.item()]
            else:
                raw_scores = scores.tolist()

            # Check if neural model produces uncalibrated or nearly constant predictions
            score_std = float(np.std(raw_scores)) if len(raw_scores) > 1 else 0.0
            score_range = float(max(raw_scores) - min(raw_scores)) if len(raw_scores) > 1 else 0.0

            if score_std < 0.03 or score_range < 0.05:
                # Model predictions are uncalibrated / near-constant.
                # Use explainable feature-based scores directly.
                for i, c in enumerate(candidates):
                    c['_match_score'] = explainable_scores[i]
            else:
                # Model has discriminative variance: blend with explainable ground truth
                for i, c in enumerate(candidates):
                    model_score_pct = raw_scores[i] * 100.0
                    blended = 0.5 * explainable_scores[i] + 0.5 * model_score_pct
                    c['_match_score'] = int(round(max(0.0, min(100.0, blended))))

        except Exception as e:
            # On any inference failure, fall back cleanly to explainable scores
            logger.warning(f"[MatchRecommender] Inference exception, using explainable scores: {e}")
            for i, c in enumerate(candidates):
                c['_match_score'] = explainable_scores[i]

        candidates.sort(key=lambda c: c.get('_match_score', 0), reverse=True)
        return candidates

    def baseline_rank(self, current_user, candidates):
        """
        Explainable baseline ranking based on feature-based compatibility.
        """
        for c in candidates:
            c['_match_score'] = self.compute_compatibility_score(current_user, c)
        candidates.sort(key=lambda c: c.get('_match_score', 0), reverse=True)
        return candidates
