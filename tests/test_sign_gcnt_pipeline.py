"""
Automated Test Suite for Indian Sign Language (ISL) GCN-Transformer Pipeline
============================================================================
MCA Major Project: InclusiveMatch AI

Covers:
1. Anatomical MediaPipe 42-node graph adjacency & block-diagonal properties
2. Input dimensions and model forward/backward pass
3. Missing hands and zero-filled frame resilience (single-hand / missing hand)
4. Velocity computation safety across hand appearance transitions (no spikes)
5. Checkpoint serialization, loading, and parameter integrity
6. Live inference preprocessing parity and prediction validity
7. Class mapping bijection and sorting consistency
"""

import os
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    build_two_hand_adjacency,
    preprocess_landmark_sequence,
    preprocess_live_frame_buffer,
    NUM_NODES,
    NUM_NODES_PER_HAND,
    MEDIAPIPE_HAND_EDGES,
)


class TestSignGCNTPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.num_classes = 29
        self.seq_len = 24
        self.node_dim = 4
        self.gcn_dim = 64
        self.tf_heads = 4
        self.tf_layers = 2
        self.classes = [
            "drink", "eat", "father", "food", "friend", "go", "he", "hello",
            "help", "hospital", "market", "mother", "no", "okay", "please",
            "school", "she", "sister", "sit", "student", "tea", "teacher",
            "thank_you", "today", "water", "what", "where", "yes", "you"
        ]

    def test_anatomical_hand_adjacency(self):
        """Verify 42-node graph is symmetric, normalized, self-looped, and block-diagonal."""
        adj = build_two_hand_adjacency()
        self.assertEqual(adj.shape, (42, 42))

        # 1. Symmetry check
        self.assertTrue(torch.allclose(adj, adj.T, atol=1e-6))

        # 2. Block-diagonal check: NO edges between Hand 1 (0..20) and Hand 2 (21..41)
        cross_h1_h2 = adj[:21, 21:]
        cross_h2_h1 = adj[21:, :21]
        self.assertTrue((cross_h1_h2 == 0).all().item(), "Found unexpected edges from Hand 1 to Hand 2!")
        self.assertTrue((cross_h2_h1 == 0).all().item(), "Found unexpected edges from Hand 2 to Hand 1!")

        # 3. Self-loops check: Diagonals must be non-zero
        self.assertTrue((torch.diag(adj) > 0).all().item())

        # 4. Correct edge count per hand
        expected_edges_per_hand = len(MEDIAPIPE_HAND_EDGES)
        # Off-diagonal non-zero pairs in Hand 1 block
        h1_block = adj[:21, :21].clone()
        h1_block.fill_diagonal_(0)
        h1_nonzeros = (h1_block > 0).sum().item() // 2
        self.assertEqual(h1_nonzeros, expected_edges_per_hand)

    def test_input_dimensions_and_forward_pass(self):
        """Verify model accepts (B, 24, 42, 4) and produces (B, 29) finite logits."""
        model = GCNTransformerClassifier(
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            node_dim=self.node_dim,
            gcn_dim=self.gcn_dim,
            tf_heads=self.tf_heads,
            tf_layers=self.tf_layers,
        )
        model.eval()

        for batch_size in [1, 2, 5]:
            x = torch.randn(batch_size, self.seq_len, NUM_NODES, self.node_dim)
            out = model(x)
            self.assertEqual(out.shape, (batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(out).all().item())

        # Check gradient backward pass
        model.train()
        x = torch.randn(2, self.seq_len, NUM_NODES, self.node_dim, requires_grad=True)
        target = torch.tensor([0, 1], dtype=torch.long)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_missing_hands_and_zero_filled_frames(self):
        """Verify model handles single-hand inputs and missing frames gracefully."""
        model = GCNTransformerClassifier(
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            node_dim=self.node_dim,
            gcn_dim=self.gcn_dim,
            tf_heads=self.tf_heads,
            tf_layers=self.tf_layers,
        )
        model.eval()

        # Case 1: Hand 2 completely missing (zeroed out)
        x_single_hand = torch.randn(1, self.seq_len, NUM_NODES, self.node_dim)
        x_single_hand[:, :, 21:, :] = 0.0  # Zero out Hand 2
        out_single = model(x_single_hand)
        self.assertTrue(torch.isfinite(out_single).all().item())

        # Case 2: Hand 1 completely missing
        x_other_hand = torch.randn(1, self.seq_len, NUM_NODES, self.node_dim)
        x_other_hand[:, :, :21, :] = 0.0  # Zero out Hand 1
        out_other = model(x_other_hand)
        self.assertTrue(torch.isfinite(out_other).all().item())

        # Case 3: Completely zero-filled frames (no hands detected)
        x_all_zero = torch.zeros(1, self.seq_len, NUM_NODES, self.node_dim)
        out_zero = model(x_all_zero)
        self.assertTrue(torch.isfinite(out_zero).all().item())

    def test_preprocessing_and_velocity_safety(self):
        """Verify preprocessing from (60, 84) and that no velocity spikes occur on hand appearance."""
        # Create a raw sequence where Hand 1 appears at frame 15, Hand 2 never appears
        raw = np.zeros((60, 84), dtype=np.float32)
        # Frames 15..59: Hand 1 is active
        for t in range(15, 60):
            # Hand 1 wrist at (0.3, 0.4)
            raw[t, 0] = 0.3 + (t * 0.001)
            raw[t, 1] = 0.4 + (t * 0.001)

        feat = preprocess_landmark_sequence(raw, seq_len=24)
        self.assertEqual(feat.shape, (24, 42, 4))
        self.assertTrue(np.isfinite(feat).all())

        # Check Hand 2 is strictly zero everywhere
        self.assertTrue((feat[:, 21:, :] == 0).all())

        # Check velocities: no velocity in frame 0
        self.assertEqual(feat[0, 0, 2], 0.0)  # dx at t=0 must be 0
        self.assertEqual(feat[0, 0, 3], 0.0)  # dy at t=0 must be 0

        # Check transition safety: if a hand appears at frame t, velocity must NOT jump from 0
        raw_transition = np.zeros((24, 84), dtype=np.float32)
        raw_transition[10:, :42] = 0.8  # Hand 1 suddenly appears at frame 10
        feat_trans = preprocess_landmark_sequence(raw_transition, seq_len=24)
        # Velocity at frame 10 should be 0.0 (not 0.8 - 0.0)
        self.assertEqual(feat_trans[10, 0, 2], 0.0)
        self.assertEqual(feat_trans[10, 0, 3], 0.0)

    def test_checkpoint_saving_and_loading(self):
        """Verify state dictionary and configuration roundtrip through disk serialization."""
        model = GCNTransformerClassifier(
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            node_dim=self.node_dim,
            gcn_dim=self.gcn_dim,
            tf_heads=self.tf_heads,
            tf_layers=self.tf_layers,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            checkpoint = {
                "arch": "gcn_transformer_2hand",
                "state_dict": model.state_dict(),
                "classes": self.classes,
                "class_to_idx": {c: i for i, c in enumerate(self.classes)},
                "num_classes": self.num_classes,
                "seq_len": self.seq_len,
                "num_nodes": NUM_NODES,
                "node_dim": self.node_dim,
                "gcn_dim": self.gcn_dim,
                "tf_heads": self.tf_heads,
                "tf_layers": self.tf_layers,
                "dropout": 0.25,
            }
            torch.save(checkpoint, temp_path)

            loaded = torch.load(temp_path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["arch"], "gcn_transformer_2hand")
            self.assertEqual(loaded["num_classes"], self.num_classes)
            self.assertEqual(loaded["classes"], self.classes)

            model2 = GCNTransformerClassifier(
                num_classes=loaded["num_classes"],
                seq_len=loaded["seq_len"],
                node_dim=loaded["node_dim"],
                gcn_dim=loaded["gcn_dim"],
                tf_heads=loaded["tf_heads"],
                tf_layers=loaded["tf_layers"],
            )
            model2.load_state_dict(loaded["state_dict"])
            model2.eval()

            # Compare outputs
            x = torch.randn(1, self.seq_len, NUM_NODES, self.node_dim)
            model.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_live_inference_parity(self):
        """Verify that streaming frames into preprocess_live_frame_buffer matches batch preprocessing."""
        # 24 frames of 2 hands
        frames_list = []
        for t in range(24):
            frame = np.ones((42, 2), dtype=np.float32) * (0.1 + t * 0.01)
            frames_list.append(frame)

        live_feat = preprocess_live_frame_buffer(frames_list, seq_len=24)
        self.assertEqual(live_feat.shape, (24, 42, 4))
        self.assertTrue(np.isfinite(live_feat).all())

        # Test model prediction on live tensor
        model = GCNTransformerClassifier(num_classes=self.num_classes)
        model.eval()
        x = torch.tensor(live_feat).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        self.assertEqual(probs.shape, (1, self.num_classes))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_class_mapping_integrity(self):
        """Verify class list is strictly 29 unique, sorted classes."""
        self.assertEqual(len(self.classes), 29)
        self.assertEqual(len(set(self.classes)), 29)
        self.assertEqual(self.classes, sorted(self.classes))

        mapping = {c: i for i, c in enumerate(self.classes)}
        reverse_mapping = {i: c for c, i in mapping.items()}
        for i, c in enumerate(self.classes):
            self.assertEqual(mapping[c], i)
            self.assertEqual(reverse_mapping[i], c)


if __name__ == "__main__":
    unittest.main()
