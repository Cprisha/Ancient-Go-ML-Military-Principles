import os
import json
import numpy as np
from network import (
    PolicyNet, CandidateNet,
    board_to_spatial_planes
)

PRINCIPLE_CATALOG = {
    f"P-T{t:02d}-{p:03d}": idx
    for idx, (t, p) in enumerate((t, p) for t in range(1, 10) for p in range(1, 35))
    if idx < 281
}

def extract_yin_yang_context(board_evidence_dict):
    """Translates board evidence JSON into the 31-channel non-spatial feature vector."""
    features = np.zeros(31, dtype=np.float32)
    if not board_evidence_dict:
        return features

    features[0] = float(board_evidence_dict.get("friendly_neighbour_count", 0))
    features[1] = float(board_evidence_dict.get("enemy_neighbour_count", 0))
    features[2] = float(board_evidence_dict.get("nearby_friendly_count", 0))
    features[3] = float(board_evidence_dict.get("nearby_enemy_count", 0))
    features[4] = float(board_evidence_dict.get("strong_enemy_group_count", 0))
    features[5] = float(board_evidence_dict.get("enemy_atari_group_count", 0))
    features[6] = float(board_evidence_dict.get("friendly_atari_group_count", 0))
    features[7] = 1.0 if board_evidence_dict.get("near_empty", False) else 0.0
    features[8] = float(board_evidence_dict.get("captured_count", 0))

    own_group = board_evidence_dict.get("own_group_after", {})
    if isinstance(own_group, dict):
        features[9] = float(own_group.get("size", 0))
        features[10] = float(own_group.get("liberties", 0))

    features[11] = 1.0 if board_evidence_dict.get("ko_candidate", False) else 0.0
    features[12] = float(board_evidence_dict.get("move_number", 0)) / 200.0
    features[13] = float(board_evidence_dict.get("total_moves", 0)) / 200.0

    sb = board_evidence_dict.get("stone_counts_before", {})
    if isinstance(sb, dict):
        features[14] = float(sb.get("black", 0)) / 180.0
        features[15] = float(sb.get("white", 0)) / 180.0

    sa = board_evidence_dict.get("stone_counts_after", {})
    if isinstance(sa, dict):
        features[16] = float(sa.get("black", 0)) / 180.0
        features[17] = float(sa.get("white", 0)) / 180.0

    return features


class AncientGoOracle:
    """Live inference engine evaluating moves based on classical Go principles."""
    def __init__(self, weights_path="model_weights.npz"):
        self.intuitive_mind = PolicyNet(in_channels=6, num_res_blocks=6)
        self.principle_mind = CandidateNet(in_channels=7, num_res_blocks=6, num_principles=281)
        self.load_ancient_knowledge(weights_path)

    def load_ancient_knowledge(self, weights_path):
        """Restores model weights from binary storage into networks."""
        if not os.path.exists(weights_path):
            print(f"Warning: {weights_path} not found. Running with uninitialized weights.")
            return

        archive = np.load(weights_path)
        
        for idx, layer in enumerate(self.intuitive_mind.layers):
            for p_idx in range(len(layer.params)):
                key = f"policy_layer_{idx}_p_{p_idx}"
                if key in archive:
                    layer.params[p_idx][:] = archive[key]

        for idx, layer in enumerate(self.principle_mind.layers):
            for p_idx in range(len(layer.params)):
                key = f"candidate_layer_{idx}_p_{p_idx}"
                if key in archive:
                    layer.params[p_idx][:] = archive[key]

        print("Ancient weights successfully loaded.")

    def seek_best_move(self, current_board, active_player, legal_coords, tactical_context, principle_code="P-T05-007"):
        """
        Evaluates legal candidate moves against classical principle alignments.
        
        Args:
            current_board: 19x19 array (1 for Black, -1 for White, 0 for Empty).
            active_player: 'B' or 'W'.
            legal_coords: List of (row, col) tuples representing legal intersections.
            tactical_context: Dictionary containing local board evidence/Yin-Yang metrics.
            principle_code: Target principle string ID (e.g. 'P-T05-007').
            
        Returns:
            best_coordinate: Selected (r, c) tuple or 'PASS'.
            rankings: Sorted list of candidate moves with alignment and confidence scores.
        """
        if not legal_coords:
            return "PASS", []

        principle_idx = PRINCIPLE_CATALOG.get(principle_code, 0)

        position_tensor = np.expand_dims(board_to_spatial_planes(current_board, active_player), axis=0)
        policy_logits = self.intuitive_mind.forward(position_tensor)[0] # Shape: (362,)

        candidate_planes = []
        numeric_vectors = []
        principle_ids = []

        context_vector = extract_yin_yang_context(tactical_context)

        for (r, c) in legal_coords:
            spatial_plane = board_to_spatial_planes(current_board, active_player, candidate_move=(r, c))
            candidate_planes.append(spatial_plane)
            numeric_vectors.append(context_vector)
            principle_ids.append(principle_idx)

        candidate_batch = np.array(candidate_planes) # Shape: (M, 7, 19, 19)
        numeric_batch = np.array(numeric_vectors)     # Shape: (M, 31)
        pids_batch = np.array(principle_ids)         # Shape: (M,)

        alignments, confidences, values = self.principle_mind.forward(candidate_batch, numeric_batch, pids_batch)

        candidate_scores = []
        for idx, (r, c) in enumerate(legal_coords):
            move_idx = r * 19 + c
            raw_policy_prior = policy_logits[move_idx]
            alignment_score = alignments[idx, principle_idx]
            confidence_score = confidences[idx, principle_idx]

            classical_harmony_score = (alignment_score * confidence_score) + (0.1 * raw_policy_prior)

            candidate_scores.append({
                "coordinate": (r, c),
                "harmony_score": classical_harmony_score,
                "principle_alignment": alignment_score,
                "confidence": confidence_score,
                "policy_prior": raw_policy_prior
            })

        candidate_scores.sort(key=lambda item: item["harmony_score"], reverse=True)
        best_candidate = candidate_scores[0]["coordinate"]

        return best_candidate, candidate_scores


if __name__ == "__main__":
    oracle = AncientGoOracle(weights_path="model_weights.npz")
    empty_board = np.zeros((19, 19), dtype=np.int8)
    sample_legal_moves = [(3, 3), (3, 15), (15, 3), (15, 15), (9, 9)]
    sample_evidence = {"friendly_neighbour_count": 0, "enemy_neighbour_count": 0, "near_empty": True}

    chosen_move, rankings = oracle.seek_best_move(
        current_board=empty_board,
        active_player='B',
        legal_coords=sample_legal_moves,
        tactical_context=sample_evidence,
        principle_code="P-T05-007"
    )

    print(f"Selected Move: {chosen_move}")
    print(f"Top Candidate Harmony Score: {rankings[0]['harmony_score']:.4f}")