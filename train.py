import os
import json
import csv
import numpy as np
from network import (
    PolicyNet, CandidateNet, NumPyAdam,
    board_to_spatial_planes, masked_policy_loss, sparse_principle_loss
)

# =====================================================================
# 1. STATIC PRINCIPLE REGISTRY
# =====================================================================

def build_principle_registry():
    """Maps principle IDs to fixed indices 0..280."""
    registry = {}
    idx = 0
    # Common standard prefix mappings for 281 principles
    for t in range(1, 10):
        for p in range(1, 35):
            pid = f"P-T{t:02d}-{p:03d}"
            registry[pid] = idx
            idx += 1
            if idx >= 281:
                break
        if idx >= 281:
            break
    return registry

PRINCIPLE_TO_IDX = build_principle_registry()


# =====================================================================
# 2. SGF & FEATURE PARSING UTILITIES
# =====================================================================

def parse_board_string(pos_str):
    """Converts 361-char string ('.', 'X', 'O') to 19x19 numpy array (0, 1, -1)."""
    board = np.zeros((19, 19), dtype=np.int8)
    if not pos_str or len(pos_str) < 361:
        return board
    
    for idx, char in enumerate(pos_str[:361]):
        r, c = divmod(idx, 19)
        if char == 'X':
            board[r, c] = 1   # Black
        elif char == 'O':
            board[r, c] = -1  # White
    return board


def sgf_coord_to_rc(coord_sgf):
    """Converts SGF 2-letter coordinate (e.g. 'dp') to (row, col) indices."""
    if not coord_sgf or coord_sgf == "PASS" or len(coord_sgf) < 2:
        return None
    c = ord(coord_sgf[0]) - ord('a')
    r = ord(coord_sgf[1]) - ord('a')
    if 0 <= r < 19 and 0 <= c < 19:
        return (r, c)
    return None


def parse_board_evidence(evidence_str):
    """Extracts 31 numerical features from the board_evidence JSON string."""
    features = np.zeros(31, dtype=np.float32)
    if not evidence_str:
        return features

    try:
        data = json.loads(evidence_str)
        features[0] = float(data.get("friendly_neighbour_count", 0))
        features[1] = float(data.get("enemy_neighbour_count", 0))
        features[2] = float(data.get("nearby_friendly_count", 0))
        features[3] = float(data.get("nearby_enemy_count", 0))
        features[4] = float(data.get("strong_enemy_group_count", 0))
        features[5] = float(data.get("enemy_atari_group_count", 0))
        features[6] = float(data.get("friendly_atari_group_count", 0))
        features[7] = 1.0 if data.get("near_empty", False) else 0.0
        features[8] = float(data.get("captured_count", 0))

        own_group = data.get("own_group_after", {})
        if isinstance(own_group, dict):
            features[9] = float(own_group.get("size", 0))
            features[10] = float(own_group.get("liberties", 0))

        features[11] = 1.0 if data.get("ko_candidate", False) else 0.0
        features[12] = float(data.get("move_number", 0)) / 200.0
        features[13] = float(data.get("total_moves", 0)) / 200.0

        sb = data.get("stone_counts_before", {})
        if isinstance(sb, dict):
            features[14] = float(sb.get("black", 0)) / 180.0
            features[15] = float(sb.get("white", 0)) / 180.0

        sa = data.get("stone_counts_after", {})
        if isinstance(sa, dict):
            features[16] = float(sa.get("black", 0)) / 180.0
            features[17] = float(sa.get("white", 0)) / 180.0
            
    except Exception:
        pass

    return features


# =====================================================================
# 3. DATASET LOADERS
# =====================================================================

def load_policy_dataset(moves_csv_path):
    """Loads moves_all.csv into training arrays for PolicyNet."""
    spatial_list, target_list, mask_list = [], [], []

    if not os.path.exists(moves_csv_path):
        return np.array(spatial_list), np.array(target_list), np.array(mask_list)

    with open(moves_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            board = parse_board_string(row.get("position_before", ""))
            color = row.get("color", "B")
            planes = board_to_spatial_planes(board, color)
            
            coord = sgf_coord_to_rc(row.get("coordinate_sgf", ""))
            if coord is not None:
                target = coord[0] * 19 + coord[1]
            else:
                target = 361 # PASS move index

            legal_mask = np.ones(362, dtype=bool)

            spatial_list.append(planes)
            target_list.append(target)
            mask_list.append(legal_mask)

    return np.array(spatial_list), np.array(target_list), np.array(mask_list)


def load_candidate_dataset(csv_paths):
    """
    Loads principles_followed, principles_violated, and principles_review_queue.
    """
    spatial_list, num_list, pid_list = [], [], []
    align_target_list, conf_target_list, principle_mask_list = [], [], []

    for path, default_target in csv_paths:
        if not os.path.exists(path):
            continue

        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid_str = row.get("principle", "")
                if pid_str not in PRINCIPLE_TO_IDX:
                    continue
                pid_idx = PRINCIPLE_TO_IDX[pid_str]

                board = parse_board_string(row.get("position_before", ""))
                color = row.get("color", "B")
                coord = sgf_coord_to_rc(row.get("coordinate_sgf", ""))

                spatial = board_to_spatial_planes(board, color, candidate_move=coord)
                numeric = parse_board_evidence(row.get("board_evidence", ""))

                conf_val = float(row.get("confidence", 0.5))
                
                if default_target is not None:
                    align_val = default_target
                else:
                    align_val = 1.0 if row.get("annotation_status") == "followed" else 0.0

                align_target = np.zeros(281, dtype=np.float32)
                conf_target = np.zeros(281, dtype=np.float32)
                p_mask = np.zeros(281, dtype=np.float32)

                align_target[pid_idx] = align_val
                conf_target[pid_idx] = conf_val
                p_mask[pid_idx] = 1.0

                spatial_list.append(spatial)
                num_list.append(numeric)
                pid_list.append(pid_idx)
                align_target_list.append(align_target)
                conf_target_list.append(conf_target)
                principle_mask_list.append(p_mask)

    return (
        np.array(spatial_list), np.array(num_list), np.array(pid_list),
        np.array(align_target_list), np.array(conf_target_list), np.array(principle_mask_list)
    )


# =====================================================================
# 4. MAIN TRAINING EXECUTION
# =====================================================================

def train():
    BATCH_SIZE = 30
    EPOCHS = 5

    print("--- Loading Datasets ---")
    pol_spatial, pol_targets, pol_masks = load_policy_dataset("CSV/moves_all.csv")
    
    cand_sources = [
        ("CSV/principles_followed.csv", 1.0),
        ("CSV/principles_violated.csv", 0.0),
        ("CSV/principles_review_queue.csv", None)
    ]
    cand_spatial, cand_num, cand_pids, cand_align, cand_conf, cand_pmasks = load_candidate_dataset(cand_sources)

    print(f"Loaded {len(pol_spatial)} policy samples.")
    print(f"Loaded {len(cand_spatial)} candidate evaluation samples.")

    policy_net = PolicyNet(in_channels=6, num_res_blocks=6)
    candidate_net = CandidateNet(in_channels=7, num_res_blocks=6, num_principles=281)

    # Collect parameters and gradients for unified Adam optimizer
    p_params, p_grads = policy_net.get_params_and_grads()
    c_params, c_grads = candidate_net.get_params_and_grads()

    all_params = p_params + c_params
    all_grads = p_grads + c_grads

    optimizer = NumPyAdam(all_params, all_grads, lr=1e-3, max_norm=5.0)

    num_pol_batches = len(pol_spatial) // BATCH_SIZE if len(pol_spatial) > 0 else 0
    num_cand_batches = len(cand_spatial) // BATCH_SIZE if len(cand_spatial) > 0 else 0
    steps_per_epoch = max(num_pol_batches, num_cand_batches, 1)

    print("\n--- Starting Combined Training Loop ---")
    for epoch in range(1, EPOCHS + 1):
        epoch_pol_loss = 0.0
        epoch_cand_loss = 0.0

        for step in range(steps_per_epoch):
            # Zero out gradients
            for g in all_grads:
                g.fill(0.0)

            # --- 1. Train PolicyNet Step ---
            if num_pol_batches > 0:
                p_idx = (step % num_pol_batches) * BATCH_SIZE
                b_pol_x = pol_spatial[p_idx:p_idx + BATCH_SIZE]
                b_pol_y = pol_targets[p_idx:p_idx + BATCH_SIZE]
                b_pol_m = pol_masks[p_idx:p_idx + BATCH_SIZE]

                pol_logits = policy_net.forward(b_pol_x)
                p_loss, d_pol_logits = masked_policy_loss(pol_logits, b_pol_y, b_pol_m)
                epoch_pol_loss += p_loss

                # Manual backward flow for policy head
                d_spatial_logits = d_pol_logits[:, :361].reshape(-1, 1, 19, 19)
                d_pass_logits = d_pol_logits[:, 361:]
                
                dx_pass = policy_net.pass_dense.backward(d_pass_logits)
                dx_spatial = policy_net.policy_conv.backward(d_spatial_logits)
                
                dx_backbone = dx_spatial + dx_pass.reshape(-1, 64, 19, 19)
                for block in reversed(policy_net.res_blocks):
                    dx_backbone = block.backward(dx_backbone)
                policy_net.stem.backward(policy_net.stem_relu.backward(dx_backbone))

            # --- 2. Train CandidateNet Step ---
            if num_cand_batches > 0:
                c_idx = (step % num_cand_batches) * BATCH_SIZE
                b_cand_x = cand_spatial[c_idx:c_idx + BATCH_SIZE]
                b_cand_num = cand_num[c_idx:c_idx + BATCH_SIZE]
                b_cand_pids = cand_pids[c_idx:c_idx + BATCH_SIZE]
                b_cand_align = cand_align[c_idx:c_idx + BATCH_SIZE]
                b_cand_conf = cand_conf[c_idx:c_idx + BATCH_SIZE]
                b_cand_pmasks = cand_pmasks[c_idx:c_idx + BATCH_SIZE]

                pred_align, pred_conf, pred_val = candidate_net.forward(b_cand_x, b_cand_num, b_cand_pids)
                
                align_loss, d_align = sparse_principle_loss(pred_align, b_cand_align, b_cand_pmasks)
                conf_loss, d_conf = sparse_principle_loss(pred_conf, b_cand_conf, b_cand_pmasks)
                c_loss = align_loss + conf_loss
                epoch_cand_loss += c_loss

                # Direct backprop through heads
                dfused_align = candidate_net.alignment_head.backward(d_align)
                dfused_conf = candidate_net.confidence_head.backward(d_conf)
                dfused = dfused_align + dfused_conf

                # Split fused gradient (64 spatial, 64 numeric, 32 embedding)
                d_spatial_vec = dfused[:, :64]
                d_num_vec = dfused[:, 64:128]
                d_emb_vec = dfused[:, 128:]

                candidate_net.embedding.backward(d_emb_vec)
                
                dx_num = candidate_net.num_dense2.backward(candidate_net.num_relu2.backward(d_num_vec))
                candidate_net.num_dense1.backward(candidate_net.num_relu1.backward(dx_num))

                dx_gap = candidate_net.gap.backward(d_spatial_vec)
                for block in reversed(candidate_net.res_blocks):
                    dx_gap = block.backward(dx_gap)
                candidate_net.stem.backward(candidate_net.stem_relu.backward(dx_gap))

            # --- 3. Update Parameters ---
            optimizer.step()

        avg_pol = epoch_pol_loss / max(num_pol_batches, 1)
        avg_cand = epoch_cand_loss / max(num_cand_batches, 1)
        print(f"Epoch {epoch}/{EPOCHS} - Policy Loss: {avg_pol:.4f} - Candidate Loss: {avg_cand:.4f}")

    # =====================================================================
    # 5. HIGH-PRECISION MODEL CHECKPOINTING
    # =====================================================================
    print("\n--- Saving Binary Checkpoint (model_weights.npz) ---")
    weights_dict = {}
    
    for idx, layer in enumerate(policy_net.layers):
        for p_idx, param in enumerate(layer.params):
            weights_dict[f"policy_layer_{idx}_p_{p_idx}"] = param.astype(np.float32)

    for idx, layer in enumerate(candidate_net.layers):
        for p_idx, param in enumerate(layer.params):
            weights_dict[f"candidate_layer_{idx}_p_{p_idx}"] = param.astype(np.float32)

    np.savez_compressed("model_weights.npz", **weights_dict)
    print("Successfully saved compressed weights to model_weights.npz")


if __name__ == "__main__":
    train()