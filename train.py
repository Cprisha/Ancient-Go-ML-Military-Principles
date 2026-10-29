import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sgfmill import boards, sgf
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


BOARD_SIZE = 19
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
STAGE_NAMES = ["opening", "middle_game", "endgame", "unknown"]
STAGE_TO_INDEX = {name: index for index, name in enumerate(STAGE_NAMES)}
YIN_YANG_DIMENSIONS = [
    "territory_influence",
    "stability_activity",
    "defense_attack",
    "emptiness_fullness",
    "sente_gote",
    "local_global",
]
YIN_YANG_DIM = len(YIN_YANG_DIMENSIONS)
BASE_NUMERIC_FEATURE_NAMES = [
    "normalized_row",
    "normalized_column",
    "pass_flag",
    "move_fraction",
    "capture_feature",
    "post_move_capture_feature",
    "friendly_neighbors",
    "enemy_neighbors",
    "own_liberties",
    "own_group_size",
    "opponent_atari_count",
    "legal_flag",
    "stage_feature",
]
NUMERIC_FEATURE_NAMES = (
    BASE_NUMERIC_FEATURE_NAMES
    + [f"yin_yang_before_{name}" for name in YIN_YANG_DIMENSIONS]
    + [f"yin_yang_after_{name}" for name in YIN_YANG_DIMENSIONS]
    + [f"yin_yang_delta_{name}" for name in YIN_YANG_DIMENSIONS]
)
LABEL_COLUMNS = {
    "sgf_file",
    "game_id",
    "move_number",
    "color",
    "coordinate_sgf",
    "principle",
    "principle_name",
    "confidence",
    "annotation_status",
}
RELATION_KEY_COLUMNS = ["sgf_file", "game_id", "move_number", "color", "coordinate_sgf", "principle"]
PAIRWISE_COLUMNS = {
    "sgf_file",
    "game_id",
    "move_number",
    "color",
    "principle",
    "candidate_a",
    "candidate_b",
    "preference",
    "confidence",
    "annotation_status",
}
PREFERENCE_VALUES = {"a": 1.0, "b": -1.0, "tie": 0.0, "1": 1.0, "-1": -1.0, "0": 0.0}
YIN_YANG_BEFORE_INDICES = list(range(13, 13 + YIN_YANG_DIM))
YIN_YANG_AFTER_INDICES = list(range(13 + YIN_YANG_DIM, 13 + 2 * YIN_YANG_DIM))
YIN_YANG_DELTA_INDICES = list(range(13 + 2 * YIN_YANG_DIM, 13 + 3 * YIN_YANG_DIM))
FEATURE_SPECS = {
    "full": {"spatial_indices": list(range(7)), "numeric_indices": list(range(len(NUMERIC_FEATURE_NAMES)))},
    "before_only": {
        "spatial_indices": [0, 1, 2, 3],
        "numeric_indices": [0, 1, 2, 3, 11, 12] + YIN_YANG_BEFORE_INDICES,
    },
    "board_candidate": {
        "spatial_indices": [0, 1, 2, 3],
        "numeric_indices": [0, 1, 2] + YIN_YANG_BEFORE_INDICES,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def property_value(root, name):
    try:
        value = root.get(name)
    except KeyError:
        value = None
    if value is None:
        return "unknown"
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"
    return str(value)


def board_value(board, row, column):
    return board.get(row, column)


def board_string(board):
    symbols = {"b": "X", "w": "O", None: "."}
    return "".join(symbols[board_value(board, row, column)] for row in range(board.side) for column in range(board.side))


def board_hash(board):
    return hashlib.sha256(board_string(board).encode("ascii")).hexdigest()


def coordinate_sgf(point):
    if point is None:
        return "pass"
    row, column = point
    return chr(ord("a") + column) + chr(ord("a") + row)


def coordinate_human(point, size):
    if point is None:
        return "pass"
    row, column = point
    letters = "".join(chr(ord("A") + index) for index in range(size + 1) if chr(ord("A") + index) != "I")
    return letters[column] + str(size - row)


def stage_for(move_number, total_moves):
    if move_number <= min(30, max(1, int(np.ceil(total_moves * 0.20)))):
        return "opening"
    if move_number <= max(60, int(np.ceil(total_moves * 0.80))):
        return "middle_game"
    return "endgame"


def all_points(size=BOARD_SIZE):
    return [(row, column) for row in range(size) for column in range(size)]


def neighboring_points(row, column, size=BOARD_SIZE):
    for row_delta, column_delta in DIRECTIONS:
        next_row = row + row_delta
        next_column = column + column_delta
        if 0 <= next_row < size and 0 <= next_column < size:
            yield next_row, next_column


def group_and_liberties(board, start_row, start_column):
    color = board[start_row, start_column]
    if color not in {"X", "O"}:
        return set(), set()
    group = set()
    liberties = set()
    stack = [(start_row, start_column)]
    while stack:
        row, column = stack.pop()
        point = (row, column)
        if point in group:
            continue
        group.add(point)
        for next_row, next_column in neighboring_points(row, column):
            value = board[next_row, next_column]
            if value == ".":
                liberties.add((next_row, next_column))
            elif value == color and (next_row, next_column) not in group:
                stack.append((next_row, next_column))
    return group, liberties


def normalized_difference(first, second, scale):
    denominator = max(float(scale), 1.0)
    return float(np.clip((float(first) - float(second)) / denominator, -1.0, 1.0))


def neighbor_count(mask):
    count = np.zeros(mask.shape, dtype=np.float32)
    count[1:, :] += mask[:-1, :]
    count[:-1, :] += mask[1:, :]
    count[:, 1:] += mask[:, :-1]
    count[:, :-1] += mask[:, 1:]
    return count


def color_groups(board, color):
    seen = set()
    groups = []
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            if board[row, column] != color or (row, column) in seen:
                continue
            stack = [(row, column)]
            group = set()
            liberties = set()
            while stack:
                point = stack.pop()
                if point in seen:
                    continue
                seen.add(point)
                group.add(point)
                for next_row, next_column in neighboring_points(point[0], point[1]):
                    value = board[next_row, next_column]
                    if value == ".":
                        liberties.add((next_row, next_column))
                    elif value == color and (next_row, next_column) not in seen:
                        stack.append((next_row, next_column))
            groups.append({"stones": group, "liberties": liberties})
    return groups


def color_state_features(board, color):
    opponent = "O" if color == "X" else "X"
    color_mask = board == color
    opponent_mask = board == opponent
    empty_mask = board == "."
    color_neighbors = neighbor_count(color_mask)
    opponent_neighbors = neighbor_count(opponent_mask)
    territory_mask = empty_mask & (color_neighbors > 0) & (opponent_neighbors == 0)
    frontier_mask = empty_mask & (color_neighbors > 0)
    groups = color_groups(board, color)
    opponent_groups = color_groups(board, opponent)
    liberties_total = sum(len(group["liberties"]) for group in groups)
    safe_groups = sum(len(group["liberties"]) >= 2 for group in groups)
    atari_groups = sum(len(group["liberties"]) == 1 for group in groups)
    opponent_atari_groups = sum(len(group["liberties"]) == 1 for group in opponent_groups)
    positions = np.argwhere(color_mask)
    if len(positions):
        row_min, column_min = positions.min(axis=0)
        row_max, column_max = positions.max(axis=0)
        bounding_box_area = (row_max - row_min + 1) * (column_max - column_min + 1)
    else:
        bounding_box_area = 0
    return {
        "stones": int(color_mask.sum()),
        "territory": int(territory_mask.sum()),
        "frontier": int(frontier_mask.sum()),
        "liberties": int(liberties_total),
        "safe_groups": int(safe_groups),
        "atari_groups": int(atari_groups),
        "opponent_atari_groups": int(opponent_atari_groups),
        "max_group_size": max((len(group["stones"]) for group in groups), default=0),
        "bounding_box_area": int(bounding_box_area),
        "group_count": len(groups),
    }


def yin_yang_state_vector(board, player):
    opponent = "O" if player == "X" else "X"
    own = color_state_features(board, player)
    enemy = color_state_features(board, opponent)
    occupied = own["stones"] + enemy["stones"]
    empty = BOARD_AREA - occupied
    territory_balance = normalized_difference(own["territory"], enemy["territory"], BOARD_AREA)
    influence_balance = normalized_difference(own["frontier"], enemy["frontier"], BOARD_AREA)
    stability_balance = normalized_difference(own["liberties"], enemy["liberties"], BOARD_AREA)
    activity_balance = normalized_difference(own["frontier"], enemy["frontier"], BOARD_AREA)
    defense_balance = normalized_difference(own["safe_groups"], enemy["safe_groups"], max(own["group_count"] + enemy["group_count"], 1))
    attack_balance = normalized_difference(enemy["atari_groups"], own["atari_groups"], max(own["group_count"] + enemy["group_count"], 1))
    sente_balance = normalized_difference(enemy["atari_groups"], own["atari_groups"], 4)
    locality_balance = normalized_difference(own["max_group_size"], enemy["max_group_size"], BOARD_AREA)
    spread_balance = normalized_difference(own["bounding_box_area"], enemy["bounding_box_area"], BOARD_AREA)
    vector = np.asarray(
        [
            np.clip(territory_balance - influence_balance, -1.0, 1.0),
            np.clip(stability_balance - activity_balance, -1.0, 1.0),
            np.clip(defense_balance - attack_balance, -1.0, 1.0),
            np.clip((empty - occupied) / BOARD_AREA, -1.0, 1.0),
            np.clip(sente_balance, -1.0, 1.0),
            np.clip(locality_balance - spread_balance, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return vector


def array_from_position(position):
    if not isinstance(position, str) or len(position) != BOARD_AREA:
        raise ValueError("A board position must contain exactly 361 characters")
    board = np.asarray(list(position), dtype="U1").reshape(BOARD_SIZE, BOARD_SIZE)
    if not np.isin(board, [".", "X", "O"]).all():
        raise ValueError("A board position contains unsupported characters")
    return board


def parse_sgf_file(path):
    raw = path.read_bytes()
    game = sgf.Sgf_game.from_bytes(raw)
    size = game.get_size()
    if size != BOARD_SIZE:
        raise ValueError(f"Expected a 19x19 board but found {size}x{size}")
    sequence = game.get_main_sequence()
    root = sequence[0]
    total_moves = sum(1 for node in sequence if node.get_move()[0] is not None)
    board = boards.Board(size)
    records = []
    move_number = 0

    for node in sequence:
        setup_black, setup_white, setup_empty = node.get_setup_stones()
        if setup_black or setup_white or setup_empty:
            board.apply_setup(setup_black, setup_white, setup_empty)
        color, point = node.get_move()
        if color is None:
            continue

        move_number += 1
        before = board.copy()
        before_position = board_string(before)
        before_hash = board_hash(before)
        legal_status = "pass" if point is None else "legal"
        captured_points = []

        try:
            if point is None:
                after = before.copy()
            else:
                board.play(point[0], point[1], color)
                after = board.copy()
            opponent = "w" if color == "b" else "b"
            before_opponent = {
                point_value for point_value in all_points(size) if board_value(before, point_value[0], point_value[1]) == opponent
            }
            after_opponent = {
                point_value for point_value in all_points(size) if board_value(after, point_value[0], point_value[1]) == opponent
            }
            captured_points = sorted(before_opponent - after_opponent)
        except Exception:
            legal_status = "illegal"
            after = before.copy()
            board = before.copy()

        records.append(
            {
                "sgf_file": path.name,
                "game_id": path.stem,
                "move_number": move_number,
                "color": "B" if color == "b" else "W",
                "coordinate_sgf": coordinate_sgf(point),
                "coordinate_human": coordinate_human(point, size),
                "legal_status": legal_status,
                "captured_count": len(captured_points),
                "captured_coordinates": json.dumps([coordinate_sgf(value) for value in captured_points]),
                "board_size": size,
                "rules": property_value(root, "RU"),
                "black_player": property_value(root, "PB"),
                "white_player": property_value(root, "PW"),
                "result": property_value(root, "RE"),
                "stage": stage_for(move_number, total_moves),
                "position_before": before_position,
                "position_before_hash": before_hash,
                "position_after": board_string(after),
                "position_after_hash": board_hash(after),
            }
        )

    return records


def parse_sgf_collection(sgf_dir, allow_parse_errors=False):
    sgf_dir = Path(sgf_dir)
    paths = sorted(sgf_dir.glob("*.sgf"))
    if not paths:
        raise FileNotFoundError(f"No SGF files were found in {sgf_dir}")

    records = []
    errors = []
    for path in paths:
        try:
            records.extend(parse_sgf_file(path))
        except Exception as error:
            errors.append({"sgf_file": path.name, "error": str(error)})

    if errors and not allow_parse_errors:
        examples = "; ".join(f"{item['sgf_file']}: {item['error']}" for item in errors[:3])
        raise RuntimeError(f"SGF parsing failed for {len(errors)} files. {examples}")

    parsed = pd.DataFrame(records)
    if parsed.empty:
        raise ValueError("SGF parsing produced no move records")
    print(f"sgf_files={len(paths)} parsed_games={parsed['game_id'].nunique()} parsed_moves={len(parsed)} parse_errors={len(errors)}")
    if errors:
        print(json.dumps({"parse_errors": errors}, indent=2))
    return parsed, errors


def canonical_game_id(value):
    text = Path(str(value)).stem.strip().lower()
    match = re.search(r"(\d+)$", text)
    if match:
        return str(int(match.group(1)))
    return text


def load_relation_labels(followed_csv, violated_csv):
    followed = pd.read_csv(followed_csv)
    violated = pd.read_csv(violated_csv)
    missing_followed = LABEL_COLUMNS.difference(followed.columns)
    missing_violated = LABEL_COLUMNS.difference(violated.columns)
    if missing_followed:
        raise ValueError(f"Missing followed label columns: {sorted(missing_followed)}")
    if missing_violated:
        raise ValueError(f"Missing violated label columns: {sorted(missing_violated)}")

    followed = followed.copy()
    violated = violated.copy()
    followed["relation_label"] = "followed"
    followed["target_alignment"] = 1.0
    violated["relation_label"] = "violated"
    violated["target_alignment"] = -1.0
    data = pd.concat([followed, violated], ignore_index=True)
    data["sgf_file"] = data["sgf_file"].map(lambda value: Path(str(value)).name)
    data["game_id"] = data["game_id"].map(canonical_game_id)
    data["move_number"] = pd.to_numeric(data["move_number"], errors="coerce").fillna(0).astype(int)
    data["color"] = data["color"].astype(str).str.upper()
    data["coordinate_sgf"] = data["coordinate_sgf"].fillna("pass").astype(str).str.lower()
    data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce").fillna(0.5).clip(0.05, 1.0)
    data["annotation_status"] = data["annotation_status"].fillna("unknown").astype(str)
    data["label_mask"] = data["annotation_status"].isin({"auto_candidate", "reviewed"}).astype(np.float32)
    data["label_weight"] = data["confidence"].astype(np.float32)
    data["confidence_target"] = data["confidence"].astype(np.float32)
    data = data.drop_duplicates(subset=RELATION_KEY_COLUMNS + ["relation_label"]).reset_index(drop=True)
    return data


def load_candidate_comparisons(comparisons_csv):
    data = pd.read_csv(comparisons_csv)
    missing = PAIRWISE_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing candidate comparison columns: {sorted(missing)}")
    data = data.copy()
    data["sgf_file"] = data["sgf_file"].map(lambda value: Path(str(value)).name)
    data["game_id"] = data["game_id"].map(canonical_game_id)
    data["move_number"] = pd.to_numeric(data["move_number"], errors="coerce").fillna(0).astype(int)
    data["color"] = data["color"].astype(str).str.upper()
    data["principle"] = data["principle"].astype(str)
    data["candidate_a"] = data["candidate_a"].fillna("pass").astype(str).str.lower()
    data["candidate_b"] = data["candidate_b"].fillna("pass").astype(str).str.lower()
    data["preference"] = data["preference"].astype(str).str.lower().map(PREFERENCE_VALUES)
    if data["preference"].isna().any():
        raise ValueError("Candidate comparison preference must be a, b, tie, 1, -1, or 0")
    data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce").fillna(0.5).clip(0.05, 1.0)
    data["annotation_status"] = data["annotation_status"].fillna("unknown").astype(str)
    data["label_mask"] = data["annotation_status"].isin({"auto_candidate", "reviewed"}).astype(np.float32)
    data["label_weight"] = data["confidence"].astype(np.float32)
    data["comparison_group"] = (
        data["sgf_file"].astype(str)
        + ":"
        + data["game_id"].astype(str)
        + ":"
        + data["move_number"].astype(str)
        + ":"
        + data["color"].astype(str)
        + ":"
        + data["principle"].astype(str)
    )
    data = data.drop_duplicates(subset=["comparison_group", "candidate_a", "candidate_b"]).reset_index(drop=True)
    return data


def join_labels_to_parsed_moves(labels, parsed_moves):
    labels = labels.copy()
    labels["sgf_file"] = labels["sgf_file"].map(lambda value: Path(str(value)).name)
    labels["game_id"] = labels["game_id"].map(canonical_game_id)
    labels["color"] = labels["color"].astype(str).str.upper()
    labels["coordinate_sgf"] = labels["coordinate_sgf"].fillna("pass").astype(str).str.lower()
    parsed_moves = parsed_moves.copy()
    parsed_moves["sgf_file"] = parsed_moves["sgf_file"].map(lambda value: Path(str(value)).name)
    parsed_moves["game_id"] = parsed_moves["game_id"].map(canonical_game_id)
    parsed_moves["color"] = parsed_moves["color"].astype(str).str.upper()
    parsed_moves["coordinate_sgf"] = parsed_moves["coordinate_sgf"].astype(str).str.lower()
    move_key = ["sgf_file", "game_id", "move_number", "color", "coordinate_sgf"]
    parsed_moves = parsed_moves.drop_duplicates(subset=move_key)
    joined = labels.merge(parsed_moves, on=move_key, how="left", validate="many_to_one", suffixes=("_label", ""))
    missing = joined["position_before"].isna()
    if missing.any():
        examples = joined.loc[missing, move_key].head(5).to_dict(orient="records")
        raise ValueError(f"{int(missing.sum())} label rows did not match parsed SGF moves. Examples: {examples}")
    joined["stage"] = joined["stage"].fillna("unknown").astype(str).str.lower()
    joined["stage"] = joined["stage"].where(joined["stage"].isin(STAGE_TO_INDEX), "unknown")
    return joined.reset_index(drop=True)


def split_by_position(data, seed, validation_fraction, test_fraction):
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")

    indices = np.arange(len(data))
    groups = data["position_before_hash"].astype(str).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=validation_fraction + test_fraction, random_state=seed)
    train_indices, heldout_indices = next(first.split(indices, groups=groups))
    heldout = data.iloc[heldout_indices].reset_index(drop=True)
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_test_fraction, random_state=seed + 1)
    validation_relative, test_relative = next(
        second.split(np.arange(len(heldout)), groups=heldout["position_before_hash"].astype(str).to_numpy())
    )
    return {
        "train": data.iloc[train_indices].reset_index(drop=True),
        "validation": heldout.iloc[validation_relative].reset_index(drop=True),
        "test": heldout.iloc[test_relative].reset_index(drop=True),
    }


def build_principle_vocabulary(train_data):
    principles = sorted(train_data["principle"].dropna().astype(str).unique())
    if not principles:
        raise ValueError("No principles are available in the training split")
    return {principle: index for index, principle in enumerate(principles)}


def attach_principle_indices(splits, principle_to_idx):
    result = {}
    for name, frame in splits.items():
        frame = frame.copy()
        frame["principle_idx"] = frame["principle"].map(principle_to_idx)
        unseen = int(frame["principle_idx"].isna().sum())
        frame = frame.dropna(subset=["principle_idx"]).reset_index(drop=True)
        frame["principle_idx"] = frame["principle_idx"].astype(int)
        result[name] = frame
        relation_counts = frame["relation_label"].value_counts().to_dict()
        print(f"{name}: rows={len(frame)} games={frame['game_id'].nunique()} unseen_principle_rows={unseen} relations={relation_counts}")
    return result


def board_array_from_position(position):
    if not isinstance(position, str) or len(position) != BOARD_AREA:
        raise ValueError("The parser must provide a 361-character position")
    board = np.asarray(list(position), dtype="U1").reshape(BOARD_SIZE, BOARD_SIZE)
    if not np.isin(board, [".", "X", "O"]).all():
        raise ValueError("The parsed board contains unsupported characters")
    return board


def sgf_coordinate_to_row_col(coordinate):
    if not isinstance(coordinate, str) or len(coordinate) != 2 or coordinate.lower() == "pass":
        return None
    coordinate = coordinate.lower()
    column = ord(coordinate[0]) - ord("a")
    row = ord(coordinate[1]) - ord("a")
    if 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE:
        return row, column
    return None


def transition_metrics(before, after, color, coordinate_sgf, legal_status, captured_count):
    player = "X" if color == "B" else "O"
    opponent = "O" if player == "X" else "X"
    row_col = sgf_coordinate_to_row_col(coordinate_sgf)
    if row_col is None:
        return {
            "normalized_row": 0.0,
            "normalized_column": 0.0,
            "pass_flag": 1.0,
            "friendly_neighbors": 0.0,
            "enemy_neighbors": 0.0,
            "own_liberties": 0.0,
            "own_group_size": 0.0,
            "opponent_atari_count": 0.0,
            "legal_flag": float(legal_status in {"legal", "pass"}),
            "capture_feature": min(float(captured_count) / 10.0, 1.0),
            "post_move_capture_feature": min(float(captured_count) / 10.0, 1.0),
        }

    row, column = row_col
    friendly_neighbors = sum(before[next_row, next_column] == player for next_row, next_column in neighboring_points(row, column))
    enemy_neighbors = sum(before[next_row, next_column] == opponent for next_row, next_column in neighboring_points(row, column))
    own_group, own_liberties = group_and_liberties(after, row, column)
    adjacent_groups = set()
    opponent_atari_count = 0
    for next_row, next_column in neighboring_points(row, column):
        if after[next_row, next_column] != opponent:
            continue
        group, liberties = group_and_liberties(after, next_row, next_column)
        key = tuple(sorted(group))
        if key not in adjacent_groups:
            adjacent_groups.add(key)
            opponent_atari_count += int(len(liberties) == 1)

    return {
        "normalized_row": row / (BOARD_SIZE - 1),
        "normalized_column": column / (BOARD_SIZE - 1),
        "pass_flag": 0.0,
        "friendly_neighbors": friendly_neighbors / 4.0,
        "enemy_neighbors": enemy_neighbors / 4.0,
        "own_liberties": min(len(own_liberties) / 20.0, 1.0),
        "own_group_size": min(len(own_group) / 20.0, 1.0),
        "opponent_atari_count": min(opponent_atari_count / 4.0, 1.0),
        "legal_flag": float(legal_status in {"legal", "pass"}),
        "capture_feature": min(float(captured_count) / 10.0, 1.0),
        "post_move_capture_feature": min(float(captured_count) / 10.0, 1.0),
    }


def encode_candidate_row(row, feature_set, cache):
    spatial_indices = FEATURE_SPECS[feature_set]["spatial_indices"]
    numeric_indices = FEATURE_SPECS[feature_set]["numeric_indices"]
    cache_key = (row["position_before_hash"], row["coordinate_sgf"], row["position_after_hash"], feature_set)
    if cache_key in cache:
        spatial, numeric, yin_yang_delta = cache[cache_key]
        return spatial, numeric, yin_yang_delta

    before = board_array_from_position(row["position_before"])
    after = board_array_from_position(row["position_after"])
    player = "X" if row["color"] == "B" else "O"
    enemy = "O" if player == "X" else "X"
    candidate_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    row_col = sgf_coordinate_to_row_col(row["coordinate_sgf"])
    if row_col is not None:
        candidate_plane[row_col] = 1.0
    capture_plane = ((before == enemy) & (after == ".")).astype(np.float32)
    spatial = torch.from_numpy(
        np.stack(
            [
                (before == player).astype(np.float32),
                (before == enemy).astype(np.float32),
                np.full((BOARD_SIZE, BOARD_SIZE), 1.0 if row["color"] == "B" else 0.0, dtype=np.float32),
                candidate_plane,
                (after == player).astype(np.float32),
                (after == enemy).astype(np.float32),
                capture_plane,
            ],
            axis=0,
        )
    )
    metrics = transition_metrics(
        before,
        after,
        row["color"],
        row["coordinate_sgf"],
        row["legal_status"],
        row["captured_count"],
    )
    yin_yang_before = yin_yang_state_vector(before, player)
    yin_yang_after = yin_yang_state_vector(after, player)
    yin_yang_delta = yin_yang_after - yin_yang_before
    move_fraction = min(float(row["move_number"]) / 400.0, 1.0)
    stage_index = STAGE_TO_INDEX.get(str(row["stage"]), STAGE_TO_INDEX["unknown"])
    stage_feature = stage_index / (len(STAGE_NAMES) - 1)
    base_numeric = [
        metrics["normalized_row"],
        metrics["normalized_column"],
        metrics["pass_flag"],
        move_fraction,
        metrics["capture_feature"],
        metrics["post_move_capture_feature"],
        metrics["friendly_neighbors"],
        metrics["enemy_neighbors"],
        metrics["own_liberties"],
        metrics["own_group_size"],
        metrics["opponent_atari_count"],
        metrics["legal_flag"],
        stage_feature,
    ]
    numeric = torch.tensor(
        base_numeric + yin_yang_before.tolist() + yin_yang_after.tolist() + yin_yang_delta.tolist(),
        dtype=torch.float32,
    )
    yin_yang_delta_tensor = torch.from_numpy(yin_yang_delta)
    cache[cache_key] = (spatial, numeric, yin_yang_delta_tensor)
    return spatial, numeric, yin_yang_delta_tensor


class GoPrincipleDataset(Dataset):
    def __init__(self, data, feature_set):
        if feature_set not in FEATURE_SPECS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.data = data.reset_index(drop=True)
        self.feature_set = feature_set
        self.spatial_indices = FEATURE_SPECS[feature_set]["spatial_indices"]
        self.numeric_indices = FEATURE_SPECS[feature_set]["numeric_indices"]
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def encode_features(self, row):
        spatial, numeric, yin_yang_delta = encode_candidate_row(row, self.feature_set, self.cache)
        return spatial[self.spatial_indices], numeric[self.numeric_indices], yin_yang_delta

    def __getitem__(self, index):
        row = self.data.iloc[index]
        spatial, numeric, yin_yang_delta = self.encode_features(row)
        return {
            "board": spatial,
            "numeric": numeric,
            "principle": torch.tensor(int(row["principle_idx"]), dtype=torch.long),
            "target": torch.tensor(float(row["target_alignment"]), dtype=torch.float32),
            "confidence_target": torch.tensor(float(row["confidence_target"]), dtype=torch.float32),
            "yin_yang_target": yin_yang_delta,
            "mask": torch.tensor(float(row["label_mask"]), dtype=torch.float32),
            "weight": torch.tensor(float(row["label_weight"]), dtype=torch.float32),
        }


def board_from_position(position):
    board = boards.Board(BOARD_SIZE)
    black_points = []
    white_points = []
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            value = position[row * BOARD_SIZE + column]
            if value == "X":
                black_points.append((row, column))
            elif value == "O":
                white_points.append((row, column))
    board.apply_setup(black_points, white_points, [])
    return board


def candidate_row_from_position(base_row, candidate):
    row = dict(base_row)
    row["coordinate_sgf"] = str(candidate).lower()
    before = board_from_position(base_row["position_before"])
    before_opponent = "w" if base_row["color"] == "B" else "b"
    opponent_points_before = {
        point for point in all_points(BOARD_SIZE) if board_value(before, point[0], point[1]) == before_opponent
    }
    point = sgf_coordinate_to_row_col(row["coordinate_sgf"])
    if point is None:
        after = before.copy()
        row["legal_status"] = "pass"
    else:
        color = "b" if base_row["color"] == "B" else "w"
        try:
            before.play(point[0], point[1], color)
            after = before.copy()
            row["legal_status"] = "legal"
        except Exception:
            after = board_from_position(base_row["position_before"])
            row["legal_status"] = "illegal"
    opponent_points_after = {
        point for point in all_points(BOARD_SIZE) if board_value(after, point[0], point[1]) == before_opponent
    }
    captured_points = sorted(opponent_points_before - opponent_points_after)
    row["position_after"] = board_string(after)
    row["position_after_hash"] = hashlib.sha256(row["position_after"].encode("ascii")).hexdigest()
    row["captured_count"] = len(captured_points)
    row["captured_coordinates"] = json.dumps([coordinate_sgf(value) for value in captured_points])
    return row


def build_pairwise_examples(comparisons, parsed_moves):
    parsed = parsed_moves.copy()
    parsed["sgf_file"] = parsed["sgf_file"].map(lambda value: Path(str(value)).name)
    parsed["game_id"] = parsed["game_id"].map(canonical_game_id)
    parsed["move_number"] = pd.to_numeric(parsed["move_number"], errors="coerce").fillna(0).astype(int)
    parsed["color"] = parsed["color"].astype(str).str.upper()
    parsed["coordinate_sgf"] = parsed["coordinate_sgf"].astype(str).str.lower()
    base_key = ["sgf_file", "game_id", "move_number", "color"]
    parsed_lookup = {}
    for _, row in parsed.iterrows():
        parsed_lookup[tuple(row[column] for column in base_key)] = row.to_dict()
    records = []
    missing = []
    for _, comparison in comparisons.iterrows():
        key = tuple(comparison[column] for column in base_key)
        base_row = parsed_lookup.get(key)
        if base_row is None:
            missing.append({column: comparison[column] for column in base_key})
            continue
        row_a = candidate_row_from_position(base_row, comparison["candidate_a"])
        row_b = candidate_row_from_position(base_row, comparison["candidate_b"])
        if row_a["legal_status"] == "illegal" or row_b["legal_status"] == "illegal":
            continue
        records.append(
            {
                "position_before_hash": base_row["position_before_hash"],
                "position_id": f"{base_row['sgf_file']}:{base_row['move_number']}:{base_row['position_before_hash']}",
                "sgf_file": base_row["sgf_file"],
                "game_id": base_row["game_id"],
                "move_number": base_row["move_number"],
                "color": base_row["color"],
                "principle": comparison["principle"],
                "row_a": row_a,
                "row_b": row_b,
                "preference": float(comparison["preference"]),
                "label_mask": float(comparison["label_mask"]),
                "label_weight": float(comparison["label_weight"]),
            }
        )
    if missing:
        examples = missing[:5]
        raise ValueError(f"{len(missing)} candidate comparison rows did not match parsed positions. Examples: {examples}")
    if not records:
        raise ValueError("No valid candidate comparisons were constructed")
    return pd.DataFrame(records)


def split_pairwise_by_position(data, seed, validation_fraction, test_fraction):
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    indices = np.arange(len(data))
    groups = data["position_before_hash"].astype(str).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=validation_fraction + test_fraction, random_state=seed)
    train_indices, heldout_indices = next(first.split(indices, groups=groups))
    heldout = data.iloc[heldout_indices].reset_index(drop=True)
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_test_fraction, random_state=seed + 1)
    validation_relative, test_relative = next(
        second.split(np.arange(len(heldout)), groups=heldout["position_before_hash"].astype(str).to_numpy())
    )
    return {
        "train": data.iloc[train_indices].reset_index(drop=True),
        "validation": heldout.iloc[validation_relative].reset_index(drop=True),
        "test": heldout.iloc[test_relative].reset_index(drop=True),
    }


def attach_pairwise_principle_indices(splits, principle_to_idx):
    result = {}
    for name, frame in splits.items():
        frame = frame.copy()
        frame["principle_idx"] = frame["principle"].map(principle_to_idx)
        unseen = int(frame["principle_idx"].isna().sum())
        frame = frame.dropna(subset=["principle_idx"]).reset_index(drop=True)
        frame["principle_idx"] = frame["principle_idx"].astype(int)
        result[name] = frame
        print(f"{name}: pair_rows={len(frame)} games={frame['game_id'].nunique()} unseen_principle_rows={unseen}")
    return result


class CandidatePairDataset(Dataset):
    def __init__(self, data, feature_set):
        if feature_set not in FEATURE_SPECS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.data = data.reset_index(drop=True)
        self.feature_set = feature_set
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        spatial_a, numeric_a, _ = encode_candidate_row(row["row_a"], self.feature_set, self.cache)
        spatial_b, numeric_b, _ = encode_candidate_row(row["row_b"], self.feature_set, self.cache)
        spatial_indices = FEATURE_SPECS[self.feature_set]["spatial_indices"]
        numeric_indices = FEATURE_SPECS[self.feature_set]["numeric_indices"]
        return {
            "board_a": spatial_a[spatial_indices],
            "numeric_a": numeric_a[numeric_indices],
            "board_b": spatial_b[spatial_indices],
            "numeric_b": numeric_b[numeric_indices],
            "principle": torch.tensor(int(row["principle_idx"]), dtype=torch.long),
            "preference": torch.tensor(float(row["preference"]), dtype=torch.float32),
            "mask": torch.tensor(float(row["label_mask"]), dtype=torch.float32),
            "weight": torch.tensor(float(row["label_weight"]), dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class PrincipleAlignmentModel(nn.Module):
    def __init__(self, num_principles, in_channels, numeric_dim, embed_dim=32):
        super().__init__()
        self.principle_embed = nn.Embedding(num_principles, embed_dim)
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.norm_in = nn.BatchNorm2d(64)
        self.residual_1 = ResidualBlock(64)
        self.residual_2 = ResidualBlock(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.alignment_head = nn.Sequential(
            nn.Linear(64 + embed_dim + numeric_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(64 + embed_dim + numeric_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, board, numeric, principle):
        features = F.relu(self.norm_in(self.conv_in(board)))
        features = self.residual_1(features)
        features = self.residual_2(features)
        board_features = self.global_pool(features).flatten(1)
        principle_features = self.principle_embed(principle)
        combined = torch.cat([board_features, principle_features, numeric], dim=1)
        return {
            "alignment": self.alignment_head(combined).squeeze(1),
            "confidence": self.confidence_head(combined).squeeze(1),
        }


def pairwise_ranking_loss(score_a, score_b, preference, mask, weight, tie_margin=0.05):
    effective_weight = mask * weight
    difference = score_a - score_b
    non_tie = preference.abs() > 0
    pair_loss = F.softplus(-preference * difference)
    tie_loss = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none")
    raw_loss = torch.where(non_tie, pair_loss, tie_loss)
    return (raw_loss * effective_weight).sum() / effective_weight.sum().clamp_min(1e-8)


def pairwise_metrics(score_a, score_b, preference, mask, weight):
    active = mask > 0
    if not torch.any(active):
        return {"loss": 0.0, "comparison_accuracy": 0.0, "active_examples": 0}
    score_a = score_a[active]
    score_b = score_b[active]
    preference = preference[active]
    weight = weight[active]
    difference = score_a - score_b
    losses = torch.where(
        preference.abs() > 0,
        F.softplus(-preference * difference),
        F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none"),
    )
    predicted = torch.where(difference > 0.05, torch.ones_like(difference), torch.where(difference < -0.05, -torch.ones_like(difference), torch.zeros_like(difference)))
    correct = predicted == preference
    return {
        "loss": float((losses * weight).sum().item() / weight.sum().clamp_min(1e-8).item()),
        "comparison_accuracy": float(correct.float().mean().item()),
        "active_examples": int(active.sum().item()),
    }


def weighted_alignment_loss(prediction, target, mask, weight):
    effective_weight = mask * weight
    raw_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (raw_loss * effective_weight).sum() / effective_weight.sum().clamp_min(1e-8)


def weighted_confidence_loss(prediction, target, mask, weight):
    effective_weight = mask * weight
    raw_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (raw_loss * effective_weight).sum() / effective_weight.sum().clamp_min(1e-8)


def curriculum_loss(outputs, batch, phase, confidence_loss_weight):
    alignment_loss = weighted_alignment_loss(
        outputs["alignment"], batch["target"], batch["mask"], batch["weight"]
    )
    confidence_loss = weighted_confidence_loss(
        outputs["confidence"], batch["confidence_target"], batch["mask"], batch["weight"]
    )
    if phase == "confidence":
        return confidence_loss
    if phase == "alignment":
        return alignment_loss + confidence_loss_weight * confidence_loss
    raise ValueError(f"Unknown curriculum phase: {phase}")


def metrics_from_arrays(prediction, target, mask, weight, principle):
    active = mask > 0
    if not torch.any(active):
        return {"loss": 0.0, "sign_accuracy": 0.0, "active_examples": 0, "per_principle": {}}
    active_prediction = prediction[active]
    active_target = target[active]
    active_weight = weight[active]
    active_principle = principle[active]
    losses = F.smooth_l1_loss(active_prediction, active_target, reduction="none")
    result = {
        "loss": float((losses * active_weight).sum().item() / active_weight.sum().clamp_min(1e-8).item()),
        "sign_accuracy": float(((active_prediction >= 0) == (active_target >= 0)).float().mean().item()),
        "active_examples": int(active.sum().item()),
        "per_principle": {},
    }
    for principle_index in torch.unique(active_principle).tolist():
        mask_for_principle = active_principle == principle_index
        principle_losses = losses[mask_for_principle]
        principle_weights = active_weight[mask_for_principle]
        result["per_principle"][str(int(principle_index))] = {
            "loss": float((principle_losses * principle_weights).sum().item() / principle_weights.sum().clamp_min(1e-8).item()),
            "sign_accuracy": float(
                ((active_prediction[mask_for_principle] >= 0) == (active_target[mask_for_principle] >= 0)).float().mean().item()
            ),
            "active_examples": int(mask_for_principle.sum().item()),
        }
    return result


def confidence_metrics(prediction, target, mask, weight):
    active = mask > 0
    if not torch.any(active):
        return {"loss": 0.0, "mae": 0.0}
    active_prediction = prediction[active]
    active_target = target[active]
    active_weight = weight[active]
    losses = F.smooth_l1_loss(active_prediction, active_target, reduction="none")
    return {
        "loss": float((losses * active_weight).sum().item() / active_weight.sum().clamp_min(1e-8).item()),
        "mae": float((torch.abs(active_prediction - active_target) * active_weight).sum().item() / active_weight.sum().clamp_min(1e-8).item()),
    }


def evaluate(model, loader, device):
    model.eval()
    alignment_predictions = []
    confidence_predictions = []
    targets = []
    confidence_targets = []
    masks = []
    weights = []
    principles = []
    with torch.no_grad():
        for batch in loader:
            board = batch["board"].to(device)
            numeric = batch["numeric"].to(device)
            principle = batch["principle"].to(device)
            outputs = model(board, numeric, principle)
            alignment_predictions.append(outputs["alignment"].cpu())
            confidence_predictions.append(outputs["confidence"].cpu())
            targets.append(batch["target"])
            confidence_targets.append(batch["confidence_target"])
            masks.append(batch["mask"])
            weights.append(batch["weight"])
            principles.append(batch["principle"])
    if not alignment_predictions:
        return {
            "alignment": {"loss": 0.0, "sign_accuracy": 0.0, "active_examples": 0, "per_principle": {}},
            "confidence": {"loss": 0.0, "mae": 0.0},
        }
    alignment_metrics = metrics_from_arrays(
        torch.cat(alignment_predictions),
        torch.cat(targets),
        torch.cat(masks),
        torch.cat(weights),
        torch.cat(principles),
    )
    confidence = confidence_metrics(
        torch.cat(confidence_predictions),
        torch.cat(confidence_targets),
        torch.cat(masks),
        torch.cat(weights),
    )
    return {"alignment": alignment_metrics, "confidence": confidence}


def save_checkpoint(path, model, optimizer, epoch, metrics, principle_to_idx, seed, feature_set, sgf_dir, training_mode="alignment", comparisons_csv=None):
    config = FEATURE_SPECS[feature_set]
    artifact = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "principle_to_idx": principle_to_idx,
        "stage_to_index": STAGE_TO_INDEX,
        "seed": seed,
        "sgf_dir": str(sgf_dir),
        "feature_set": feature_set,
        "model_config": {
            "num_principles": len(principle_to_idx),
            "embed_dim": 32,
            "numeric_dim": len(config["numeric_indices"]),
            "in_channels": len(config["spatial_indices"]),
            "board_size": BOARD_SIZE,
        },
        "split_group": "position_before_hash",
        "input_source": "replayed_sgf_board_positions",
        "training_mode": training_mode,
        "comparisons_csv": str(comparisons_csv) if comparisons_csv is not None else None,
        "yin_yang_dimensions": YIN_YANG_DIMENSIONS,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
    }
    torch.save(artifact, path)


def make_train_loader(dataset, frame, batch_size, balanced_sampling):
    if balanced_sampling:
        relation_counts = frame["relation_label"].value_counts().to_dict()
        sample_weights = frame["relation_label"].map(
            {label: 1.0 / count for label, count in relation_counts.items()}
        ).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def train_alignment_model(
    sgf_dir,
    followed_csv,
    violated_csv,
    save_path,
    feature_set="full",
    epochs=5,
    batch_size=48,
    learning_rate=1e-3,
    validation_fraction=0.13,
    test_fraction=0.14,
    seed=39,
    confidence_epochs=1,
    confidence_loss_weight=0.2,
    balanced_sampling=True,
    allow_parse_errors=False,
):
    if feature_set not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if epochs < 1 or confidence_epochs < 0 or confidence_epochs >= epochs:
        raise ValueError("epochs must be at least 2 and confidence_epochs must be smaller than epochs")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_relation_labels(followed_csv, violated_csv)
    parsed_moves, parse_errors = parse_sgf_collection(sgf_dir, allow_parse_errors=allow_parse_errors)
    data = join_labels_to_parsed_moves(labels, parsed_moves)
    splits = split_by_position(data, seed, validation_fraction, test_fraction)
    principle_to_idx = build_principle_vocabulary(splits["train"])
    splits = attach_principle_indices(splits, principle_to_idx)

    datasets = {name: GoPrincipleDataset(frame, feature_set) for name, frame in splits.items()}
    loaders = {
        "train": make_train_loader(datasets["train"], splits["train"], batch_size, balanced_sampling),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    model = PrincipleAlignmentModel(
        num_principles=len(principle_to_idx),
        in_channels=len(FEATURE_SPECS[feature_set]["spatial_indices"]),
        numeric_dim=len(FEATURE_SPECS[feature_set]["numeric_indices"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_metrics = {}

    print(f"device={device}")
    print(f"feature_set={feature_set}")
    print(f"input_source=replayed_sgf_board_positions")
    print(f"principles={len(principle_to_idx)}")
    print(f"split_group=position_before_hash")
    print(f"balanced_sampling={balanced_sampling}")
    print(f"confidence_epochs={confidence_epochs}")
    print(f"parse_errors={len(parse_errors)}")

    for epoch in range(1, epochs + 1):
        phase = "confidence" if epoch <= confidence_epochs else "alignment"
        model.train()
        total_loss = 0.0
        active_examples = 0
        for batch in loaders["train"]:
            board = batch["board"].to(device)
            numeric = batch["numeric"].to(device)
            principle = batch["principle"].to(device)
            target = batch["target"].to(device)
            confidence_target = batch["confidence_target"].to(device)
            mask = batch["mask"].to(device)
            weight = batch["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(board, numeric, principle)
            loss = curriculum_loss(
                outputs,
                {
                    "target": target,
                    "confidence_target": confidence_target,
                    "mask": mask,
                    "weight": weight,
                },
                phase,
                confidence_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_active = int((mask > 0).sum().item())
            total_loss += loss.item() * max(batch_active, 1)
            active_examples += batch_active

        train_loss = total_loss / max(active_examples, 1)
        validation_metrics = evaluate(model, loaders["validation"], device)
        alignment_metrics = validation_metrics["alignment"]
        confidence_metrics_value = validation_metrics["confidence"]
        print(
            f"epoch={epoch:03d} phase={phase} train_loss={train_loss:.5f} "
            f"validation_alignment_loss={alignment_metrics['loss']:.5f} "
            f"validation_sign_accuracy={alignment_metrics['sign_accuracy']:.4f} "
            f"validation_confidence_loss={confidence_metrics_value['loss']:.5f} "
            f"validation_confidence_mae={confidence_metrics_value['mae']:.5f}"
        )
        if phase == "alignment" and alignment_metrics["loss"] < best_validation_loss:
            best_validation_loss = alignment_metrics["loss"]
            best_metrics = {"epoch": epoch, "phase": phase, "validation": validation_metrics}
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                best_metrics,
                principle_to_idx,
                seed,
                feature_set,
                sgf_dir,
            )

    if not save_path.exists():
        save_checkpoint(
            save_path,
            model,
            optimizer,
            epochs,
            {"epoch": epochs, "phase": phase, "validation": validation_metrics},
            principle_to_idx,
            seed,
            feature_set,
            sgf_dir,
        )
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)
    checkpoint["metrics"]["test"] = test_metrics
    checkpoint["curriculum"] = {
        "confidence_epochs": confidence_epochs,
        "confidence_loss_weight": confidence_loss_weight,
        "balanced_sampling": balanced_sampling,
    }
    torch.save(checkpoint, save_path)
    print(json.dumps({"best_checkpoint": str(save_path), "metrics": checkpoint["metrics"]}, indent=2))


def evaluate_pairwise(model, loader, device):
    model.eval()
    score_a = []
    score_b = []
    preferences = []
    masks = []
    weights = []
    with torch.no_grad():
        for batch in loader:
            board_a = batch["board_a"].to(device)
            numeric_a = batch["numeric_a"].to(device)
            board_b = batch["board_b"].to(device)
            numeric_b = batch["numeric_b"].to(device)
            principle = batch["principle"].to(device)
            output_a = model(board_a, numeric_a, principle)
            output_b = model(board_b, numeric_b, principle)
            score_a.append(output_a["alignment"].cpu())
            score_b.append(output_b["alignment"].cpu())
            preferences.append(batch["preference"])
            masks.append(batch["mask"])
            weights.append(batch["weight"])
    if not score_a:
        return {"loss": 0.0, "comparison_accuracy": 0.0, "active_examples": 0}
    return pairwise_metrics(
        torch.cat(score_a),
        torch.cat(score_b),
        torch.cat(preferences),
        torch.cat(masks),
        torch.cat(weights),
    )


def train_pairwise_model(
    sgf_dir,
    comparisons_csv,
    save_path,
    feature_set="full",
    epochs=4,
    batch_size=48,
    learning_rate=1e-3,
    validation_fraction=0.13,
    test_fraction=0.14,
    seed=39,
    allow_parse_errors=False,
):
    if feature_set not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparisons = load_candidate_comparisons(comparisons_csv)
    parsed_moves, parse_errors = parse_sgf_collection(sgf_dir, allow_parse_errors=allow_parse_errors)
    pairs = build_pairwise_examples(comparisons, parsed_moves)
    splits = split_pairwise_by_position(pairs, seed, validation_fraction, test_fraction)
    principle_to_idx = build_principle_vocabulary(splits["train"])
    splits = attach_pairwise_principle_indices(splits, principle_to_idx)
    datasets = {name: CandidatePairDataset(frame, feature_set) for name, frame in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", num_workers=0)
        for name, dataset in datasets.items()
    }
    model = PrincipleAlignmentModel(
        num_principles=len(principle_to_idx),
        in_channels=len(FEATURE_SPECS[feature_set]["spatial_indices"]),
        numeric_dim=len(FEATURE_SPECS[feature_set]["numeric_indices"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_metrics = {}
    print(f"device={device}")
    print(f"feature_set={feature_set}")
    print(f"input_source=replayed_sgf_board_positions")
    print(f"training_mode=pairwise")
    print(f"principles={len(principle_to_idx)}")
    print(f"pair_rows={len(pairs)}")
    print(f"split_group=position_before_hash")
    print(f"parse_errors={len(parse_errors)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        active_examples = 0
        for batch in loaders["train"]:
            board_a = batch["board_a"].to(device)
            numeric_a = batch["numeric_a"].to(device)
            board_b = batch["board_b"].to(device)
            numeric_b = batch["numeric_b"].to(device)
            principle = batch["principle"].to(device)
            preference = batch["preference"].to(device)
            mask = batch["mask"].to(device)
            weight = batch["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output_a = model(board_a, numeric_a, principle)
            output_b = model(board_b, numeric_b, principle)
            loss = pairwise_ranking_loss(
                output_a["alignment"],
                output_b["alignment"],
                preference,
                mask,
                weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_active = int((mask > 0).sum().item())
            total_loss += loss.item() * max(batch_active, 1)
            active_examples += batch_active
        train_loss = total_loss / max(active_examples, 1)
        validation_metrics = evaluate_pairwise(model, loaders["validation"], device)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"validation_pairwise_loss={validation_metrics['loss']:.5f} "
            f"validation_comparison_accuracy={validation_metrics['comparison_accuracy']:.4f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_metrics = {"epoch": epoch, "validation": validation_metrics}
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                best_metrics,
                principle_to_idx,
                seed,
                feature_set,
                sgf_dir,
                training_mode="pairwise",
                comparisons_csv=comparisons_csv,
            )
    if not save_path.exists():
        save_checkpoint(
            save_path,
            model,
            optimizer,
            epochs,
            {"epoch": epochs, "validation": validation_metrics},
            principle_to_idx,
            seed,
            feature_set,
            sgf_dir,
            training_mode="pairwise",
            comparisons_csv=comparisons_csv,
        )
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["metrics"]["test"] = evaluate_pairwise(model, loaders["test"], device)
    torch.save(checkpoint, save_path)
    print(json.dumps({"best_checkpoint": str(save_path), "metrics": checkpoint["metrics"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alignment", "pairwise"], default="alignment")
    parser.add_argument("--sgf-dir", default="sgf")
    parser.add_argument("--comparisons-csv", default=None)
    parser.add_argument("--followed-csv", default="csv/principles_followed.csv")
    parser.add_argument("--violated-csv", default="csv/principles_violated.csv")
    parser.add_argument("--save-path", default="checkpoints/alignment_model.pt")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SPECS), default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.13)
    parser.add_argument("--test-fraction", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--confidence-epochs", type=int, default=1)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.2)
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.set_defaults(balanced_sampling=True)
    parser.add_argument("--allow-parse-errors", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "alignment":
        train_alignment_model(
            sgf_dir=arguments.sgf_dir,
            followed_csv=arguments.followed_csv,
            violated_csv=arguments.violated_csv,
            save_path=arguments.save_path,
            feature_set=arguments.feature_set,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            validation_fraction=arguments.validation_fraction,
            test_fraction=arguments.test_fraction,
            seed=arguments.seed,
            confidence_epochs=arguments.confidence_epochs,
            confidence_loss_weight=arguments.confidence_loss_weight,
            balanced_sampling=arguments.balanced_sampling,
            allow_parse_errors=arguments.allow_parse_errors,
        )
    else:
        if arguments.comparisons_csv is None:
            parser.error("--comparisons-csv is required when --mode pairwise is selected")
        train_pairwise_model(
            sgf_dir=arguments.sgf_dir,
            comparisons_csv=arguments.comparisons_csv,
            save_path=arguments.save_path,
            feature_set=arguments.feature_set,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            validation_fraction=arguments.validation_fraction,
            test_fraction=arguments.test_fraction,
            seed=arguments.seed,
            allow_parse_errors=arguments.allow_parse_errors,
        )
