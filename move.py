import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sgfmill import boards, sgf

import train as training


class MoveEngine:
    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.artifact = torch.load(self.checkpoint_path, map_location=self.device)
        self.feature_set = self.artifact["feature_set"]
        self.principle_to_idx = self.artifact["principle_to_idx"]
        model_config = self.artifact["model_config"]
        self.model = training.PrincipleAlignmentModel(
            num_principles=model_config["num_principles"],
            in_channels=model_config["in_channels"],
            numeric_dim=model_config["numeric_dim"],
            embed_dim=model_config.get("embed_dim", 32),
        ).to(self.device)
        self.model.load_state_dict(self.artifact["model_state_dict"])
        self.model.eval()
        self.yin_yang_dimensions = self.artifact.get("yin_yang_dimensions", training.YIN_YANG_DIMENSIONS)
        self.numeric_feature_names = self.artifact.get("numeric_feature_names", training.NUMERIC_FEATURE_NAMES)

    def board_from_position(self, position):
        board = boards.Board(training.BOARD_SIZE)
        black_points = []
        white_points = []
        if len(position) != training.BOARD_AREA:
            raise ValueError(f"Position must contain {training.BOARD_AREA} characters")
        for row in range(training.BOARD_SIZE):
            for column in range(training.BOARD_SIZE):
                value = position[row * training.BOARD_SIZE + column]
                if value == "X":
                    black_points.append((row, column))
                elif value == "O":
                    white_points.append((row, column))
                elif value != ".":
                    raise ValueError("Position may contain only X, O, and .")
        board.apply_setup(black_points, white_points, [])
        return board

    def ensure_board(self, board):
        if isinstance(board, str):
            return self.board_from_position(board)
        if not hasattr(board, "side") or not hasattr(board, "get"):
            raise TypeError("Board must be a 361-character position string or an sgfmill board")
        if board.side != training.BOARD_SIZE:
            raise ValueError(f"Board must be {training.BOARD_SIZE}x{training.BOARD_SIZE}")
        return board

    def base_row(self, board, color, move_number, stage):
        position_before = training.board_string(board)
        return {
            "sgf_file": "runtime.sgf",
            "game_id": "runtime",
            "move_number": int(move_number),
            "color": str(color).upper(),
            "stage": stage,
            "position_before": position_before,
            "position_before_hash": training.board_hash(board),
            "coordinate_sgf": "pass",
            "legal_status": "unknown",
            "captured_count": 0,
            "captured_coordinates": "[]",
        }

    def simulate_candidate(self, board, color, move_number, stage, candidate):
        base = self.base_row(board, color, move_number, stage)
        row = training.candidate_row_from_position(base, candidate)
        row["coordinate_human"] = training.coordinate_human(
            training.sgf_coordinate_to_row_col(row["coordinate_sgf"]), training.BOARD_SIZE
        )
        return row

    def candidate_priority(self, row):
        before = training.board_array_from_position(row["position_before"])
        after = training.board_array_from_position(row["position_after"])
        metrics = training.transition_metrics(
            before,
            after,
            row["color"],
            row["coordinate_sgf"],
            row["legal_status"],
            row["captured_count"],
        )
        point = training.sgf_coordinate_to_row_col(row["coordinate_sgf"])
        center_distance = 0.0
        if point is not None:
            center = (training.BOARD_SIZE - 1) / 2.0
            center_distance = abs(point[0] - center) + abs(point[1] - center)
        tactical = (
            float(row["captured_count"]) * 8.0
            + metrics["opponent_atari_count"] * 3.0
            + metrics["own_liberties"] * 1.5
            + metrics["enemy_neighbors"] * 0.5
        )
        strategic = max(0.0, 8.0 - center_distance * 0.25)
        if row["coordinate_sgf"] == "pass":
            strategic = -1.0
        return tactical + strategic

    def generate_candidates(self, board, color, move_number, stage, candidate_limit=32):
        board = self.ensure_board(board)
        simulated = []
        for row, column in training.all_points(training.BOARD_SIZE):
            candidate = training.coordinate_sgf((row, column))
            record = self.simulate_candidate(board, color, move_number, stage, candidate)
            if record["legal_status"] == "legal":
                record["priority"] = self.candidate_priority(record)
                simulated.append(record)
        pass_record = self.simulate_candidate(board, color, move_number, stage, "pass")
        pass_record["priority"] = self.candidate_priority(pass_record)
        simulated.append(pass_record)
        simulated.sort(key=lambda value: value["priority"], reverse=True)
        tactical = [
            row for row in simulated
            if row["legal_status"] == "legal" and int(row["captured_count"]) > 0
        ]
        urgent = [
            row for row in simulated
            if row["legal_status"] == "legal" and self.candidate_priority(row) >= 3.0
        ]
        selected = []
        seen = set()
        for row in tactical + urgent + simulated:
            key = row["coordinate_sgf"]
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= max(1, int(candidate_limit)):
                break
        return selected

    def context_vector(self, board_array, color, stage):
        player = "X" if str(color).upper() == "B" else "O"
        opponent = "O" if player == "X" else "X"
        own = training.color_state_features(board_array, player)
        enemy = training.color_state_features(board_array, opponent)
        own_weakness = float(np.clip((own["atari_groups"] + 1) / max(own["group_count"] + 1, 1), 0.0, 1.0))
        enemy_weakness = float(np.clip((enemy["atari_groups"] + 1) / max(enemy["group_count"] + 1, 1), 0.0, 1.0))
        local_urgency = float(np.clip((own["atari_groups"] + enemy["atari_groups"]) / 4.0, 0.0, 1.0))
        board_openness = float(np.clip((board_array == ".").mean(), 0.0, 1.0))
        phase_values = {"opening": 0.0, "middle_game": 0.5, "endgame": 1.0, "unknown": 0.5}
        phase = phase_values.get(str(stage), 0.5)
        spread = float(np.clip((own["bounding_box_area"] + enemy["bounding_box_area"]) / (2.0 * training.BOARD_AREA), 0.0, 1.0))
        return np.asarray([own_weakness, enemy_weakness, local_urgency, board_openness, phase, spread], dtype=np.float32)

    def contextual_weights_and_target(self, context):
        own_weakness, enemy_weakness, urgency, openness, phase, spread = context
        relevance = np.asarray(
            [
                1.0 + openness + phase,
                1.0 + 2.0 * own_weakness + urgency,
                1.0 + own_weakness + enemy_weakness + urgency,
                1.0 + openness + phase,
                1.0 + 2.0 * urgency + enemy_weakness,
                1.0 + openness + spread,
            ],
            dtype=np.float32,
        )
        relevance = relevance - relevance.max()
        weights = np.exp(relevance) / np.exp(relevance).sum()
        target = np.asarray(
            [
                0.45 * phase - 0.25 * openness,
                0.65 * own_weakness - 0.35 * enemy_weakness,
                0.60 * own_weakness - 0.60 * enemy_weakness,
                0.45 * openness - 0.45 * phase,
                0.65 * enemy_weakness + 0.35 * urgency - 0.25 * own_weakness,
                0.25 * (openness - spread),
            ],
            dtype=np.float32,
        )
        return weights.astype(np.float32), np.clip(target, -1.0, 1.0)

    def yin_yang_score(self, row):
        before = training.board_array_from_position(row["position_before"])
        after = training.board_array_from_position(row["position_after"])
        player = "X" if row["color"] == "B" else "O"
        before_vector = training.yin_yang_state_vector(before, player)
        after_vector = training.yin_yang_state_vector(after, player)
        delta = after_vector - before_vector
        context = self.context_vector(before, row["color"], row.get("stage", "unknown"))
        weights, target = self.contextual_weights_and_target(context)
        distance = np.abs(after_vector - target)
        balance = 1.0 - np.clip(distance / 2.0, 0.0, 1.0)
        desired = target - before_vector
        direction = np.where(
            np.abs(desired) < 0.1,
            1.0 - np.clip(np.abs(delta) / 2.0, 0.0, 1.0),
            np.clip(np.sign(desired) * delta / 2.0, -1.0, 1.0),
        )
        quality = balance * direction
        score = float(np.clip(np.dot(weights, quality), -1.0, 1.0))
        return {
            "before": before_vector.astype(float).tolist(),
            "after": after_vector.astype(float).tolist(),
            "delta": delta.astype(float).tolist(),
            "context": context.astype(float).tolist(),
            "weights": weights.astype(float).tolist(),
            "target": target.astype(float).tolist(),
            "dimension_quality": quality.astype(float).tolist(),
            "score": score,
        }

    def selected_principles(self, principle_codes):
        if principle_codes is None:
            return list(self.principle_to_idx.keys())
        unknown = [code for code in principle_codes if code not in self.principle_to_idx]
        if unknown:
            raise ValueError(f"Principles are not present in checkpoint: {unknown}")
        return list(dict.fromkeys(principle_codes))

    def score_candidates(self, candidate_rows, principle_codes=None):
        principles = self.selected_principles(principle_codes)
        if not principles:
            raise ValueError("Checkpoint contains no principles")
        cache = {}
        encoded = []
        for row in candidate_rows:
            spatial, numeric, _ = training.encode_candidate_row(row, self.feature_set, cache)
            encoded.append((row, spatial, numeric))
        board_batch = []
        numeric_batch = []
        principle_batch = []
        lookup = []
        for candidate_index, (_, spatial, numeric) in enumerate(encoded):
            for principle_code in principles:
                board_batch.append(spatial[training.FEATURE_SPECS[self.feature_set]["spatial_indices"]])
                numeric_batch.append(numeric[training.FEATURE_SPECS[self.feature_set]["numeric_indices"]])
                principle_batch.append(self.principle_to_idx[principle_code])
                lookup.append((candidate_index, principle_code))
        with torch.no_grad():
            outputs = self.model(
                torch.stack(board_batch).to(self.device),
                torch.stack(numeric_batch).to(self.device),
                torch.tensor(principle_batch, dtype=torch.long, device=self.device),
            )
        alignment = torch.tanh(outputs["alignment"]).cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        per_candidate = {index: {} for index in range(len(candidate_rows))}
        for index, (candidate_index, principle_code) in enumerate(lookup):
            per_candidate[candidate_index][principle_code] = {
                "alignment": float(alignment[index]),
                "confidence": float(confidence[index]),
            }
        results = []
        for candidate_index, row in enumerate(candidate_rows):
            principle_scores = per_candidate[candidate_index]
            alignment_score = float(np.mean([value["alignment"] for value in principle_scores.values()]))
            annotation_confidence = float(np.mean([value["confidence"] for value in principle_scores.values()]))
            yin_yang = self.yin_yang_score(row)
            context = np.asarray(yin_yang["context"], dtype=np.float32)
            yin_yang_relevance = float(np.clip(0.20 + 0.55 * context[2] + 0.25 * (1.0 - context[3]), 0.0, 1.0))
            final_score = alignment_score + yin_yang_relevance * yin_yang["score"]
            results.append(
                {
                    "move_sgf": row["coordinate_sgf"],
                    "move_human": row["coordinate_human"],
                    "legal": row["legal_status"] in {"legal", "pass"},
                    "captured_count": int(row["captured_count"]),
                    "alignment_score": alignment_score,
                    "annotation_confidence": annotation_confidence,
                    "yin_yang_score": yin_yang["score"],
                    "yin_yang_relevance": yin_yang_relevance,
                    "yin_yang_contribution": yin_yang_relevance * yin_yang["score"],
                    "final_score": float(final_score),
                    "per_principle": principle_scores,
                    "yin_yang": yin_yang,
                }
            )
        results.sort(key=lambda value: value["final_score"], reverse=True)
        return results

    def choose_move(
        self,
        board,
        color,
        move_number=1,
        stage="unknown",
        candidate_limit=32,
        principle_codes=None,
    ):
        board = self.ensure_board(board)
        candidates = self.generate_candidates(board, color, move_number, stage, candidate_limit)
        scored = self.score_candidates(candidates, principle_codes)
        if not scored:
            raise RuntimeError("No legal candidates were available")
        return {
            "selected_move": scored[0]["move_sgf"],
            "selected_move_human": scored[0]["move_human"],
            "candidates": scored,
            "principle_codes": self.selected_principles(principle_codes),
            "yin_yang_dimensions": self.yin_yang_dimensions,
            "numeric_feature_names": self.numeric_feature_names,
            "warnings": [
                "V1 scores learned principle alignment and a contextual engineered Yin–Yang score.",
                "The alignment checkpoint does not prove objective move strength.",
                "Deception and sacrifice require opponent-response and delayed-effect analysis.",
                "Full ko and superko history must be enforced by the surrounding game controller.",
            ],
        }


def choose_move(board, color, checkpoint_path="checkpoints/alignment_model.pt", **kwargs):
    engine = MoveEngine(checkpoint_path)
    return engine.choose_move(board, color, **kwargs)


def read_position_file(path):
    position = Path(path).read_text(encoding="utf-8").strip()
    if len(position) != training.BOARD_AREA:
        raise ValueError(f"Position file must contain exactly {training.BOARD_AREA} board characters")
    return position


def load_sgf_state(path, target_move_number=None):
    path = Path(path)
    game = sgf.Sgf_game.from_bytes(path.read_bytes())
    if game.get_size() != training.BOARD_SIZE:
        raise ValueError(f"Expected a 19x19 SGF but found {game.get_size()}x{game.get_size()}")
    sequence = game.get_main_sequence()
    root = sequence[0]
    total_moves = sum(1 for node in sequence if node.get_move()[0] is not None)
    board = boards.Board(training.BOARD_SIZE)
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
        if point is None:
            after = before.copy()
        else:
            board.play(point[0], point[1], color)
            after = board.copy()
        records.append(
            {
                "position_before": training.board_string(before),
                "position_after": training.board_string(after),
                "color": "B" if color == "b" else "W",
                "move_number": move_number,
                "stage": training.stage_for(move_number, total_moves),
            }
        )
    if not records:
        raise ValueError(f"SGF contains no playable moves: {path}")
    if target_move_number is not None:
        target = [record for record in records if int(record["move_number"]) == int(target_move_number)]
        if not target:
            raise ValueError(f"SGF has no main-line move number {target_move_number}")
        record = target[0]
        return {
            "position": record["position_before"],
            "color": record["color"],
            "move_number": int(record["move_number"]),
            "stage": record["stage"],
        }
    last = records[-1]
    next_color = "W" if str(last["color"]).upper() == "B" else "B"
    next_move_number = int(last["move_number"]) + 1
    return {
        "position": last["position_after"],
        "color": next_color,
        "move_number": next_move_number,
        "stage": training.stage_for(next_move_number, max(next_move_number, len(records))),
    }


def resolve_input(arguments):
    supplied = [arguments.position is not None, arguments.position_file is not None, arguments.sgf_file is not None]
    if sum(supplied) != 1:
        raise ValueError("Exactly one of --position, --position-file, or --sgf-file is required")
    if arguments.position is not None:
        if arguments.color is None:
            raise ValueError("--color is required with --position")
        return {
            "position": arguments.position.strip(),
            "color": arguments.color.upper(),
            "move_number": arguments.move_number,
            "stage": arguments.stage,
        }
    if arguments.position_file is not None:
        if arguments.color is None:
            raise ValueError("--color is required with --position-file")
        return {
            "position": read_position_file(arguments.position_file),
            "color": arguments.color.upper(),
            "move_number": arguments.move_number,
            "stage": arguments.stage,
        }
    state = load_sgf_state(arguments.sgf_file, arguments.sgf_move_number)
    if arguments.color is not None:
        state["color"] = arguments.color.upper()
    if arguments.move_number != 1:
        state["move_number"] = arguments.move_number
    if arguments.stage != "unknown":
        state["stage"] = arguments.stage
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Generate and rank legal Go moves from a board position or SGF state."
    )
    parser.add_argument("--checkpoint", default="checkpoints/alignment_model.pt")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--position", default=None)
    input_group.add_argument("--position-file", default=None)
    input_group.add_argument("--sgf-file", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sgf-move-number", type=int, default=None)
    parser.add_argument("--color", choices=["B", "W", "b", "w"], default=None)
    parser.add_argument("--move-number", type=int, default=1)
    parser.add_argument("--stage", choices=["opening", "middle_game", "endgame", "unknown"], default="unknown")
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--principles", default=None)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    has_input = any(
        value is not None for value in [arguments.position, arguments.position_file, arguments.sgf_file]
    )
    if arguments.demo and has_input:
        parser.error("--demo cannot be combined with --position, --position-file, or --sgf-file")
    if not arguments.demo and not has_input:
        parser.print_help()
        print("\nProvide --position, --position-file, or --sgf-file, or run with --demo for an empty-board test.")
        return
    if arguments.demo:
        state = {
            "position": "." * training.BOARD_AREA,
            "color": "B",
            "move_number": 1,
            "stage": "opening",
        }
        source = "demo"
    else:
        try:
            state = resolve_input(arguments)
        except ValueError as error:
            parser.error(str(error))
        source = "position" if arguments.position is not None else "position_file" if arguments.position_file is not None else "sgf_file"
    principles = None
    if arguments.principles:
        principles = [value.strip() for value in arguments.principles.split(",") if value.strip()]
    result = choose_move(
        state["position"],
        state["color"],
        checkpoint_path=arguments.checkpoint,
        move_number=state["move_number"],
        stage=state["stage"],
        candidate_limit=arguments.candidate_limit,
        principle_codes=principles,
    )
    result["input"] = {
        "source": source,
        "move_number": state["move_number"],
        "color": state["color"],
        "stage": state["stage"],
    }
    text = json.dumps(result, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
