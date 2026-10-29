import argparse
import json
from pathlib import Path

import numpy as np
from sgfmill import boards, sgf

import train as training

BOARD_SIZE = training.BOARD_SIZE
BOARD_AREA = training.BOARD_AREA
DIRECTIONS = training.DIRECTIONS
EMPTY, BLACK, WHITE = 0, 1, 2
COLUMN_LETTERS = 'ABCDEFGHJKLMNOPQRST'
STAR_POINTS = [(3, 3), (3, 15), (15, 3), (15, 15), (3, 9), (9, 3), (9, 15), (15, 9), (9, 9)]
DIMENSIONS = ['territory_influence', 'stability_activity', 'defense_attack', 'emptiness_fullness', 'sente_gote', 'local_global']
CONTEXT_NAMES = ['own_weakness', 'enemy_weakness', 'local_urgency', 'board_openness', 'game_phase', 'board_shape']
WARNINGS = [
    'Alignment is learned from annotated principle labels and does not prove move strength.',
    'Opponent-intent signals are probabilistic evidence, not conclusions.',
    'Deception and sacrifice are not searched: there is no opponent-reply lookahead in this version.',
    'Ko and superko must be enforced by the game controller.',
    'Numeric features 5-16 use assumed board_evidence definitions until checked with --verify-evidence.',
    'Passing uses an empty-region territory rule, not scoring: dead stones are not recognised.'
]
PARAMETERS = {
    'history_plies': 20,
    'answer_radius': 2,
    'nearby_radius': 1,
    'zone_edge': 4,
    'influence_sigma': 2.5,
    'influence_threshold': 0.3,
    'engagement_radius': 2,
    'spread_norm': 10.0,
    'smoothing': 1.0,
    'phase_moves': 200.0,
    'weakness': {1: 1.0, 2: 0.5, 3: 0.15},
    'target_matrix': [
        [0.6, -0.3, 0.2, 0.0, 0.0, -1.2, 0.0],
        [0.0, -0.7, 0.5, 0.0, 0.2, 0.0, 0.3],
        [0.0, -0.8, 0.8, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        [0.5, -0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.8, 0.6, 0.2, 0.0]
    ],
    'relevance_matrix': [
        [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        [1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
    ],
    'static_gain': 20.0,
    'pair_gain': 2.0,
    'y_weights': [0.4, 0.3, 0.3],
    'relevance_coefficients': [0.2, 0.45, 0.2, 0.25],
    'threat_liberties': 2,
    'intent_minimum_moves': 4,
    'intent_radius': 2,
    'intent_sigma': 1.5,
    'intent_reference': 0.3,
    'intent_coefficients': [2.0, 2.0, 2.0],
    'shift_free': 4.0,
    'shift_scale': 8.0,
    'focus_spread': 6.0,
    'opening_distance': 4,
    'opening_spacing': 5,
    'pass_min_move': 100
}
KERNELS = {}

def coordinate_sgf(point): return 'pass' if point is None else chr(ord('a') + point[1]) + chr(ord('a') + point[0])

def coordinate_human(point): return 'pass' if point is None else f'{COLUMN_LETTERS[point[1]]}{BOARD_SIZE - point[0]}'

def mirror_point(point): return None if point is None else (BOARD_SIZE - 1 - point[0], point[1])

def chebyshev(first, second): return max(abs(first[0] - second[0]), abs(first[1] - second[1]))

def parse_color(value):
    if value in (BLACK, WHITE): return int(value)
    text = str(value).strip().upper()
    if text == 'B': return BLACK
    if text == 'W': return WHITE
    raise ValueError('Color must be B or W')

def parse_history(history):
    result = []
    for color, coordinate in history or []:
        point = coordinate if isinstance(coordinate, tuple) else training.coordinate_to_point(coordinate)
        result.append((parse_color(color), point))
    return result

def board_from_position(position):
    if not isinstance(position, str) or len(position) != BOARD_AREA: raise ValueError(f'Position must contain {BOARD_AREA} characters')
    lookup = {'.': EMPTY, 'X': BLACK, 'O': WHITE}
    if any(value not in lookup for value in position): raise ValueError('Position may contain only X, O, and .')
    return np.asarray([lookup[value] for value in position], dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)

def array_from_sgfmill(board):
    lookup = {None: EMPTY, 'b': BLACK, 'w': WHITE}
    return np.asarray([[lookup[board.get(BOARD_SIZE - 1 - row, column)] for column in range(BOARD_SIZE)] for row in range(BOARD_SIZE)], dtype=np.int8)

class BoardView:
    def __init__(self, array): self.array = array

    def get(self, row, column): return (None, 'b', 'w')[int(self.array[BOARD_SIZE - 1 - row, column])]

def neighbors(row, column):
    for row_delta, column_delta in DIRECTIONS:
        next_row, next_column = row + row_delta, column + column_delta
        if 0 <= next_row < BOARD_SIZE and 0 <= next_column < BOARD_SIZE: yield next_row, next_column

def group_at(board, row, column):
    color = board[row, column]
    stones, liberties, stack = set(), set(), [(row, column)]
    while stack:
        point = stack.pop()
        if point in stones: continue
        stones.add(point)
        for next_point in neighbors(*point):
            value = board[next_point]
            if value == EMPTY: liberties.add(next_point)
            elif value == color and next_point not in stones: stack.append(next_point)
    return stones, liberties

def all_groups(board, color):
    seen, groups = set(), []
    for row, column in np.argwhere(board == color).tolist():
        if (row, column) in seen: continue
        stones, liberties = group_at(board, row, column)
        seen |= stones
        groups.append((stones, liberties))
    return groups

def play_stone(board, row, column, color):
    if board[row, column] != EMPTY: return None
    after = board.copy()
    after[row, column] = color
    captured = []
    for point in neighbors(row, column):
        if after[point] != 3 - color: continue
        stones, liberties = group_at(after, *point)
        if not liberties:
            captured.extend(stones)
            for stone in stones: after[stone] = EMPTY
    if not group_at(after, row, column)[1]: return None
    return after, sorted(captured)

def weakness(liberty_count): return PARAMETERS['weakness'].get(int(liberty_count), 0.0)

def safety(liberty_count): return 1.0 - weakness(liberty_count)

def ratio(yang, yin): return float((yang - yin) / (yang + yin + PARAMETERS['smoothing']))

def blur(values):
    sigma = PARAMETERS['influence_sigma']
    if sigma not in KERNELS:
        index = np.arange(BOARD_SIZE, dtype=np.float32)
        KERNELS[sigma] = np.exp(-((index[:, None] - index[None, :]) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel = KERNELS[sigma]
    return kernel @ values @ kernel.T

def dilate(mask, radius):
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    for row_shift in range(2 * radius + 1):
        for column_shift in range(2 * radius + 1): result |= padded[row_shift:row_shift + BOARD_SIZE, column_shift:column_shift + BOARD_SIZE]
    return result

def neighbor_any(mask):
    result = np.zeros_like(mask)
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result

def vulnerability(groups):
    total = sum(len(stones) for stones, _ in groups)
    if not total: return 0.0
    return float(min(1.0, sum(len(stones) * weakness(len(liberties)) for stones, liberties in groups) / total))

def initiative(history, player):
    mine, theirs = [], []
    plies = history[-PARAMETERS['history_plies']:]
    for (previous_color, previous_point), (color, point) in zip(plies, plies[1:]):
        if previous_color == color or previous_point is None or point is None: continue
        answered = 1.0 if chebyshev(previous_point, point) <= PARAMETERS['answer_radius'] else 0.0
        (mine if color == player else theirs).append(answered)
    return float((np.mean(theirs) if theirs else 0.0) - (np.mean(mine) if mine else 0.0))

def context_vector(board, player, move_number):
    own_groups = all_groups(board, player)
    opponent_groups = all_groups(board, 3 - player)
    urgency = max([weakness(len(liberties)) * min(1.0, len(stones) / 4.0) for stones, liberties in own_groups + opponent_groups], default=0.0)
    own_field = blur((board == player).astype(np.float32))
    opponent_field = blur((board == 3 - player).astype(np.float32))
    shape = 0.0 if own_field.sum() <= 0 or opponent_field.sum() <= 0 else float(np.minimum(own_field / own_field.sum(), opponent_field / opponent_field.sum()).sum())
    phase = min(max(move_number / PARAMETERS['phase_moves'], 0.0), 1.0)
    return np.asarray([vulnerability(own_groups), vulnerability(opponent_groups), urgency, float((board == EMPTY).mean()), phase, shape], dtype=np.float32)

def state_vector(board, player, initiative_value):
    own_mask = board == player
    opponent_mask = board == 3 - player
    empty_mask = board == EMPTY
    controlled = empty_mask & ((blur(own_mask.astype(np.float32)) - blur(opponent_mask.astype(np.float32))) > PARAMETERS['influence_threshold'])
    frontier = controlled & neighbor_any((empty_mask & ~controlled) | opponent_mask)
    enclosed = controlled & ~frontier
    engaged = own_mask & dilate(opponent_mask, PARAMETERS['engagement_radius'])
    quiet = own_mask & ~engaged
    attack = sum(len(stones) * weakness(len(liberties)) for stones, liberties in all_groups(board, 3 - player))
    defense = sum(len(stones) * weakness(len(liberties)) for stones, liberties in all_groups(board, player))
    stones = np.argwhere(own_mask)
    spread = 0.0 if not len(stones) else float(np.clip(2.0 * np.sqrt(((stones - stones.mean(axis=0)) ** 2).sum(axis=1).mean()) / PARAMETERS['spread_norm'] - 1.0, -1.0, 1.0))
    return np.asarray([
        ratio(frontier.sum(), enclosed.sum()),
        ratio(engaged.sum(), quiet.sum()),
        ratio(attack, defense),
        float(2.0 * (board != EMPTY).sum() / BOARD_AREA - 1.0),
        float(np.clip(initiative_value, -1.0, 1.0)),
        spread
    ], dtype=np.float32)

def targets_and_weights(context):
    extended = np.concatenate([[1.0], context]).astype(np.float32)
    target = np.clip(np.asarray(PARAMETERS['target_matrix'], dtype=np.float32) @ extended, -1.0, 1.0)
    weights = np.maximum(np.asarray(PARAMETERS['relevance_matrix'], dtype=np.float32) @ extended, 0.05)
    return target, weights / weights.sum()

def static_progress(before, after, target, weights):
    progress = (np.abs(before - target) - np.abs(after - target)) / (1.0 + np.abs(target))
    return float(np.dot(weights, progress)), progress

def pair_score(board, after, row, column, player):
    saved_stones, gains = 0, 0.0
    for stones, liberties in all_groups(board, player):
        if weakness(len(liberties)) <= 0.0: continue
        gain = safety(len(group_at(after, *next(iter(stones)))[1])) - safety(len(liberties))
        if gain > 0.0:
            saved_stones += len(stones)
            gains += len(stones) * gain
    if not saved_stones: return 0.0, 0.0, 0.0
    yang = gains / saved_stones
    lone_liberties = sum(int(after[point] == EMPTY) for point in neighbors(row, column))
    yin = max(0.0, safety(len(group_at(after, row, column)[1])) - safety(lone_liberties))
    reciprocity = 0.0 if max(yang, yin) <= 0.0 else min(yang, yin) / max(yang, yin)
    return float(reciprocity * np.tanh(PARAMETERS['pair_gain'] * (yang + yin))), float(yang), float(yin)

def proximity_map(points):
    result = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    rows, columns = np.indices((BOARD_SIZE, BOARD_SIZE))
    for row, column in points:
        distance = np.maximum(np.abs(rows - row), np.abs(columns - column))
        result = np.maximum(result, np.exp(-(distance ** 2) / (2.0 * PARAMETERS['intent_sigma'] ** 2)))
    return result

def contact(groups, moves):
    radius = PARAMETERS['intent_radius']
    return [any(chebyshev(stone, move) <= radius for stone in stones for move in moves) for stones, _ in groups]

def weighted_fraction(selected, total):
    denominator = sum(len(stones) * weakness(len(liberties)) for stones, liberties in total)
    if denominator <= 0: return 0.0
    return float(sum(len(stones) * weakness(len(liberties)) for stones, liberties in selected) / denominator)

def intent_signals(board, player, history):
    plies = history[-PARAMETERS['history_plies']:]
    opponent_moves = [point for color, point in plies if color == 3 - player and point is not None]
    own_moves = [point for color, point in plies if color == player and point is not None]
    if len(opponent_moves) < PARAMETERS['intent_minimum_moves']: return None
    half = len(opponent_moves) // 2
    earlier = np.asarray(opponent_moves[:half], dtype=np.float32)
    recent = np.asarray(opponent_moves[half:], dtype=np.float32)
    shift = float(np.linalg.norm(recent.mean(axis=0) - earlier.mean(axis=0)))
    spread = float(np.sqrt(((recent - recent.mean(axis=0)) ** 2).sum(axis=1).mean()))
    concentration = float(np.clip((shift - PARAMETERS['shift_free']) / PARAMETERS['shift_scale'], 0.0, 1.0) * np.clip(1.0 - spread / PARAMETERS['focus_spread'], 0.0, 1.0))
    threat = PARAMETERS['threat_liberties']
    own_weak = [group for group in all_groups(board, player) if len(group[1]) <= threat]
    opponent_weak = [group for group in all_groups(board, 3 - player) if len(group[1]) <= threat + 1]
    touched = contact(own_weak, opponent_moves)
    ignored = [group for group, flag in zip(own_weak, touched) if not flag]
    attacked = [group for group, flag in zip(own_weak, touched) if flag]
    neglect_own = weighted_fraction(ignored, own_weak)
    unattacked = [group for group, flag in zip(opponent_weak, contact(opponent_weak, own_moves)) if not flag]
    neglect_enemy = weighted_fraction(unattacked, opponent_weak)
    threatened_liberties = set().union(*[liberties for _, liberties in own_weak]) if own_weak else set()
    enemy_liberties = set().union(*[liberties for _, liberties in opponent_weak]) if opponent_weak else set()
    shared = threatened_liberties & enemy_liberties
    shared_fraction = len(shared) / len(threatened_liberties) if threatened_liberties else 0.0
    signals = np.asarray([concentration, neglect_own, shared_fraction], dtype=np.float32)
    logit = float(np.dot(np.asarray(PARAMETERS['intent_coefficients'], dtype=np.float32), signals - PARAMETERS['intent_reference']))
    probability = float(1.0 / (1.0 + np.exp(-logit)))
    apparent_points = [point for stones, liberties in attacked for point in stones | liberties] or opponent_moves[half:]
    hidden_points = [point for stones, liberties in ignored for point in stones | liberties] + sorted(shared)
    return {
        'concentration': concentration,
        'neglect_own': neglect_own,
        'neglect_enemy': neglect_enemy,
        'shared_liberties': shared_fraction,
        'probability': probability,
        'uncertainty': 4.0 * probability * (1.0 - probability),
        'apparent': proximity_map(apparent_points),
        'hidden': proximity_map(hidden_points)
    }

def board_evidence(board, player, point, move_number, captured, after):
    row, column = point
    radius = PARAMETERS['nearby_radius']
    window = board[max(row - radius, 0):row + radius + 1, max(column - radius, 0):column + radius + 1]
    edge_row, edge_column = min(row, BOARD_SIZE - 1 - row), min(column, BOARD_SIZE - 1 - column)
    limit = PARAMETERS['zone_edge']
    zone = 'corner' if edge_row < limit and edge_column < limit else 'side' if min(edge_row, edge_column) < limit else 'center'
    stones, liberties = group_at(after, row, column)
    return {
        'friendly_neighbour_count': sum(int(board[next_point] == player) for next_point in neighbors(row, column)),
        'enemy_neighbour_count': sum(int(board[next_point] == 3 - player) for next_point in neighbors(row, column)),
        'nearby_friendly_count': int((window == player).sum()),
        'nearby_enemy_count': int((window == 3 - player).sum()),
        'near_empty': int((window == EMPTY).sum()) > 1,
        'captured_count': len(captured),
        'own_group_after': {'size': len(stones), 'liberties': len(liberties)},
        'move_number': int(move_number),
        'zone': zone
    }

def empty_regions(board):
    seen, regions = set(), []
    for row, column in np.argwhere(board == EMPTY).tolist():
        if (row, column) in seen: continue
        points, borders, stack = set(), set(), [(row, column)]
        while stack:
            point = stack.pop()
            if point in points: continue
            points.add(point)
            for next_point in neighbors(*point):
                value = board[next_point]
                if value == EMPTY: stack.append(next_point)
                else: borders.add(int(value))
        seen |= points
        regions.append((points, borders))
    return regions

def contested_points(board):
    if not ((board == BLACK).any() and (board == WHITE).any()): return None
    return {point for points, borders in empty_regions(board) if len(borders) == 2 for point in points}

def strategic_openings(board, count):
    stones = np.argwhere(board != EMPTY)
    if not len(stones): return list(STAR_POINTS)
    points = [(row, column) for row in range(2, BOARD_SIZE - 2) for column in range(2, BOARD_SIZE - 2) if board[row, column] == EMPTY]
    if not points: return []
    array = np.asarray(points)
    distance = np.abs(array[:, None, :] - stones[None, :, :]).max(axis=2).min(axis=1).astype(np.float32)
    score = distance - 0.001 * (np.abs(array[:, 0] - 9) + np.abs(array[:, 1] - 9))
    chosen = []
    for index in np.argsort(-score, kind='stable'):
        if distance[index] < PARAMETERS['opening_distance'] or len(chosen) >= count: break
        if all(chebyshev(points[index], other) >= PARAMETERS['opening_spacing'] for other in chosen): chosen.append(points[index])
    return chosen

def candidate_pool(board, player, history, count, openings, shape, allowed=None):
    priority = {}
    def bump(point, value): priority[point] = max(priority.get(point, -99.0), value)
    for stones, liberties in all_groups(board, 3 - player):
        if len(liberties) <= 2:
            for point in liberties: bump(point, (12.0 if len(liberties) == 1 else 6.0) + 0.3 * min(len(stones), 10))
    for stones, liberties in all_groups(board, player):
        if len(liberties) <= 2:
            for point in liberties: bump(point, (11.0 if len(liberties) == 1 else 5.0) + 0.3 * min(len(stones), 10))
    last = next((point for color, point in reversed(history) if color == 3 - player and point is not None), None)
    near_stones = dilate(board != EMPTY, 2) & (board == EMPTY)
    for row, column in np.argwhere(near_stones).tolist():
        line = min(row, column, BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - column)
        touching = sum(int(board[point] != EMPTY) for point in neighbors(row, column))
        bump((row, column), touching + (2.0 if last is not None and chebyshev((row, column), last) <= 2 else 0.0) + (-3.0 if line == 0 else -1.5 if line == 1 else 0.0))
    ranked = sorted(priority, key=lambda point: -priority[point])
    tactical = [point for point in ranked if priority[point] >= 5.0]
    local = [point for point in ranked if priority[point] < 5.0]
    opening_count = openings if not (board != EMPTY).any() else max(1, int(round(openings * (1.0 - shape))))
    head = max(1, count // 2)
    ordered = tactical[:head] + strategic_openings(board, opening_count) + local + tactical[head:]
    ordered = list(dict.fromkeys(ordered))
    if allowed is None: return ordered
    listed = set(ordered)
    return [point for point in ordered if point in allowed] + sorted(point for point in allowed if point not in listed)

def generate_candidates(board, player, history, count, openings, shape, allowed=None):
    records = []
    for point in candidate_pool(board, player, history, count, openings, shape, allowed):
        result = play_stone(board, point[0], point[1], player)
        if result is None: continue
        records.append({'point': point, 'after': result[0], 'captured': result[1]})
        if len(records) >= max(1, int(count)): break
    return records

def resolve(model, name):
    owner = model
    parts = name.split('.')
    for part in parts[:-1]: owner = getattr(owner, part)
    return owner, parts[-1]

def trunk_features(model, board):
    conv_in, _ = training.conv_forward(board, model.conv_in_weight, model.conv_in_bias)
    norm_in, _ = training.batch_norm_forward(conv_in, model.norm_in_gamma, model.norm_in_beta, False, model.norm_in_running_mean, model.norm_in_running_var)
    features = model.residual_2.forward(model.residual_1.forward(training.relu(norm_in), False), False)
    return features.mean(axis=(2, 3))

def head_outputs(model, board_features, principle, numeric):
    combined = np.concatenate([board_features, model.principle_embedding[principle], numeric], axis=1)
    alignment = model.alignment_linear_2.forward(training.relu(model.alignment_linear_1.forward(combined))).reshape(-1)
    confidence = training.sigmoid(model.confidence_linear_2.forward(training.relu(model.confidence_linear_1.forward(combined))).reshape(-1))
    return alignment, confidence

class MoveEngine:
    def __init__(self, weights_path='model_weights.npz', config_path='model_config.json', vocab_path='principle_vocab.json', parameters=None):
        if parameters: PARAMETERS.update(parameters)
        PARAMETERS['weakness'] = {int(key): float(value) for key, value in PARAMETERS['weakness'].items()}
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        expected = {'spatial_planes': training.SPATIAL_PLANES, 'conv_channels': training.CONV_CHANNELS, 'alignment_hidden': training.ALIGNMENT_HIDDEN, 'confidence_hidden': training.CONFIDENCE_HIDDEN, 'embed_dim': training.PRINCIPLE_EMBED_DIM, 'numeric_dim': training.NUMERIC_DIM, 'board_size': BOARD_SIZE}
        for key, value in expected.items():
            if config.get(key) != value: raise ValueError(f'Model config {key}={config.get(key)} does not match train.py ({value})')
        self.principle_to_idx = json.loads(Path(vocab_path).read_text(encoding='utf-8'))
        self.evidence_features = bool(config.get('evidence_features', True))
        low, high = float(config.get('target_min', 0.0)), float(config.get('target_max', 1.0))
        self.target_middle = 0.5 * (low + high)
        self.target_half = max(0.5 * (high - low), 1e-06)
        self.model = training.PrincipleAlignmentModel(num_principles=config['num_principles'], numeric_dim=config['numeric_dim'], embed_dim=config['embed_dim'], rng=np.random.default_rng(0))
        weights = np.load(weights_path)
        entries = [name for _, _, name in self.model.parameters_and_gradients()] + [name for _, name in self.model.running_statistics()]
        missing = [name for name in entries if name not in weights.files]
        if missing: raise ValueError(f'Weights file is missing {missing[:3]}; retrain with the current train.py so batch-norm statistics are saved')
        for name in entries:
            owner, attribute = resolve(self.model, name)
            target = getattr(owner, attribute)
            if weights[name].shape != target.shape: raise ValueError(f'Weight shape mismatch for {name}: {weights[name].shape} vs {target.shape}')
            target[...] = weights[name]

    def ensure_board(self, board):
        if isinstance(board, str): return board_from_position(board)
        array = np.asarray(board)
        if array.shape != (BOARD_SIZE, BOARD_SIZE): raise ValueError(f'Board must be {BOARD_SIZE}x{BOARD_SIZE}')
        return array.astype(np.int8)

    def selected_principles(self, principle_codes):
        if principle_codes is None: return list(self.principle_to_idx.keys())
        unknown = [code for code in principle_codes if code not in self.principle_to_idx]
        if unknown: raise ValueError(f'Principles are not present in the vocabulary: {unknown}')
        return list(dict.fromkeys(principle_codes))

    def score_alignment(self, view, player, records, principles):
        letter = 'B' if player == BLACK else 'W'
        spatial = np.stack([training.extract_spatial_features(view, letter, mirror_point(record['point'])) for record in records])
        numeric = np.stack([training.extract_numeric_features(view, letter, mirror_point(record['point']), record['evidence']) for record in records])
        if not self.evidence_features: numeric[:, 5:17] = 0.0
        board_features = np.concatenate([trunk_features(self.model, spatial[start:start + 16]) for start in range(0, len(records), 16)], axis=0)
        indices = np.asarray([self.principle_to_idx[code] for code in principles], dtype=np.int64)
        count = len(principles)
        alignment, confidence = head_outputs(self.model, np.repeat(board_features, count, axis=0), np.tile(indices, len(records)), np.repeat(numeric, count, axis=0))
        return alignment.reshape(len(records), count), confidence.reshape(len(records), count)

    def choose_move(self, board, color, move_number=1, history=None, candidate_limit=24, openings=4, principle_codes=None):
        board = self.ensure_board(board)
        player = parse_color(color)
        history = parse_history(history)
        principles = self.selected_principles(principle_codes)
        context = context_vector(board, player, move_number)
        target, weights = targets_and_weights(context)
        intent = intent_signals(board, player, history)
        allowed = None
        if move_number >= PARAMETERS['pass_min_move'] and not any(len(liberties) <= 1 for color in (BLACK, WHITE) for _, liberties in all_groups(board, color)): allowed = contested_points(board)
        records = generate_candidates(board, player, history, candidate_limit, openings, float(context[5]), allowed)
        summary = {
            'principle_codes': principles,
            'yin_yang_dimensions': DIMENSIONS,
            'context': dict(zip(CONTEXT_NAMES, context.astype(float).tolist())),
            'target': target.astype(float).tolist(),
            'dimension_weights': weights.astype(float).tolist(),
            'intent': None if intent is None else {key: value for key, value in intent.items() if key not in {'apparent', 'hidden'}},
            'parameters': PARAMETERS,
            'warnings': WARNINGS + ([] if self.evidence_features else ['The model was trained without board_evidence: features 5-16 are zeroed to match training.']),
            'pass_reason': None
        }
        if not records: return {**summary, 'selected_move': 'pass', 'selected_move_human': 'pass', 'candidates': [], 'pass_reason': 'settled' if allowed is not None else 'no_legal_move'}
        for record in records: record['evidence'] = board_evidence(board, player, record['point'], move_number, record['captured'], record['after'])
        alignment, confidence = self.score_alignment(BoardView(board), player, records, principles)
        before = state_vector(board, player, initiative(history, player))
        uncertainty = 0.0 if intent is None else intent['uncertainty']
        coefficients = PARAMETERS['relevance_coefficients']
        relevance = float(np.clip(coefficients[0] + coefficients[1] * context[2] + coefficients[2] * (1.0 - context[3]) + coefficients[3] * uncertainty, 0.0, 1.0))
        y_weights = PARAMETERS['y_weights']
        afters = [state_vector(record['after'], player, initiative(history + [(player, record['point'])], player)) for record in records]
        progress = [static_progress(before, after, target, weights) for after in afters]
        centered = np.asarray([value for value, _ in progress], dtype=np.float32)
        centered -= np.median(centered)
        results = []
        for index, record in enumerate(records):
            row, column = record['point']
            after = afters[index]
            static = float(np.tanh(PARAMETERS['static_gain'] * centered[index]))
            pair, yang, yin = pair_score(board, record['after'], row, column, player)
            intent_score = 0.0 if intent is None else float((1.0 - intent['probability']) * intent['apparent'][row, column] + intent['probability'] * intent['hidden'][row, column])
            yin_yang = y_weights[0] * static + y_weights[1] * pair + y_weights[2] * intent_score
            alignment_raw = float(np.mean(alignment[index]))
            alignment_score = float(np.tanh((alignment_raw - self.target_middle) / self.target_half))
            results.append({
                'move_sgf': coordinate_sgf(record['point']),
                'move_human': coordinate_human(record['point']),
                'legal': True,
                'captured_count': len(record['captured']),
                'alignment_raw': alignment_raw,
                'alignment_score': alignment_score,
                'annotation_confidence': float(np.mean(confidence[index])),
                'yin_yang_score': float(yin_yang),
                'yin_yang_relevance': relevance,
                'yin_yang_contribution': float(relevance * yin_yang),
                'final_score': float(alignment_score + relevance * yin_yang),
                'components': {'static': static, 'pair': pair, 'intent': intent_score},
                'pair_detail': {'yang_saved': yang, 'yin_returned': yin},
                'per_principle': {code: {'alignment': float(alignment[index, number]), 'confidence': float(confidence[index, number])} for number, code in enumerate(principles)},
                'yin_yang': {'before': before.astype(float).tolist(), 'after': after.astype(float).tolist(), 'progress': progress[index][1].astype(float).tolist()}
            })
        results.sort(key=lambda value: value['final_score'], reverse=True)
        return {**summary, 'selected_move': results[0]['move_sgf'], 'selected_move_human': results[0]['move_human'], 'candidates': results}

def choose_move(board, color, weights_path='model_weights.npz', config_path='model_config.json', vocab_path='principle_vocab.json', **kwargs):
    return MoveEngine(weights_path, config_path, vocab_path).choose_move(board, color, **kwargs)

def load_sgf_state(path, target_move_number=None):
    game = sgf.Sgf_game.from_bytes(Path(path).read_bytes())
    if game.get_size() != BOARD_SIZE: raise ValueError(f'Expected a {BOARD_SIZE}x{BOARD_SIZE} SGF but found {game.get_size()}x{game.get_size()}')
    board = boards.Board(BOARD_SIZE)
    history = []
    for node in game.get_main_sequence():
        setup_black, setup_white, setup_empty = node.get_setup_stones()
        if setup_black or setup_white or setup_empty: board.apply_setup(setup_black, setup_white, setup_empty)
        color, point = node.get_move()
        if color is None: continue
        if target_move_number is not None and len(history) + 1 == int(target_move_number): return {'board': array_from_sgfmill(board), 'color': 'B' if color == 'b' else 'W', 'move_number': len(history) + 1, 'history': history}
        if point is not None:
            try: board.play(point[0], point[1], color)
            except ValueError: pass
        history.append((BLACK if color == 'b' else WHITE, mirror_point(point)))
    if not history: raise ValueError(f'SGF contains no playable moves: {path}')
    if target_move_number is not None: raise ValueError(f'SGF has no main-line move number {target_move_number}')
    return {'board': array_from_sgfmill(board), 'color': 'W' if history[-1][0] == BLACK else 'B', 'move_number': len(history) + 1, 'history': history}

def compare_evidence(recorded, computed):
    outcome = {}
    for key, value in computed.items():
        if key not in recorded: continue
        other = recorded[key]
        try:
            if isinstance(value, dict): outcome[key] = isinstance(other, dict) and all(float(other.get(inner, -1.0)) == float(inner_value) for inner, inner_value in value.items())
            elif isinstance(value, bool): outcome[key] = bool(other) == value
            elif isinstance(value, (int, float)): outcome[key] = abs(float(other) - float(value)) < 1e-06
            else: outcome[key] = other == value
        except (TypeError, ValueError): outcome[key] = False
    return outcome

def verify_evidence(c_models, sgf_dir):
    games = training.replay_sgf_games(sgf_dir)
    rows = training.load_and_merge_c_models(c_models)
    totals, matches, examples, seen, unchecked = {}, {}, {}, set(), set()
    for row in rows:
        if not row.get('board_evidence'): continue
        game_id = training.canonical_game_id(row['game_id'])
        key = (game_id, int(row['move_number']))
        if key in seen or game_id not in games or not 1 <= key[1] <= len(games[game_id]): continue
        seen.add(key)
        try: recorded = json.loads(row['board_evidence'])
        except ValueError: continue
        move = games[game_id][key[1] - 1]
        if move['point'] is None: continue
        board = array_from_sgfmill(move['before'])
        point = mirror_point(move['point'])
        player = BLACK if move['color'] == 'B' else WHITE
        result = play_stone(board, point[0], point[1], player)
        if result is None: continue
        computed = board_evidence(board, player, point, key[1], result[1], result[0])
        unchecked |= set(recorded) - set(computed)
        for name, same in compare_evidence(recorded, computed).items():
            totals[name] = totals.get(name, 0) + 1
            matches[name] = matches.get(name, 0) + int(same)
            if not same and len(examples.setdefault(name, [])) < 3: examples[name].append({'game': game_id, 'move': key[1], 'recorded': recorded[name], 'computed': computed[name]})
    report = {'positions_checked': len(seen), 'keys': {name: {'matched': matches[name], 'total': totals[name], 'mismatch_examples': examples.get(name, [])} for name in totals}, 'recorded_keys_not_computed': sorted(unchecked)}
    print(json.dumps(report, indent=2))
    return report

def verify_features(c_models, sgf_dir):
    games = training.replay_sgf_games(sgf_dir)
    data = training.prepare_and_validate_dataset(training.load_and_merge_c_models(c_models), games)
    seen, spatial_bad, numeric_bad, numeric_gap, examples = set(), {}, {}, {}, []
    for row in data:
        key = (row['game_id'], row['move_number'])
        if key in seen: continue
        seen.add(key)
        record = games[row['game_id']][row['move_number'] - 1]
        if record['point'] is None: continue
        board = array_from_sgfmill(record['before'])
        point = mirror_point(record['point'])
        player = BLACK if record['color'] == 'B' else WHITE
        result = play_stone(board, point[0], point[1], player)
        if result is None: continue
        view, letter = BoardView(board), record['color']
        spatial = training.extract_spatial_features(view, letter, mirror_point(point))
        numeric = training.extract_numeric_features(view, letter, mirror_point(point), board_evidence(board, player, point, row['move_number'], result[1], result[0]))
        for plane in range(spatial.shape[0]):
            if not np.array_equal(spatial[plane], row['spatial_representation'][plane]): spatial_bad[plane] = spatial_bad.get(plane, 0) + 1
        for index in range(numeric.shape[0]):
            gap = abs(float(numeric[index]) - float(row['numeric_representation'][index]))
            if gap > 1e-06:
                numeric_bad[index] = numeric_bad.get(index, 0) + 1
                numeric_gap[index] = max(numeric_gap.get(index, 0.0), gap)
                if len(examples) < 5 and index not in {entry['index'] for entry in examples}: examples.append({'index': index, 'game': row['game_id'], 'move': row['move_number'], 'training': float(row['numeric_representation'][index]), 'inference': float(numeric[index])})
    report = {'positions_checked': len(seen), 'identical': not spatial_bad and not numeric_bad, 'spatial_planes_differing': spatial_bad, 'numeric_indices_differing': {str(index): {'positions': count, 'max_abs_difference': numeric_gap[index]} for index, count in sorted(numeric_bad.items())}, 'examples': examples}
    print(json.dumps(report, indent=2))
    return report

def resolve_input(arguments):
    supplied = [arguments.position is not None, arguments.position_file is not None, arguments.sgf_file is not None]
    if sum(supplied) != 1: raise ValueError('Exactly one of --position, --position-file, or --sgf-file is required')
    if arguments.sgf_file is not None:
        state = load_sgf_state(arguments.sgf_file, arguments.sgf_move_number)
        if arguments.color is not None: state['color'] = arguments.color.upper()
        if arguments.move_number is not None: state['move_number'] = arguments.move_number
        return state
    if arguments.color is None: raise ValueError('--color is required with --position or --position-file')
    position = arguments.position if arguments.position is not None else Path(arguments.position_file).read_text(encoding='utf-8')
    return {'board': board_from_position(position.strip()), 'color': arguments.color.upper(), 'move_number': arguments.move_number or 1, 'history': []}

def main():
    parser = argparse.ArgumentParser(description='Generate and rank legal Go moves with the numpy alignment model and the Yin-Yang evaluation.')
    parser.add_argument('--weights', default='model_weights.npz')
    parser.add_argument('--config', default='model_config.json')
    parser.add_argument('--vocab', default='principle_vocab.json')
    parser.add_argument('--parameters', default=None)
    parser.add_argument('--position', default=None)
    parser.add_argument('--position-file', default=None)
    parser.add_argument('--sgf-file', default=None)
    parser.add_argument('--sgf-move-number', type=int, default=None)
    parser.add_argument('--color', choices=['B', 'W', 'b', 'w'], default=None)
    parser.add_argument('--move-number', type=int, default=None)
    parser.add_argument('--candidate-limit', type=int, default=24)
    parser.add_argument('--openings', type=int, default=4)
    parser.add_argument('--principles', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--verify-evidence', action='store_true')
    parser.add_argument('--verify-features', action='store_true')
    parser.add_argument('--c-models', nargs='+', default=['CSV/principles_followed.csv', 'CSV/principles_violated.csv'])
    parser.add_argument('--sgf-dir', default='SGF')
    arguments = parser.parse_args()
    if arguments.verify_evidence:
        verify_evidence(arguments.c_models, arguments.sgf_dir)
        return
    if arguments.verify_features:
        verify_features(arguments.c_models, arguments.sgf_dir)
        return
    has_input = any(value is not None for value in [arguments.position, arguments.position_file, arguments.sgf_file])
    if arguments.demo and has_input: parser.error('--demo cannot be combined with --position, --position-file, or --sgf-file')
    if not arguments.demo and not has_input:
        parser.print_help()
        print('\nProvide --position, --position-file, or --sgf-file, or run with --demo for an empty-board test.')
        return
    if arguments.demo: state = {'board': np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8), 'color': 'B', 'move_number': 1, 'history': []}
    else:
        try: state = resolve_input(arguments)
        except ValueError as error: parser.error(str(error))
    parameters = json.loads(Path(arguments.parameters).read_text(encoding='utf-8')) if arguments.parameters else None
    principles = [value.strip() for value in arguments.principles.split(',') if value.strip()] if arguments.principles else None
    engine = MoveEngine(arguments.weights, arguments.config, arguments.vocab, parameters)
    result = engine.choose_move(state['board'], state['color'], move_number=state['move_number'], history=[(color, coordinate_sgf(point)) for color, point in state['history']], candidate_limit=arguments.candidate_limit, openings=arguments.openings, principle_codes=principles)
    result['input'] = {'move_number': state['move_number'], 'color': state['color'], 'history_plies': len(state['history'])}
    text = json.dumps(result, indent=2)
    if arguments.output: Path(arguments.output).write_text(text, encoding='utf-8')
    print(text)

if __name__ == '__main__':
    main()