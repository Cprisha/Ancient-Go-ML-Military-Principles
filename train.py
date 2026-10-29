import argparse
import ast
import csv
import json
import random
from pathlib import Path

import numpy as np
from sgfmill import boards, sgf

BOARD_SIZE = 19
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
SPATIAL_PLANES = 7
PRINCIPLE_EMBED_DIM = 32
NUMERIC_DIM = 31
CONV_CHANNELS = 64
ALIGNMENT_HIDDEN = 96
CONFIDENCE_HIDDEN = 64
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

def canonical_game_id(value): return Path(str(value)).stem.strip().lower()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def coordinate_to_point(coordinate):
    coordinate = str(coordinate).strip().lower()
    if coordinate in {'', 'pass', 'none'}: return None
    if len(coordinate) != 2: raise ValueError(f'Invalid SGF coordinate: {coordinate}')
    column = ord(coordinate[0]) - ord('a')
    row = ord(coordinate[1]) - ord('a')
    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE): raise ValueError(f'Coordinate outside board: {coordinate}')
    return row, column

def extract_spatial_features(board_before, move_color, point):
    planes = np.zeros((SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    me = move_color.lower()
    opp = 'w' if me == 'b' else 'b'
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board_before.get(r, c)
            if stone == me: planes[0, r, c] = 1.0
            elif stone == opp: planes[1, r, c] = 1.0
            else: planes[2, r, c] = 1.0
    if point is not None:
        pr, pc = point
        planes[3, pr, pc] = 1.0
        for dr, dc in DIRECTIONS:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                planes[4, nr, nc] = 1.0
                adj_stone = board_before.get(nr, nc)
                if adj_stone == opp: planes[5, nr, nc] = 1.0
                elif adj_stone == me: planes[6, nr, nc] = 1.0
    return planes

def extract_numeric_features(board_before, move_color, point, evidence_dict):
    vector = np.zeros(NUMERIC_DIM, dtype=np.float32)
    if point is not None:
        pr, pc = point
        vector[0] = pr / 18.0
        vector[1] = pc / 18.0
        vector[2] = min(pr, 18 - pr) / 9.0
        vector[3] = min(pc, 18 - pc) / 9.0
        dist_center = abs(pr - 9) + abs(pc - 9)
        vector[4] = dist_center / 18.0
    if isinstance(evidence_dict, dict):
        vector[5] = float(evidence_dict.get('friendly_neighbour_count', 0)) / 4.0
        vector[6] = float(evidence_dict.get('enemy_neighbour_count', 0)) / 4.0
        vector[7] = float(evidence_dict.get('nearby_friendly_count', 0)) / 8.0
        vector[8] = float(evidence_dict.get('nearby_enemy_count', 0)) / 8.0
        vector[9] = 1.0 if evidence_dict.get('near_empty', False) else 0.0
        vector[10] = float(evidence_dict.get('captured_count', 0)) / 10.0
        own_group = evidence_dict.get('own_group_after', {})
        if isinstance(own_group, dict):
            vector[11] = float(own_group.get('size', 0)) / 20.0
            vector[12] = float(own_group.get('liberties', 0)) / 10.0
        vector[13] = float(evidence_dict.get('move_number', 0)) / 200.0
        zone = evidence_dict.get('zone', '')
        if zone == 'corner': vector[14] = 1.0
        elif zone == 'side': vector[15] = 1.0
        elif zone == 'center': vector[16] = 1.0
    return vector

def load_and_merge_c_models(csv_paths):
    combined_rows = []
    for path_str in csv_paths:
        path = Path(path_str)
        if not path.exists(): 
            continue
        with open(path, 'r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if 'target_alignment' not in row or not row['target_alignment']:
                    if 'followed' in path.name.lower(): row['target_alignment'] = '1.0'
                    elif 'violated' in path.name.lower(): row['target_alignment'] = '0.0'
                    else: row['target_alignment'] = '1.0'
                if 'confidence_target' not in row or not row['confidence_target']:
                    row['confidence_target'] = row.get('confidence') or '1.0'
                if 'label_mask' not in row or not row['label_mask']: row['label_mask'] = '1.0'
                if 'label_weight' not in row or not row['label_weight']: row['label_weight'] = '1.0'
                combined_rows.append(row)
    if not combined_rows: raise ValueError('No valid rows were loaded from the provided CSV files.')
    return combined_rows

def replay_sgf_games(sgf_dir):
    sgf_dir = Path(sgf_dir)
    paths = sorted(sgf_dir.glob('*.sgf'))
    if not paths: raise FileNotFoundError(f'No SGF files were found in {sgf_dir}')
    games = {}
    for path in paths:
        game = sgf.Sgf_game.from_bytes(path.read_bytes())
        if game.get_size() != BOARD_SIZE: raise ValueError(f'Expected a 19x19 board in {path.name}')
        board = boards.Board(BOARD_SIZE)
        sequence = game.get_main_sequence()
        moves = []
        for node in sequence:
            setup_black, setup_white, setup_empty = node.get_setup_stones()
            if setup_black or setup_white or setup_empty: board.apply_setup(setup_black, setup_white, setup_empty)
            color, point = node.get_move()
            if color is None: continue
            before = board.copy()
            legal = True
            if point is not None:
                try: board.play(point[0], point[1], color)
                except ValueError: legal = False
            moves.append({'move_number': len(moves) + 1, 'color': color.upper(), 'point': point, 'legal': legal, 'before': before, 'after': board.copy()})
        games[canonical_game_id(path.name)] = moves
    return games

def prepare_and_validate_dataset(frame, games):
    required = {'sgf_file', 'game_id', 'move_number', 'color', 'principle', 'target_alignment', 'confidence_target', 'label_mask', 'label_weight'}
    if not frame: raise ValueError('Dataset contains no rows')
    missing = required.difference(frame[0].keys())
    if missing: raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')
    known_games = set(games)
    processed = []
    for row in frame:
        game_id = canonical_game_id(row['game_id'])
        if game_id not in known_games: raise ValueError(f'Dataset contains game without SGF file: {game_id}')
        try: move_number = int(row['move_number'])
        except (TypeError, ValueError): raise ValueError('Dataset contains invalid move numbers')
        sgf_moves = games[game_id]
        if move_number < 1 or move_number > len(sgf_moves): raise ValueError(f'Move {move_number} is outside SGF game {game_id}')
        sgf_move = sgf_moves[move_number - 1]
        if row['color'].upper() != sgf_move['color'].upper(): raise ValueError(f'Color mismatch for {game_id} move {move_number}')
        point = sgf_move['point']
        board_before = sgf_move['before']
        evidence = {}
        if 'board_evidence' in row and row['board_evidence']:
            try: evidence = json.loads(row['board_evidence'])
            except Exception: pass
        if 'spatial_representation' in row and row['spatial_representation']: spatial = np.asarray(ast.literal_eval(row['spatial_representation']), dtype=np.float32)
        else: spatial = extract_spatial_features(board_before, row['color'], point)
        if 'numeric_representation' in row and row['numeric_representation']: numeric = np.asarray(ast.literal_eval(row['numeric_representation']), dtype=np.float32)
        else: numeric = extract_numeric_features(board_before, row['color'], point, evidence)
        item = dict(row)
        item['game_id'] = game_id
        item['move_number'] = move_number
        item['spatial_representation'] = spatial
        item['numeric_representation'] = numeric
        item['target_alignment'] = float(row['target_alignment'])
        item['confidence_target'] = float(row['confidence_target'])
        item['label_mask'] = float(row['label_mask'])
        item['label_weight'] = float(row['label_weight'])
        processed.append(item)
    return processed

def chronological_game_split(frame):
    game_order = list(dict.fromkeys((row['game_id'] for row in frame)))
    split_index = int(len(game_order) * 0.75)
    if split_index <= 0 or split_index >= len(game_order): raise ValueError('At least two games are required for a 75/25 game split')
    training_games = set(game_order[:split_index])
    validation_games = set(game_order[split_index:])
    training = [row for row in frame if row['game_id'] in training_games]
    validation = [row for row in frame if row['game_id'] in validation_games]
    return training, validation

def build_principle_vocabulary(frame):
    principles = list(dict.fromkeys((str(row['principle']) for row in frame)))
    if not principles: raise ValueError('No principles found in training data')
    return {principle: index for index, principle in enumerate(principles)}

def attach_principle_indices(frame, principle_to_idx):
    result = []
    for row in frame:
        principle = str(row['principle'])
        if principle not in principle_to_idx: continue
        item = dict(row)
        item['principle_idx'] = principle_to_idx[principle]
        result.append(item)
    return result

def iter_epoch_batches(frame, batch_size, rng):
    order = rng.permutation(len(frame))
    for start in range(0, len(frame), batch_size): yield [frame[index] for index in order[start:start + batch_size]]

def batch_arrays(frame):
    spatial = np.stack([row['spatial_representation'] for row in frame], axis=0)
    numeric = np.stack([row['numeric_representation'] for row in frame], axis=0)
    principle = np.asarray([row['principle_idx'] for row in frame], dtype=np.int64)
    target = np.asarray([row['target_alignment'] for row in frame], dtype=np.float32)
    confidence_target = np.asarray([row['confidence_target'] for row in frame], dtype=np.float32)
    mask = np.asarray([row['label_mask'] for row in frame], dtype=np.float32)
    weight = np.asarray([row['label_weight'] for row in frame], dtype=np.float32)
    return spatial, numeric, principle, target, confidence_target, mask, weight

def smooth_l1(value):
    absolute = np.abs(value)
    return np.where(absolute < 1.0, 0.5 * value * value, absolute - 0.5)

def smooth_l1_gradient(value):
    absolute = np.abs(value)
    return np.where(absolute < 1.0, value, np.sign(value))

def weighted_smooth_l1(prediction, target, mask, weight):
    effective_weight = mask * weight
    denominator = max(float(np.sum(effective_weight)), 1e-08)
    raw = smooth_l1(prediction - target)
    return float(np.sum(raw * effective_weight) / denominator)

def weighted_smooth_l1_gradient(prediction, target, mask, weight):
    effective_weight = mask * weight
    denominator = max(float(np.sum(effective_weight)), 1e-08)
    return smooth_l1_gradient(prediction - target) * effective_weight / denominator

def training_loss(outputs, target, confidence_target, mask, weight, confidence_loss_weight):
    alignment_loss = weighted_smooth_l1(outputs['alignment'], target, mask, weight)
    confidence_loss = weighted_smooth_l1(outputs['confidence'], confidence_target, mask, weight)
    return alignment_loss + confidence_loss_weight * confidence_loss

def relu(x): return np.maximum(x, 0.0)

def relu_gradient(x): return (x > 0.0).astype(np.float32)

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def im2col(x):
    batch, channels, height, width = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    rows = np.empty((batch, height, width, channels, 3, 3), dtype=x.dtype)
    for row in range(height):
        for column in range(width): rows[:, row, column] = padded[:, :, row:row + 3, column:column + 3]
    return rows.reshape(batch * height * width, channels * 9)

def col2im(columns, shape):
    batch, channels, height, width = shape
    columns = columns.reshape(batch, height, width, channels, 3, 3)
    padded = np.zeros((batch, channels, height + 2, width + 2), dtype=columns.dtype)
    for row in range(height):
        for column in range(width): padded[:, :, row:row + 3, column:column + 3] += columns[:, row, column]
    return padded[:, :, 1:-1, 1:-1]

def conv_forward(x, weight, bias):
    batch, channels, height, width = x.shape
    output_channels = weight.shape[0]
    columns = im2col(x)
    flattened_weight = weight.reshape(output_channels, -1)
    output = columns @ flattened_weight.T + bias
    output = output.reshape(batch, height, width, output_channels).transpose(0, 3, 1, 2)
    cache = x, columns
    return output, cache

def conv_backward(gradient, cache, weight):
    x, columns = cache
    batch, channels, height, width = x.shape
    gradient_columns = gradient.transpose(0, 2, 3, 1).reshape(batch * height * width, weight.shape[0])
    gradient_weight = (gradient_columns.T @ columns).reshape(weight.shape)
    gradient_bias = gradient_columns.sum(axis=0)
    gradient_columns_input = gradient_columns @ weight.reshape(weight.shape[0], -1)
    gradient_input = col2im(gradient_columns_input, x.shape)
    return gradient_input, gradient_weight, gradient_bias

def batch_norm_forward(x, gamma, beta, training, running_mean, running_var, momentum=0.1, epsilon=1e-05):
    if training:
        mean = x.mean(axis=(0, 2, 3))
        variance = x.var(axis=(0, 2, 3))
        running_mean *= 1.0 - momentum
        running_mean += momentum * mean
        running_var *= 1.0 - momentum
        running_var += momentum * variance
    else:
        mean = running_mean
        variance = running_var
    centered = x - mean[None, :, None, None]
    inverse_std = 1.0 / np.sqrt(variance + epsilon)
    normalized = centered * inverse_std[None, :, None, None]
    output = gamma[None, :, None, None] * normalized + beta[None, :, None, None]
    cache = x, normalized, mean, variance, inverse_std, gamma
    return output, cache

def batch_norm_backward(gradient, cache):
    x, normalized, mean, variance, inverse_std, gamma = cache
    count = x.shape[0] * x.shape[2] * x.shape[3]
    gradient_gamma = np.sum(gradient * normalized, axis=(0, 2, 3))
    gradient_beta = np.sum(gradient, axis=(0, 2, 3))
    gradient_normalized = gradient * gamma[None, :, None, None]
    gradient_variance = np.sum(gradient_normalized * (x - mean[None, :, None, None]) * -0.5 * inverse_std[None, :, None, None] ** 3, axis=(0, 2, 3))
    gradient_mean = np.sum(gradient_normalized * -inverse_std[None, :, None, None], axis=(0, 2, 3))
    gradient_input = gradient_normalized * inverse_std[None, :, None, None] + gradient_variance[None, :, None, None] * 2.0 * (x - mean[None, :, None, None]) / count + gradient_mean[None, :, None, None] / count
    return gradient_input, gradient_gamma, gradient_beta

class Dropout:
    def __init__(self, probability, rng):
        self.probability = probability
        self.rng = rng
        self.mask = None

    def forward(self, x, training):
        if not training or self.probability == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.probability
        self.mask = (self.rng.random(x.shape) < keep).astype(np.float32) / keep
        return x * self.mask

    def backward(self, gradient):
        if self.mask is None: return gradient
        return gradient * self.mask

class Linear:
    def __init__(self, input_dim, output_dim, rng):
        scale = np.float32(np.sqrt(2.0 / input_dim))
        self.weight = rng.standard_normal((output_dim, input_dim)).astype(np.float32) * scale
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.cache = None
        self.gradient_weight = np.zeros_like(self.weight)
        self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.cache = x
        return x @ self.weight.T + self.bias

    def backward(self, gradient):
        x = self.cache
        self.gradient_weight = gradient.T @ x
        self.gradient_bias = gradient.sum(axis=0)
        return gradient @ self.weight

class ResidualBlock:
    def __init__(self, channels, rng):
        self.conv1_weight = rng.standard_normal((channels, channels, 3, 3)).astype(np.float32) * np.float32(np.sqrt(2.0 / (channels * 9)))
        self.conv1_bias = np.zeros(channels, dtype=np.float32)
        self.bn1_gamma = np.ones(channels, dtype=np.float32)
        self.bn1_beta = np.zeros(channels, dtype=np.float32)
        self.bn1_running_mean = np.zeros(channels, dtype=np.float32)
        self.bn1_running_var = np.ones(channels, dtype=np.float32)
        self.conv2_weight = rng.standard_normal((channels, channels, 3, 3)).astype(np.float32) * np.float32(np.sqrt(2.0 / (channels * 9)))
        self.conv2_bias = np.zeros(channels, dtype=np.float32)
        self.bn2_gamma = np.ones(channels, dtype=np.float32)
        self.bn2_beta = np.zeros(channels, dtype=np.float32)
        self.bn2_running_mean = np.zeros(channels, dtype=np.float32)
        self.bn2_running_var = np.ones(channels, dtype=np.float32)
        self.gradient_conv1_weight = np.zeros_like(self.conv1_weight)
        self.gradient_conv1_bias = np.zeros_like(self.conv1_bias)
        self.gradient_bn1_gamma = np.zeros_like(self.bn1_gamma)
        self.gradient_bn1_beta = np.zeros_like(self.bn1_beta)
        self.gradient_conv2_weight = np.zeros_like(self.conv2_weight)
        self.gradient_conv2_bias = np.zeros_like(self.conv2_bias)
        self.gradient_bn2_gamma = np.zeros_like(self.bn2_gamma)
        self.gradient_bn2_beta = np.zeros_like(self.bn2_beta)
        self.cache = None

    def forward(self, x, training):
        conv1, conv1_cache = conv_forward(x, self.conv1_weight, self.conv1_bias)
        bn1, bn1_cache = batch_norm_forward(conv1, self.bn1_gamma, self.bn1_beta, training, self.bn1_running_mean, self.bn1_running_var)
        relu1 = relu(bn1)
        conv2, conv2_cache = conv_forward(relu1, self.conv2_weight, self.conv2_bias)
        bn2, bn2_cache = batch_norm_forward(conv2, self.bn2_gamma, self.bn2_beta, training, self.bn2_running_mean, self.bn2_running_var)
        added = bn2 + x
        output = relu(added)
        self.cache = x, bn1, added, conv1_cache, bn1_cache, conv2_cache, bn2_cache
        return output

    def backward(self, gradient):
        x, bn1, added, conv1_cache, bn1_cache, conv2_cache, bn2_cache = self.cache
        gradient_added = gradient * relu_gradient(added)
        gradient_residual = gradient_added
        gradient_bn2 = gradient_added
        gradient_conv2, gradient_bn2_gamma, gradient_bn2_beta = batch_norm_backward(gradient_bn2, bn2_cache)
        gradient_relu1, gradient_conv2_weight, gradient_conv2_bias = conv_backward(gradient_conv2, conv2_cache, self.conv2_weight)
        gradient_bn1 = gradient_relu1 * relu_gradient(bn1)
        gradient_conv1, gradient_bn1_gamma, gradient_bn1_beta = batch_norm_backward(gradient_bn1, bn1_cache)
        gradient_input, gradient_conv1_weight, gradient_conv1_bias = conv_backward(gradient_conv1, conv1_cache, self.conv1_weight)
        gradient_input += gradient_residual
        self.gradient_conv1_weight = gradient_conv1_weight
        self.gradient_conv1_bias = gradient_conv1_bias
        self.gradient_bn1_gamma = gradient_bn1_gamma
        self.gradient_bn1_beta = gradient_bn1_beta
        self.gradient_conv2_weight = gradient_conv2_weight
        self.gradient_conv2_bias = gradient_conv2_bias
        self.gradient_bn2_gamma = gradient_bn2_gamma
        self.gradient_bn2_beta = gradient_bn2_beta
        return gradient_input

class PrincipleAlignmentModel:
    def __init__(self, num_principles, numeric_dim=NUMERIC_DIM, embed_dim=PRINCIPLE_EMBED_DIM, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.principle_embedding = self.rng.standard_normal((num_principles, embed_dim)).astype(np.float32) * 0.02
        self.conv_in_weight = self.rng.standard_normal((CONV_CHANNELS, SPATIAL_PLANES, 3, 3)).astype(np.float32) * np.float32(np.sqrt(2.0 / (SPATIAL_PLANES * 9)))
        self.conv_in_bias = np.zeros(CONV_CHANNELS, dtype=np.float32)
        self.norm_in_gamma = np.ones(CONV_CHANNELS, dtype=np.float32)
        self.norm_in_beta = np.zeros(CONV_CHANNELS, dtype=np.float32)
        self.norm_in_running_mean = np.zeros(CONV_CHANNELS, dtype=np.float32)
        self.norm_in_running_var = np.ones(CONV_CHANNELS, dtype=np.float32)
        self.residual_1 = ResidualBlock(CONV_CHANNELS, self.rng)
        self.residual_2 = ResidualBlock(CONV_CHANNELS, self.rng)
        combined_dim = CONV_CHANNELS + embed_dim + numeric_dim
        self.alignment_linear_1 = Linear(combined_dim, ALIGNMENT_HIDDEN, self.rng)
        self.alignment_linear_2 = Linear(ALIGNMENT_HIDDEN, 1, self.rng)
        self.confidence_linear_1 = Linear(combined_dim, CONFIDENCE_HIDDEN, self.rng)
        self.confidence_linear_2 = Linear(CONFIDENCE_HIDDEN, 1, self.rng)
        self.alignment_dropout = Dropout(0.2, self.rng)
        self.confidence_dropout = Dropout(0.1, self.rng)
        self.gradient_principle_embedding = np.zeros_like(self.principle_embedding)
        self.gradient_conv_in_weight = np.zeros_like(self.conv_in_weight)
        self.gradient_conv_in_bias = np.zeros_like(self.conv_in_bias)
        self.gradient_norm_in_gamma = np.zeros_like(self.norm_in_gamma)
        self.gradient_norm_in_beta = np.zeros_like(self.norm_in_beta)
        self.cache = None

    def forward(self, board, numeric, principle, training=True):
        conv_in, conv_in_cache = conv_forward(board, self.conv_in_weight, self.conv_in_bias)
        norm_in, norm_in_cache = batch_norm_forward(conv_in, self.norm_in_gamma, self.norm_in_beta, training, self.norm_in_running_mean, self.norm_in_running_var)
        relu_in = relu(norm_in)
        residual_1 = self.residual_1.forward(relu_in, training)
        residual_2 = self.residual_2.forward(residual_1, training)
        board_features = residual_2.mean(axis=(2, 3))
        principle_features = self.principle_embedding[principle]
        combined = np.concatenate([board_features, principle_features, numeric], axis=1)
        alignment_hidden_pre = self.alignment_linear_1.forward(combined)
        alignment_hidden = relu(alignment_hidden_pre)
        alignment_dropout = self.alignment_dropout.forward(alignment_hidden, training)
        alignment = self.alignment_linear_2.forward(alignment_dropout).reshape(-1)
        confidence_hidden_pre = self.confidence_linear_1.forward(combined)
        confidence_hidden = relu(confidence_hidden_pre)
        confidence_dropout = self.confidence_dropout.forward(confidence_hidden, training)
        confidence_logits = self.confidence_linear_2.forward(confidence_dropout).reshape(-1)
        confidence = sigmoid(confidence_logits)
        self.cache = board, norm_in, residual_2, principle, numeric, alignment_hidden_pre, confidence_hidden_pre, confidence_logits, conv_in_cache, norm_in_cache
        return {'alignment': alignment, 'confidence': confidence}

    def backward(self, alignment_gradient, confidence_gradient):
        board, norm_in, residual_2, principle, numeric, alignment_hidden_pre, confidence_hidden_pre, confidence_logits, conv_in_cache, norm_in_cache = self.cache
        
        alignment_gradient = alignment_gradient.reshape(-1, 1)
        gradient_alignment_hidden = self.alignment_linear_2.backward(alignment_gradient)
        gradient_alignment_hidden = self.alignment_dropout.backward(gradient_alignment_hidden)
        gradient_alignment_hidden *= relu_gradient(alignment_hidden_pre)
        gradient_combined_alignment = self.alignment_linear_1.backward(gradient_alignment_hidden)
        
        confidence_gradient = confidence_gradient.reshape(-1, 1)
        confidence_sig = sigmoid(confidence_logits.reshape(-1, 1))
        confidence_gradient *= confidence_sig * (1.0 - confidence_sig)
        gradient_confidence_hidden = self.confidence_linear_2.backward(confidence_gradient)
        gradient_confidence_hidden = self.confidence_dropout.backward(gradient_confidence_hidden)
        gradient_confidence_hidden *= relu_gradient(confidence_hidden_pre)
        gradient_combined_confidence = self.confidence_linear_1.backward(gradient_confidence_hidden)
        gradient_combined = gradient_combined_alignment + gradient_combined_confidence
        board_gradient = gradient_combined[:, :CONV_CHANNELS]
        principle_gradient = gradient_combined[:, CONV_CHANNELS:CONV_CHANNELS + PRINCIPLE_EMBED_DIM]
        numeric_gradient = gradient_combined[:, CONV_CHANNELS + PRINCIPLE_EMBED_DIM:]
        gradient_board_features = board_gradient / (BOARD_SIZE * BOARD_SIZE)
        gradient_residual_2 = gradient_board_features[:, :, None, None] * np.ones((1, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        gradient_residual_1 = self.residual_2.backward(gradient_residual_2)
        gradient_relu_in = self.residual_1.backward(gradient_residual_1)
        gradient_norm_in = gradient_relu_in * relu_gradient(norm_in)
        gradient_conv_in, gradient_norm_in_gamma, gradient_norm_in_beta = batch_norm_backward(gradient_norm_in, norm_in_cache)
        gradient_board, gradient_conv_in_weight, gradient_conv_in_bias = conv_backward(gradient_conv_in, conv_in_cache, self.conv_in_weight)
        self.gradient_principle_embedding.fill(0.0)
        np.add.at(self.gradient_principle_embedding, principle, principle_gradient)
        self.gradient_conv_in_weight = gradient_conv_in_weight
        self.gradient_conv_in_bias = gradient_conv_in_bias
        self.gradient_norm_in_gamma = gradient_norm_in_gamma
        self.gradient_norm_in_beta = gradient_norm_in_beta
        return numeric_gradient

    def parameters_and_gradients(self):
        return [
            (self.principle_embedding, self.gradient_principle_embedding, 'principle_embedding'),
            (self.conv_in_weight, self.gradient_conv_in_weight, 'conv_in_weight'),
            (self.conv_in_bias, self.gradient_conv_in_bias, 'conv_in_bias'),
            (self.norm_in_gamma, self.gradient_norm_in_gamma, 'norm_in_gamma'),
            (self.norm_in_beta, self.gradient_norm_in_beta, 'norm_in_beta'),
            (self.residual_1.conv1_weight, self.residual_1.gradient_conv1_weight, 'residual_1.conv1_weight'),
            (self.residual_1.conv1_bias, self.residual_1.gradient_conv1_bias, 'residual_1.conv1_bias'),
            (self.residual_1.bn1_gamma, self.residual_1.gradient_bn1_gamma, 'residual_1.bn1_gamma'),
            (self.residual_1.bn1_beta, self.residual_1.gradient_bn1_beta, 'residual_1.bn1_beta'),
            (self.residual_1.conv2_weight, self.residual_1.gradient_conv2_weight, 'residual_1.conv2_weight'),
            (self.residual_1.conv2_bias, self.residual_1.gradient_conv2_bias, 'residual_1.conv2_bias'),
            (self.residual_1.bn2_gamma, self.residual_1.gradient_bn2_gamma, 'residual_1.bn2_gamma'),
            (self.residual_1.bn2_beta, self.residual_1.gradient_bn2_beta, 'residual_1.bn2_beta'),
            (self.residual_2.conv1_weight, self.residual_2.gradient_conv1_weight, 'residual_2.conv1_weight'),
            (self.residual_2.conv1_bias, self.residual_2.gradient_conv1_bias, 'residual_2.conv1_bias'),
            (self.residual_2.bn1_gamma, self.residual_2.gradient_bn1_gamma, 'residual_2.bn1_gamma'),
            (self.residual_2.bn1_beta, self.residual_2.gradient_bn1_beta, 'residual_2.bn1_beta'),
            (self.residual_2.conv2_weight, self.residual_2.gradient_conv2_weight, 'residual_2.conv2_weight'),
            (self.residual_2.conv2_bias, self.residual_2.gradient_conv2_bias, 'residual_2.conv2_bias'),
            (self.residual_2.bn2_gamma, self.residual_2.gradient_bn2_gamma, 'residual_2.bn2_gamma'),
            (self.residual_2.bn2_beta, self.residual_2.gradient_bn2_beta, 'residual_2.bn2_beta'),
            (self.alignment_linear_1.weight, self.alignment_linear_1.gradient_weight, 'alignment_linear_1.weight'),
            (self.alignment_linear_1.bias, self.alignment_linear_1.gradient_bias, 'alignment_linear_1.bias'),
            (self.alignment_linear_2.weight, self.alignment_linear_2.gradient_weight, 'alignment_linear_2.weight'),
            (self.alignment_linear_2.bias, self.alignment_linear_2.gradient_bias, 'alignment_linear_2.bias'),
            (self.confidence_linear_1.weight, self.confidence_linear_1.gradient_weight, 'confidence_linear_1.weight'),
            (self.confidence_linear_1.bias, self.confidence_linear_1.gradient_bias, 'confidence_linear_1.bias'),
            (self.confidence_linear_2.weight, self.confidence_linear_2.gradient_weight, 'confidence_linear_2.weight'),
            (self.confidence_linear_2.bias, self.confidence_linear_2.gradient_bias, 'confidence_linear_2.bias')
        ]

    def running_statistics(self):
        return [
            (self.norm_in_running_mean, 'norm_in_running_mean'),
            (self.norm_in_running_var, 'norm_in_running_var'),
            (self.residual_1.bn1_running_mean, 'residual_1.bn1_running_mean'),
            (self.residual_1.bn1_running_var, 'residual_1.bn1_running_var'),
            (self.residual_1.bn2_running_mean, 'residual_1.bn2_running_mean'),
            (self.residual_1.bn2_running_var, 'residual_1.bn2_running_var'),
            (self.residual_2.bn1_running_mean, 'residual_2.bn1_running_mean'),
            (self.residual_2.bn1_running_var, 'residual_2.bn1_running_var'),
            (self.residual_2.bn2_running_mean, 'residual_2.bn2_running_mean'),
            (self.residual_2.bn2_running_var, 'residual_2.bn2_running_var')
        ]

class AdamW:
    def __init__(self, parameters, learning_rate, weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.state = {id(parameter): {'m': np.zeros_like(parameter), 'v': np.zeros_like(parameter)} for parameter, _, _ in parameters}

    def step(self, parameters):
        self.step_count += 1
        bias_correction_1 = 1.0 - self.beta1 ** self.step_count
        bias_correction_2 = 1.0 - self.beta2 ** self.step_count
        for parameter, gradient, _ in parameters:
            state = self.state[id(parameter)]
            state['m'] *= self.beta1
            state['m'] += (1.0 - self.beta1) * gradient
            state['v'] *= self.beta2
            state['v'] += (1.0 - self.beta2) * gradient * gradient
            m_hat = state['m'] / bias_correction_1
            v_hat = state['v'] / bias_correction_2
            parameter *= 1.0 - self.learning_rate * self.weight_decay
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def clip_gradients(parameters, maximum_norm=1.0):
    squared_norm = 0.0
    for _, gradient, _ in parameters: squared_norm += float(np.sum(gradient * gradient))
    norm = np.sqrt(squared_norm)
    if norm > maximum_norm:
        scale = maximum_norm / (norm + 1e-12)
        for _, gradient, _ in parameters: gradient *= scale

def evaluate(model, frame, batch_size):
    predictions, confidence_predictions, targets, confidence_targets, masks, weights, principles = [], [], [], [], [], [], []
    for start in range(0, len(frame), batch_size):
        batch_frame = frame[start:start + batch_size]
        spatial, numeric, principle, target, confidence_target, mask, weight = batch_arrays(batch_frame)
        outputs = model.forward(spatial, numeric, principle, training=False)
        predictions.append(outputs['alignment'])
        confidence_predictions.append(outputs['confidence'])
        targets.append(target)
        confidence_targets.append(confidence_target)
        masks.append(mask)
        weights.append(weight)
        principles.append(principle)
    if not predictions: return {'alignment': {'loss': 0.0, 'sign_accuracy': 0.0, 'active_examples': 0, 'per_principle': {}}, 'confidence': {'loss': 0.0, 'mae': 0.0}}
    prediction = np.concatenate(predictions)
    confidence_prediction = np.concatenate(confidence_predictions)
    target = np.concatenate(targets)
    confidence_target = np.concatenate(confidence_targets)
    mask = np.concatenate(masks)
    weight = np.concatenate(weights)
    principle = np.concatenate(principles)
    active = mask > 0
    if not np.any(active): return {'alignment': {'loss': 0.0, 'sign_accuracy': 0.0, 'active_examples': 0, 'per_principle': {}}, 'confidence': {'loss': 0.0, 'mae': 0.0}}
    active_prediction = prediction[active]
    active_target = target[active]
    active_weight = weight[active]
    sign_threshold = 0.5 * (float(np.max(active_target)) + float(np.min(active_target)))
    alignment_loss = weighted_smooth_l1(active_prediction, active_target, np.ones_like(active_prediction), active_weight)
    sign_accuracy = float(np.mean((active_prediction >= sign_threshold) == (active_target >= sign_threshold)))
    per_principle = {}
    for principle_index in np.unique(principle[active]):
        principle_mask = active & (principle == principle_index)
        principle_prediction = prediction[principle_mask]
        principle_target = target[principle_mask]
        principle_weight = weight[principle_mask]
        principle_loss = weighted_smooth_l1(principle_prediction, principle_target, np.ones_like(principle_prediction), principle_weight)
        per_principle[str(int(principle_index))] = {'loss': principle_loss, 'sign_accuracy': float(np.mean((principle_prediction >= sign_threshold) == (principle_target >= sign_threshold))), 'active_examples': int(np.sum(principle_mask))}
    confidence_loss = weighted_smooth_l1(confidence_prediction[active], confidence_target[active], np.ones_like(confidence_prediction[active]), active_weight)
    confidence_mae = float(np.sum(np.abs(confidence_prediction[active] - confidence_target[active]) * active_weight) / max(float(np.sum(active_weight)), 1e-08))
    return {'alignment': {'loss': alignment_loss, 'sign_accuracy': sign_accuracy, 'active_examples': int(np.sum(active)), 'per_principle': per_principle}, 'confidence': {'loss': confidence_loss, 'mae': confidence_mae}}

def train(c_model_csvs, sgf_dir, epochs, batch_size, learning_rate, confidence_loss_weight, seed):
    set_seed(seed)
    games = replay_sgf_games(sgf_dir)
    raw_data = load_and_merge_c_models(c_model_csvs)
    data = prepare_and_validate_dataset(raw_data, games)
    training_frame, validation_frame = chronological_game_split(data)
    principle_to_idx = build_principle_vocabulary(training_frame)
    training_frame = attach_principle_indices(training_frame, principle_to_idx)
    validation_frame = attach_principle_indices(validation_frame, principle_to_idx)
    if not training_frame: raise ValueError('Training set contains no valid examples')
    if not validation_frame: raise ValueError('Validation set contains no principles seen in training data')
    rng = np.random.default_rng(seed)
    model = PrincipleAlignmentModel(num_principles=len(principle_to_idx), numeric_dim=NUMERIC_DIM, embed_dim=PRINCIPLE_EMBED_DIM, rng=rng)
    parameters = model.parameters_and_gradients()
    optimizer = AdamW(parameters, learning_rate=learning_rate, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        active_examples = 0
        for batch_frame in iter_epoch_batches(training_frame, batch_size, rng):
            spatial, numeric, principle, target, confidence_target, mask, weight = batch_arrays(batch_frame)
            outputs = model.forward(spatial, numeric, principle, training=True)
            loss = training_loss(outputs, target, confidence_target, mask, weight, confidence_loss_weight)
            alignment_gradient = weighted_smooth_l1_gradient(outputs['alignment'], target, mask, weight)
            confidence_gradient = confidence_loss_weight * weighted_smooth_l1_gradient(outputs['confidence'], confidence_target, mask, weight)
            model.backward(alignment_gradient, confidence_gradient)
            parameters = model.parameters_and_gradients()
            clip_gradients(parameters, maximum_norm=1.0)
            optimizer.step(parameters)
            batch_active = int(np.sum(mask > 0))
            total_loss += loss * max(batch_active, 1)
            active_examples += batch_active
        train_loss = total_loss / max(active_examples, 1)
        validation_metrics = evaluate(model, validation_frame, batch_size)
        alignment_metrics = validation_metrics['alignment']
        confidence_metrics = validation_metrics['confidence']
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} validation_alignment_loss={alignment_metrics['loss']:.5f} validation_sign_accuracy={alignment_metrics['sign_accuracy']:.4f} validation_confidence_loss={confidence_metrics['loss']:.5f} validation_confidence_mae={confidence_metrics['mae']:.5f}")
    
    weights_dict = {name: param.astype(np.float32) for param, _, name in model.parameters_and_gradients()}
    weights_dict.update({name: array.astype(np.float32) for array, name in model.running_statistics()})
    np.savez_compressed("model_weights.npz", **weights_dict)
    active_targets = np.asarray([row['target_alignment'] for row in training_frame if row['label_mask'] > 0], dtype=np.float32)
    training_numeric = np.stack([row['numeric_representation'] for row in training_frame])
    with open("model_config.json", "w") as f: json.dump({'num_principles': len(principle_to_idx), 'numeric_dim': NUMERIC_DIM, 'embed_dim': PRINCIPLE_EMBED_DIM, 'spatial_planes': SPATIAL_PLANES, 'conv_channels': CONV_CHANNELS, 'alignment_hidden': ALIGNMENT_HIDDEN, 'confidence_hidden': CONFIDENCE_HIDDEN, 'board_size': BOARD_SIZE, 'evidence_features': bool(np.any(training_numeric[:, 5:17] != 0)), 'target_min': float(active_targets.min()) if active_targets.size else 0.0, 'target_max': float(active_targets.max()) if active_targets.size else 1.0, 'orientation': 'sgfmill_row0_bottom'}, f, indent=2)
    with open("principle_vocab.json", "w") as f: json.dump(principle_to_idx, f, indent=2)
    return model, principle_to_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c-models', nargs='+', default=['CSV/principles_followed.csv', 'CSV/principles_violated.csv'])
    parser.add_argument('--sgf-dir', default='SGF')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--confidence-loss-weight', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=39)
    args = parser.parse_args()
    model, principle_to_idx = train(c_model_csvs=args.c_models, sgf_dir=args.sgf_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, confidence_loss_weight=args.confidence_loss_weight, seed=args.seed)
    print(json.dumps({'principles': len(principle_to_idx)}, indent=2))

if __name__ == '__main__': 
    main()