# Principle-Guided Go

A 19×19 Go move selector that combines two signals:

1. a **learned alignment model** trained on moves annotated with the Go principles they follow or violate, and
2. a **Yin–Yang equilibrium evaluation**: a contextual, mathematical reading of balance on the board, including a read on what the opponent may *really* be doing.

For a candidate move `m` in position `S`, the engine scores

```
F(S, m) = A(S, m) + R(S) · Y(S, m)
```

where `A` is the learned alignment, `Y` is the Yin–Yang evaluation, and `R` says how much Yin–Yang matters in the current position.

The project is a research prototype. It has no tree search and no reading; every decision is a single evaluation of a small set of generated candidates. The current version, **V3**, is implemented in pure NumPy (no PyTorch) and ships with a command-line move selector, a graphical game interface, and a GTP front end. It follows **V1** (PyTorch) and **V2** (an earlier NumPy prototype); see [section 2](#2-versions-v1-v2-and-v3) and [Appendix B](#appendix-b-the-v2-source-in-detail).

> **Version naming.** Earlier drafts of this documentation called the current engine "V2". That was a mislabel: V2 is the earlier NumPy prototype whose source is documented in Appendix B, and the current engine is V3. This single README replaces the former short README and `docs/TECHNICAL_REFERENCE.md`.

V3 is four scripts:

| File | Role |
|---|---|
| `train.py` | Replays SGFs, joins them to principle labels, trains the NumPy network, saves weights/config/vocabulary |
| `move.py` | Loads the trained model, generates candidates, computes `A`, `Y`, `R`, `F`, ranks moves. Also a command-line tool and verification tools |
| `play.py` | Tkinter GUI: human vs AI or AI vs AI, candidate analysis panel, SGF export |
| `gtp.py` | GTP (Go Text Protocol) front end: lets OGS (via `gtp2ogs`), KataGo match scripts and GoGui play against the engine |

---

## Contents

1. [The idea](#1-the-idea)
2. [Versions: V1, V2 and V3](#2-versions-v1-v2-and-v3)
3. [Status and evaluation](#3-status-and-evaluation)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Repository layout](#6-repository-layout)
7. [Data format](#7-data-format)
8. [Training](#8-training)
9. [Choosing moves from the command line](#9-choosing-moves-from-the-command-line)
10. [Playing with the GUI](#10-playing-with-the-gui)
11. [How V3 scores moves](#11-how-v3-scores-moves)
12. [Candidate generation and the endgame/pass rule](#12-candidate-generation-and-the-endgamepass-rule)
13. [Validating by playing matches (OGS, then KataGo)](#13-validating-by-playing-matches-ogs-then-katago)
14. [Conventions: orientation, coordinates, colors](#14-conventions-orientation-coordinates-colors)
15. [Verification tools and what has been tested](#15-verification-tools-and-what-has-been-tested)
16. [Known limitations](#16-known-limitations)
17. [Troubleshooting](#17-troubleshooting)
- [Appendix A: Design brief (the blueprint)](#appendix-a-design-brief-the-blueprint)
- [Appendix B: The V2 source in detail](#appendix-b-the-v2-source-in-detail)
- [Appendix C: Recorded game records](#appendix-c-recorded-game-records)

---

## 1. The idea

The design starts from a reading of Go through Yin and Yang:

- **Yin** is inward, soft, passive, receptive. **Yang** is outward, hard, active, expressive.
- Yin and Yang are **not static**. What counts as balanced depends on context: how weak each side is, how urgent the local situation is, how open the board is, the phase of the game, and the overall shape of the position.
- The goal is not symmetry. It is to play the move that brings the position **one step closer to equilibrium**. A move that overshoots the balance point disturbs it as much as one that falls short.
- Protection is mutual. The stone that saves a chain is yang and the chain it saves is yin, but the saved chain also protects the stone that saved it, so the pair ends up balanced.
- **What the opponent appears to want and what it actually wants can differ.** A model has no human intuition for deception and sacrifice, so the engine looks for observable evidence instead: has the opponent suddenly concentrated on one region while ignoring others? Are there weak stones the opponent is not attacking? Do the liberties of the stones it seems to be attacking also belong to another group? Each signal may or may not be intentional, so they are combined into a probability, not a verdict.

The learned model supplies `A`, a data-driven sense of which moves follow or violate named Go principles. The Yin–Yang evaluation supplies `Y`, a hand-designed sense of equilibrium and intent. `R` weights the second against the first according to context.

---

## 2. Versions: V1, V2 and V3

**V1** was the first implementation, built on PyTorch. **V2** was a NumPy prototype: it dropped PyTorch and wrote its layers by hand, but its source contains no Yin–Yang evaluation. It ranks moves with a learned "harmony" score built from a policy network and a principle-alignment network (details in [Appendix B](#appendix-b-the-v2-source-in-detail)). **V3**, the current version, is a rewrite with its own logic, also in pure NumPy: the data handling, training procedure, Yin–Yang evaluation, move selector and interfaces were redesigned, and the network was re-implemented from scratch with a hand-written forward and backward pass.

### Comparison

| Area | V1 | V2 | V3 |
|---|---|---|---|
| **Implementation** | PyTorch, pandas, scikit-learn and sgfmill. One `.pt` checkpoint holding weights, optimizer state, vocabulary and metrics | NumPy only. `network.py` holds hand-written layers (convolution through im2col, ReLU, residual block, dense, embedding, global average pool) and a NumPy Adam. Weights are saved to one `model_weights.npz`; there is no config or vocabulary file | NumPy and sgfmill only. Hand-written forward and backward pass, batch normalization and AdamW. Saved as `model_weights.npz`, `model_config.json` and `principle_vocab.json` |
| **Networks** | The candidate network described under [What V3 kept from V1](#what-v3-kept-from-v1) | Two networks: a **policy network** (362 outputs: 361 points plus pass) and a **candidate network** (alignment and confidence per principle, plus a value output that is never trained). Both use 64 channels and 6 residual blocks (about 0.47 M and 0.55 M parameters). No batch normalization, no dropout | One candidate network with 2 residual blocks, batch normalization and dropout in the heads. No policy network |
| **Network inputs** | 7 spatial planes plus 31 numeric slots: 13 base features (position, move fraction, captures, neighbors, liberties, group size, opponent atari count, legality flag, game stage) and 18 Yin–Yang slots (before, after and delta of the six dimensions). Three selectable feature sets (`full`, `before_only`, `board_candidate`) | Policy network: 6 planes. Candidate network: 7 planes. The planes are own stones, opponent stones, a constant side-to-move plane, three planes that are never filled, and (candidate network only) the candidate point. Stones are coded 1 / −1. Plus 31 numeric slots, 18 in use: raw board-evidence counts, `move_number/200`, `total_moves/200` and stone counts before and after the move ([B.3](#b3-inputs)) | 7 spatial planes plus 31 numeric slots, 17 in use: candidate position, board evidence, move number and zone. Yin–Yang no longer feeds the network; it lives only in the move scorer |
| **Training data** | Label files required `coordinate_sgf` and `annotation_status`. Rows were joined to moves on game, move number, color and coordinate, and only `auto_candidate` and `reviewed` rows counted toward the loss. Followed = +1, violated = −1. Features were computed by the training code from replayed positions | The CSV rows carry the board themselves: `position_before` (a 361-character string), `coordinate_sgf`, `color`, `principle`, `board_evidence` and `confidence`. No SGF files are read. The policy network trains on `CSV/moves_all.csv`. Followed = 1 and violated = 0 by file name; an optional third file takes its label from `annotation_status`. Principle codes go through a fixed 281-entry registry; rows with other codes are skipped | Rows are joined on game and move number. Each row can carry `board_evidence`, and precomputed feature columns are accepted verbatim. `label_mask` and `label_weight` columns replace the status filter. Followed = 1, violated = 0 by default, overridable per row |
| **Training procedure** | First epoch trains the confidence head only, then alignment plus 0.2 × confidence. Class-balanced sampling. Best epoch kept by validation alignment loss. Train, validation and test split (about 73 / 13 / 14) grouped by board position. Optional pairwise-preference mode trained on comparisons between candidate moves | Both networks in one loop: 5 epochs, batch size 30, Adam (learning rate 1e-3, no weight decay, gradient-norm clip 5.0), no seed, no validation split, batches taken in file order. Squared-error loss for alignment plus squared-error loss for confidence, cross-entropy for the policy. Final-epoch weights saved. As attached, the optimizer receives no gradient for any convolution or dense layer ([B.6](#b6-findings-from-reading-and-running-the-attached-code)) | One joint loss throughout. Row-shuffled batches. Final-epoch weights saved. 75 / 25 split by game, with no test split. No pairwise mode |
| **Alignment output** | Linear head, `tanh` at inference, on −1…+1 targets | Linear outputs with no squash. The confidence output is also linear (unbounded) | `tanh((raw − mid) / half)`, using the target range recorded in `model_config.json` |
| **Move score** | `F = A + R·Y` | `harmony = alignment × confidence + 0.1 × policy logit`, for one fixed principle (default `P-T05-007`). No `R`, no `Y` | `F = A + R·Y`, with `A` the mean over the selected principles |
| **Yin–Yang dimensions** | Fixed formulas. Some dimensions shared inputs (two pairs measured nearly the same quantity), so six dimensions carried fewer than six independent signals | None. `extract_yin_yang_context` only converts board evidence into the numeric feature vector | Six dimensions with disjoint inputs, each a yang-versus-yin ratio in [−1, 1] |
| **Context** | Weakness from atari counts with additive smoothing; a categorical game stage derived from the length of the finished game | Not present | Liberty-weighted weakness, urgency, openness, a continuous game phase from the move number (no dependence on game length), and board shape defined as the overlap of the two colors' stone densities |
| **Adaptation** | Targets from hand-written formulas; softmax weights | Not present | Targets and weights come from two small matrices applied to the context (`target_matrix`, `relevance_matrix`), all exposed in one parameter dictionary |
| **Equilibrium** | Distance to target multiplied by direction of change | Not present | `Y_static`: progress toward the target, with overshoot penalized, scored relative to the median candidate |
| **Protector and protected** | Not modelled | Not present | `Y_pair`: yang is the safety the saved chain gains, yin is the safety the protector gets back, scored by their reciprocity |
| **Opponent intent** | Not modelled | Not present | `Y_intent`: concentration, neglect and shared-liberty signals over the last 20 plies, combined into a feint probability that also raises `R` when intent is ambiguous |
| **Relevance `R`** | `0.20 + 0.55·urgency + 0.25·(1 − openness)` | Not present | `0.20 + 0.45·urgency + 0.20·(1 − openness) + 0.25·intent uncertainty` |
| **Candidates** | Simulated all 361 points plus pass; ranked by tactical value plus a center bias; default limit 32 | Every legal point (up to 361) is scored in one batch; no candidate limit | Tactical points, strategic openings (sized by stone concentration) and local contact points; default limit 24 |
| **Passing** | Pass was a lowest-priority candidate | Passes only when no legal move exists. The play loop stops after 100 moves | Territory rule after move 100: only contested points are played, and the engine passes when none are left |
| **Rules** | Board rules came from `sgfmill`, which accepts suicide | Own rules engine (recursive liberty search): occupied points and suicide are rejected, captures are resolved. No ko or superko rule | Built-in rules engine that rejects suicide; positional superko in the GUI and GTP front end |
| **Coordinates and orientation** | Everything stayed in sgfmill's frame (row 0 at the bottom of the board), but coordinate labels were computed as if row 0 were the top, so exported SGF coordinates and printed board labels were vertically mirrored relative to the standard | Row 0 is the top row throughout: board strings are read row-major from the top-left and SGF letters map directly to row and column. sgfmill is not used | Standard SGF and board orientation in every interface. Boards are mirrored into the training frame only at the model boundary, and `--verify-features` checks that training and inference agree |
| **Interfaces** | Command line and GUI | `move.py` has a demo entry point (five hard-coded points on an empty board). `play.py` is a console loop in which the engine plays both colors for 100 moves. No command-line options, GUI, SGF export or GTP | Adds a GTP front end and two verification tools (see [section 15](#15-verification-tools-and-what-has-been-tested)) |
| **GUI** | Analysis list cleared after every move; fixed SGF player names; AI thread updated Tk directly | None | Analysis stays visible, **Analyze Position** and **Play Selected Candidate**, correct SGF headers and result, AI results returned through a queue |

V1 also had a candidate filter that was effectively inactive (nearly every point passed its threshold) and GUI move-list letters that disagreed with the board labels. Both are gone in V3.

### What V3 kept from V1

- The scoring structure `F = A + R·Y`.
- Principle-conditioned alignment learned from labelled games (*followed* / *violated*), with a second head predicting annotation confidence.
- The network architecture: 7 input planes, a 3×3 convolution with 64 channels, two residual blocks, global average pooling, a 32-dimensional principle embedding, an alignment head (hidden 96, dropout 0.2) and a confidence head (hidden 64, dropout 0.1).
- Optimizer settings and defaults: AdamW (weight decay 1e-4), gradient clipping at 1.0, learning rate 1e-3, batch size 48, 5 epochs, seed 39, confidence loss weight 0.2.
- Six Yin–Yang dimensions with the same names, whose targets depend on a context vector.
- A Tkinter GUI with human-vs-AI and AI-vs-AI modes, a candidate analysis panel and SGF export.

### What V3 changed relative to V2

- **Scoring.** V2 ranks moves by `alignment × confidence + 0.1 × policy logit` for one principle. V3 averages alignment over the selected principles, recenters it, and adds the Yin–Yang term `R·Y`.
- **No policy network.** V3 drops the policy network. Candidates come from a rule-based generator ([section 12](#12-candidate-generation-and-the-endgamepass-rule)) instead of scoring every legal point.
- **Data.** V2 reads each board from the `position_before` string in the CSV. V3 replays SGF games and joins label rows by game and move number, so the position and the move history (needed for the opponent-intent signals) come from the game record. V3 uses a subset of the `board_evidence` keys that V2 reads and adds a zone one-hot.
- **Principle vocabulary.** V2 uses a fixed registry of 281 codes that does not cover every code in the sample label files ([B.6](#b6-findings-from-reading-and-running-the-attached-code)). V3 builds its vocabulary from the codes found in training.
- **Training.** V3 adds batch normalization, AdamW, a seed, shuffled batches, a 75 / 25 game-level validation split and per-epoch validation metrics, and its backpropagation was gradient-checked ([section 15](#15-verification-tools-and-what-has-been-tested)). In the attached V2 source the optimizer receives no gradient for any convolution or dense layer.
- **Rules and passing.** V3's rules engine was compared with `sgfmill` on random moves, positional superko is enforced by the GUI and the GTP front end, and passing follows a territory rule.
- **Interfaces.** V3 adds a command-line move selector, a Tkinter GUI, a GTP front end and two verification tools.

---

## 3. Status and evaluation

V3 has been checked in two ways.

**Engineering checks** (synthetic data; see [section 15](#15-verification-tools-and-what-has-been-tested)). The NumPy backpropagation was gradient-checked, the rules engine was compared with `sgfmill` on thousands of random moves, and the inference path was shown to reproduce the model's own outputs on stored training arrays.

**Informal play against KataGo.** V3 was played against KataGo side by side, with moves relayed by hand, in three to four games. These games were stopped before the end. Observations from them:

- Capture rates were low and similar for both sides through about fifty moves each.
- No clear winner could be read from the unfinished positions.
- Neither side moved toward passing, which is expected at that stage: V3's pass rule only applies after move 100, and only when nothing contested remains.
- In play, the Yin–Yang evaluation and the opponent-intent reasoning behaved as designed.

Two longer recorded games, one against KataGo, are analysed in [section 13.5](#135-results-so-far-recorded-games). These games are informal observations, not a benchmark. **Playing strength has not been measured**, and the engine should be expected to be far weaker than a search-based program such as KataGo. A proper evaluation (calibrated opponents, many games, alternating colors, randomized openings) is outlined in [section 13](#13-validating-by-playing-matches-ogs-then-katago).

**V2** was not evaluated for this documentation. As provided, its training loop does not update the convolution or dense layers ([B.6](#b6-findings-from-reading-and-running-the-attached-code)), so no results are reported for it.

---

## 4. Installation

Requirements:

- Python 3.10 or later (developed and tested on 3.12)
- `numpy` (tested with 2.x)
- `sgfmill`
- `tkinter`, for the GUI only (Debian and Ubuntu: `sudo apt-get install python3-tk`)

```
pip install numpy sgfmill
```

No GPU or deep-learning framework is needed.

---

## 5. Quick start

Place the training games and labels in the default locations (see [section 7](#7-data-format)), then:

```
# 1. Train. Writes model_weights.npz, model_config.json, principle_vocab.json
python train.py

# 2. Check that training and inference see identical features (recommended on real data)
python move.py --verify-features

# 3. Ask for a move
python move.py --demo                                     # empty-board smoke test
python move.py --sgf-file game.sgf --sgf-move-number 40   # position before move 40, history replayed

# 4. Play
python play.py --mode human_vs_ai --human-color B

# 5. Or expose the engine over GTP
python gtp.py
```

---

## 6. Repository layout

Expected layout (default paths; all are overridable on the command line):

```
project/
├── train.py                 # data loading, NumPy network, training, model files
├── move.py                  # candidate generation, A, Y, R, F, passing rule, CLI, verification tools
├── play.py                  # Tkinter GUI
├── gtp.py                   # GTP front end
├── SGF/                     # one .sgf per game (19x19; you provide)
│   └── *.sgf
├── CSV/                     # principle labels (you provide)
│   ├── principles_followed.csv
│   └── principles_violated.csv
│
│   # generated by train.py, read by move.py / play.py / gtp.py
├── model_weights.npz
├── model_config.json
└── principle_vocab.json
```

The V2 sources (`network.py`, `train.py`, `move.py`, `play.py`) are a different program that happens to share file names with V3. They are described in [Appendix B](#appendix-b-the-v2-source-in-detail) and should not be mixed with the V3 files in one directory.

---

## 7. Data format

Training joins **CSV label rows** to **SGF games**.

**SGF:** 19×19 games. Setup stones are honored. A row is matched to its position by `game_id` and `move_number`.

**CSV:** one row = one (move, principle) label.

| Column | Required | Meaning |
|---|---|---|
| `game_id` | yes | The SGF file name without `.sgf` |
| `move_number` | yes | 1-based index in the game's main line (**passes count as moves**) |
| `color` | yes | `B` or `W`; must agree with the SGF move |
| `principle` | yes | Principle code (text). The set of codes seen in training becomes the vocabulary |
| `sgf_file` | yes | SGF file name |
| `target_alignment` | no | Defaults to `1.0` if the CSV file name contains `followed`, `0.0` if it contains `violated`, else `1.0` |
| `confidence` / `confidence_target` | no | Annotation confidence in [0, 1]; default `1.0` |
| `label_mask` | no | `0` removes the row from the loss; default `1.0` |
| `label_weight` | no | Per-row loss weight; default `1.0` |
| `board_evidence` | no (but important) | JSON dict used for numeric features 5-16, see below |
| `spatial_representation`, `numeric_representation` | no | If present, **used verbatim** instead of being computed. Use `--verify-features` to check them against `move.py` |

`board_evidence` keys used by the model:
`friendly_neighbour_count`, `enemy_neighbour_count`, `nearby_friendly_count`, `nearby_enemy_count`, `near_empty`, `captured_count`, `own_group_after` (`{"size", "liberties"}`), `move_number`, `zone` (`corner`/`side`/`center`).

### Model inputs

- **Spatial (7 planes of 19×19):** own stones, opponent stones, empty points (all *before* the move), the candidate point, its 4-neighbours, opponent stones among those neighbours, own stones among those neighbours.
- **Numeric (31 slots; only 0-16 are filled, 17-30 are always zero):**
  0-4 candidate position (`row/18`, `col/18`, edge distances, distance from centre) · 5-12 evidence (neighbour counts, nearby counts, near-empty flag, captures, resulting group size and liberties) · 13 `move_number/200` · 14-16 zone one-hot.
- **Principle:** an embedding looked up by the principle code.

### Targets

The alignment head regresses `target_alignment` (1 = followed, 0 = violated by default); the confidence head regresses the annotation confidence. Loss: weighted smooth-L1. Optimizer: AdamW with gradient clipping. Validation split: the last 25% of games by first appearance in the CSV.

### Sample label files

The sample label files provided with this documentation (`principles_followed.csv`: 7,860 rows, 13 principles; `principles_violated.csv`: 33,257 rows, 6 principles) follow the layout used by V1 and V2 and carry more columns than V3 reads: the board as a `position_before` string, `coordinate_sgf`, `annotation_status` (every row is `auto_candidate`), `annotation_method`, source citations and a per-row `confidence`. All of V3's required columns (`game_id`, `move_number`, `color`, `principle`, `sgf_file`) are present, and `board_evidence` contains every key V3 uses (see [B.3](#b3-inputs) for the full key list). The file names contain `followed` and `violated`, so the default `target_alignment` of 1.0 and 0.0 applies.

---

## 8. Training

```
python train.py
```

Options:

| Option | Default |
|---|---|
| `--c-models` | `CSV/principles_followed.csv CSV/principles_violated.csv` |
| `--sgf-dir` | `SGF` |
| `--epochs` | `5` |
| `--batch-size` | `48` |
| `--learning-rate` | `0.001` |
| `--confidence-loss-weight` | `0.2` |
| `--seed` | `39` |

Per-epoch output: training loss, validation alignment loss, `validation_sign_accuracy` (thresholded at the midpoint of the target range, 0.5 for 0/1 targets), validation confidence loss and MAE.

Outputs (written to the working directory):

| File | Contents |
|---|---|
| `model_weights.npz` | All weights **plus the batch-norm running statistics** (needed for inference) |
| `model_config.json` | Architecture sizes, `numeric_dim`, `num_principles`, and what the model was trained on: `evidence_features` (were features 5-16 ever non-zero?), `target_min`/`target_max`, `orientation` |
| `principle_vocab.json` | Principle code → embedding index |

`move.py` refuses to load weights whose architecture constants differ from `train.py`, and tells you to retrain if the batch-norm statistics are missing (weights from an earlier `train.py`).

---

## 9. Choosing moves from the command line

```
python move.py --demo                                        # empty-board smoke test
python move.py --sgf-file game.sgf --sgf-move-number 40      # position before move 40, history replayed
python move.py --sgf-file game.sgf                           # position after the last move
python move.py --position <361 chars> --color B              # X = black, O = white, . = empty
python move.py --position-file pos.txt --color W --move-number 80
```

The position string is row-major starting at the **top-left** (A19). Positions given this way carry **no move history**, so the opponent-intent term is off. SGF input replays the history automatically.

Useful options: `--candidate-limit 24`, `--openings 4`, `--principles p1,p2` (default: all in the vocabulary), `--parameters overrides.json`, `--output out.json`.

Output (JSON): `selected_move` (SGF coordinate), `selected_move_human`, and for every candidate: `alignment_score` (A), `yin_yang_score` (Y), `yin_yang_relevance` (R), `final_score` (F), `components` (`static`, `pair`, `intent`), per-principle alignment/confidence, and the before/after state vectors. The summary also contains the context vector, targets, dimension weights, the intent diagnostics, the parameters used, warnings and `pass_reason`.

Python API:

```python
import move
engine = move.MoveEngine('model_weights.npz', 'model_config.json', 'principle_vocab.json')
result = engine.choose_move(board, 'B', move_number=41, history=[('B', 'dd'), ('W', 'pp')])
```

`board` is a 19×19 array (0 empty, 1 black, 2 white, row 0 = top) or a 361-character string; `history` is a list of `(color, sgf_coordinate)`.

---

## 10. Playing with the GUI

```
python play.py --mode human_vs_ai --human-color B
python play.py --mode human_vs_ai --human-color random
python play.py --mode ai_vs_ai --max-moves 300
```

Other options: `--weights`, `--config`, `--vocab`, `--parameters`, `--candidate-limit` (24), `--openings` (4), `--principles`, `--output-sgf`, `--output-json`.

Controls: click an intersection to play; **Pass**, **Resign**, **New Game**; **Analyze Position** asks the engine for candidates on your turn; **Play Selected Candidate** plays the highlighted suggestion (only while the analysis matches the current position); **Save SGF** writes the game (correct player names, `RE[...]` on resignation).

The analysis panel lists each candidate with `F` (final), `A` (alignment) and `Y` (yin-yang), shows the opponent-intent estimate (feint probability and its three signals), and under the list the static/pair/intent breakdown of the selected move. After an AI move the list stays visible so you can see why it chose that move.

Rules: suicide is rejected, positional superko is enforced (the AI falls through its ranked candidates to a legal one), two consecutive passes end the game, and `--max-moves` stops the game. The AI runs on a worker thread and hands results back through a queue.

---

## 11. How V3 scores moves

All of this is in `move.py`. Every number below is a named entry in `PARAMETERS` (top of the file).

### 11.1 `A`: learned alignment

**Network.** A small residual convolutional network written directly in NumPy:

```
7 planes (19×19) ─► 3×3 conv, 64 ch ─► batch norm ─► ReLU ─► 2 residual blocks ─► global average pool
                                                                                          │
principle embedding (32) ─────────────┐                                                   ▼
numeric features (31) ────────────────┴──────────────► concatenate ──► alignment head (hidden 96)
                                                                    └─► confidence head (hidden 64, sigmoid)
```

For each candidate the network produces an alignment value per principle; `A` is their **mean** over the selected principles, recentred to (-1, 1):

```
A = tanh( (raw - mid) / half ),   mid = (target_min + target_max)/2,  half = (target_max - target_min)/2
```

With the default 0/1 targets this is `tanh(2·raw − 1)`. The model never saw passes and was trained only on moves that were actually played, so `A` is evidence, not proof of strength.

### 11.2 The state vector `y(S)` (six yang-vs-yin dimensions)

Each dimension is `(yang − yin) / (yang + yin + 1)` from the mover's point of view (-1 = yin pole, +1 = yang pole). The six dimensions use **disjoint inputs**:

| # | Dimension | Yang | Yin |
|---|---|---|---|
| 1 | territory ↔ influence | own-controlled empty points on the frontier | own-controlled empty points that are enclosed |
| 2 | stability ↔ activity | own stones with an enemy stone within 2 points ("engaged") | own stones with none ("quiet") |
| 3 | defense ↔ attack | pressure on enemy groups (size × weakness) | own groups' need of defense |
| 4 | emptiness ↔ fullness | `2·(stones/361) − 1` | |
| 5 | gote ↔ sente | initiative: how often the opponent answers my moves minus how often I answer theirs (answer = within 2 points of the previous move, last 20 plies) | |
| 6 | local ↔ global | spread of own stones (RMS distance from their centroid) | |

"Control" comes from a Gaussian influence field (σ = 2.5): an empty point is own-controlled where own − opponent field exceeds 0.3. **Weakness** of a group by liberties: 1 → 1.0, 2 → 0.5, 3 → 0.15, 4+ → 0.

### 11.3 The context vector `c(S)` (adaptation)

| Factor | Definition |
|---|---|
| own weakness | stone-weighted vulnerability of own groups |
| enemy weakness | same for the opponent |
| local urgency | max over all groups of weakness × min(1, size/4) |
| board openness | fraction of empty points |
| game phase | `min(move_number/200, 1)` |
| board shape | `d(z,l)`: overlap of own/opponent blurred stone densities (0 = separated camps, 1 = fully mixed) |

Two small matrices turn the context into the position's **target** and **weights** for the six dimensions:

```
τ = clip( T · [1, c], -1, 1 )            (target_matrix)
w = normalize( max( Rm · [1, c], 0.05 ) )   (relevance_matrix)
```

So what counts as "balanced" changes with the position (more outward and active early and when the enemy is weak; more inward and stable when you are weak or late in the game).

### 11.4 `Y_static`: one step closer to equilibrium

```
progress_k = ( |y_k − τ_k| − |y'_k − τ_k| ) / ( 1 + |τ_k| )     y' = state after the move
raw        = Σ_k w_k · progress_k
Y_static   = tanh( static_gain · (raw − median over candidates) )
```

Moving *past* the target increases the distance again, so overshooting is penalized (an extreme reverses into its opposite). Scoring relative to the median candidate removes shifts that every move shares (for example a side's very first stone), which would otherwise saturate the score.

### 11.5 `Y_pair`: protector and protected

For a move that saves one or more of your weak chains:

```
yang  = average safety gain of the saved chains            (the placed stone protects)
yin   = safety the placed stone's group gains vs being alone (the saved chain protects back)
ρ     = min(yang, yin) / max(yang, yin)                     (reciprocity)
Y_pair = ρ · tanh( pair_gain · (yang + yin) )
```

`Y_pair` is high when the exchange is **mutual** (for example connecting two chains that each have their own liberties) and 0 when nothing is saved or the protector gets nothing back (a plain extension from atari is handled by `A` and dimension 3).

### 11.6 `Y_intent`: what the opponent appears to want vs what it wants

Needs at least 4 opponent moves in the last 20 plies, otherwise this term is 0. Three signals in [0, 1]:

- **concentration:** the recent half of the opponent's moves shifted far from the earlier half *and* are tightly clustered
- **neglect:** weighted fraction of your weak groups (≤2 liberties) that no recent opponent move came near
- **shared liberties:** fraction of your threatened groups' liberties that are also liberties of weak enemy groups (≤3 liberties)

```
p           = sigmoid( 2 · Σ (signal_i − 0.3) )       probability the apparent threat is a feint
uncertainty = 4 p (1 − p)                              (raises R)
Y_intent    = (1 − p) · near(apparent target) + p · near(hidden target)
```

*Apparent target:* your weak groups the opponent is touching (or, failing that, its recent moves). *Hidden target:* your weak groups it is ignoring, plus the shared-liberty points. `near()` is a Gaussian proximity (σ = 1.5) of the candidate to the region.

### 11.7 Putting it together

```
Y = 0.4·Y_static + 0.3·Y_pair + 0.3·Y_intent
R = clip( 0.20 + 0.45·urgency + 0.20·(1 − openness) + 0.25·uncertainty, 0, 1 )
F = A + R·Y
```

### 11.8 Tuning

Every weight, target, gain and threshold is in `PARAMETERS`. To override without editing code, pass a JSON file:

```json
{ "static_gain": 15.0, "pass_min_move": 120, "y_weights": [0.5, 0.25, 0.25] }
```

```
python move.py --sgf-file game.sgf --parameters overrides.json
python play.py --parameters overrides.json
```

The defaults are **hand-set and uncalibrated**; they need tuning against real games.

---

## 12. Candidate generation and the endgame/pass rule

**Candidates (`φ`).** Up to `--candidate-limit` (default 24) legal moves, taken in this order:

1. *Tactical* points: liberties of any group (either color) with 1-2 liberties (captures and saves ranked first), up to half the limit.
2. *Strategic openings:* `n` (default 4) points in the largest open areas (≥4 from any stone, ≥5 apart, line 3 or higher). The quota is scaled by `(1 − d(z,l))`, so fewer openings are considered when stones are intermixed. On an empty board these are the star points.
3. *Local contact points* within 2 points of any stone, ranked by contact count, closeness to the opponent's last move, and an edge penalty.
4. Any remaining tactical points.

**Rules engine.** `move.py` has its own: occupied points and suicide are rejected (sgfmill silently allows suicide). Ko is left to the controller; `play.py` enforces positional superko.

**Endgame and passing.** After `pass_min_move` (default 100), if no group on the board is in atari, only **contested** points are considered: empty regions that touch **both** colors (dame and boundaries). Regions bordered by one color alone count as settled territory, so the AI neither fills its own territory nor invades the opponent's. If no legal contested point remains, the AI **passes** (`pass_reason: "settled"`). Two consecutive passes end the game.

---

## 13. Validating by playing matches (OGS, then KataGo)

### 13.1 `gtp.py`

OGS's bot bridge (`gtp2ogs`) launches any engine that speaks GTP, and KataGo match scripts do the same, so one adapter serves both.

```
python gtp.py --weights model_weights.npz --config model_config.json --vocab principle_vocab.json
```

Options: `--parameters overrides.json`, `--candidate-limit 24`, `--openings 4`, `--principles a,b`, and `--decision-log decisions.jsonl`.

Supported commands: `protocol_version`, `name`, `version`, `known_command`, `list_commands`, `quit`, `boardsize` (19 only), `clear_board`, `komi` (accepted, ignored), `play`, `genmove`, `reg_genmove`, `undo`, `time_settings` / `time_left` (accepted, ignored), `fixed_handicap`, `place_free_handicap`, `set_free_handicap`, `showboard`. Not supported: `final_score`, `final_status_list`, `kgs-genmove_cleanup` (the engine cannot judge dead stones).

The adapter has been exercised with scripted GTP sessions; its use with online servers or match tooling is not yet verified.

Behaviour to know about:

- **Never resigns.** It passes when the territory rule in §12 says nothing contested is left.
- **Superko is enforced** on its own moves; it falls through its ranked candidates to a legal one, and passes if none exists.
- Handicap stones arrive as `play` commands, so they count toward `move_number` (training counted only SGF moves). This slightly shifts the game-phase inputs in handicap games.
- Speed: about 0.2 s per `genmove` in testing (24 candidates), which is well inside normal online time limits.
- **`--decision-log`** appends one JSON line per `genmove`: move number, color, chosen move, the context vector, the intent estimate, the chosen move's static/pair/intent/relevance breakdown, the top 5 candidates, time taken, and the pass reason. This is the data for judging *why* it played what it played.

### 13.2 OGS (online-go.com)

The following comes from the `gtp2ogs` project's README (https://github.com/online-go/gtp2ogs):

1. Create a **second OGS account** for the bot (keep your human account).
2. Ask an OGS **moderator** to flag the bot account as a bot. Give them both account names.
3. Log in with your human account, open the bot's profile page, and **generate an API key**.
4. Install `gtp2ogs` (`npm install -g gtp2ogs`, or a pre-built executable from its releases page).
5. Copy its `example_config.json5`, set the API key, the bot command, and the games you will accept (19×19, rules, time controls), then run `gtp2ogs -c yourconfig.json5`. A command-line form also exists: `gtp2ogs --apikey KEY -- python /path/to/gtp.py --weights ...` (everything after `--` is the engine command).
6. `gtp2ogs` supports the **beta site** (`--beta`, with a bot account created there). Test the whole chain there first.

Cautions:

- **Scoring is unconfirmed.** I could not verify from the documentation how OGS scores bot games. Because the engine cannot mark dead stones (§16), check how finished games are scored and how disputes are resolved before relying on results.
- Rules, komi and time controls are chosen by the opponent and the `gtp2ogs` config. The engine ignores komi and time.
- Bot accounts have ratings that need many games to settle. Treat early OGS results as qualitative.

### 13.3 KataGo (later)

- **Status.** One full-length game against KataGo has now been recorded and is analysed in [section 13.5.1](#1351-game-against-katago). KataGo's settings, the engine's colour and the result were not recorded, so it is an observation of play and not yet a calibrated measurement. The bullets below describe what a proper evaluation still needs.
- **Calibrated opponents.** Limit KataGo's visits (a small `maxVisits`) and/or use its **HumanSL** model, which imitates human ranks via the `humanSLProfile` setting (`rank_*` or `preaz_*` profiles, roughly 20k to 9d). See KataGo's HumanSL example config. This gives opponents of known approximate strength, which a full-strength KataGo would not.
- **Referee harness.** Two GTP engines have to be connected and the game scored. GoGui's `gogui-twogtp` or a small script can do this; I have not written the harness. Because neither side reliably marks dead stones, plan how the result is decided (for example, ask KataGo to score the final position, or play the game out). Check which scoring commands your KataGo version supports.
- **Games would be identical.** `gtp.py` is deterministic. Against a deterministic opponent, repeated games from the empty board would be the same game, so they add no statistical information. Vary the games with randomised openings or with randomness in the opponent and/or the engine. The recorded games confirm this: the engine gave identical replies to identical positions ([section 13.5.3](#1353-what-the-two-games-show)).
- Alternate colors and use the same komi and rules in every game.

### 13.4 What to measure

- **Win rate with a confidence interval** (for example Wilson) against each opponent level, split by color. Use at least 50 games per level; 100 or more gives usable intervals.
- A **strength estimate** (Elo or rank) by locating where the win rate crosses 50% across a ladder of calibrated opponents.
- **Ablations**, using `--parameters`: `{"y_weights": [0, 0, 0]}` is alignment only (F = A); `[1, 0, 0]` static only; `[0, 1, 0]` pair only; `[0, 0, 1]` intent only. Compare each against the full engine.
- **Principle subsets** with `--principles`, to see whether restricting the engine to one principle changes how it plays.
- **From the decision logs:** how often each component drives the choice, how the feint probability behaves in games, and whether games the engine wins differ from games it loses.

Expect weak play. The engine has no reading or lookahead (§16), and OGS and KataGo opponents search deeply. The useful result is a measured strength ladder plus evidence about *how* it plays, not a high win rate.

### 13.5 Results so far: recorded games

Two longer games played through the GUI were recorded: one against KataGo and one against a human. Their move lists were read from the recordings, replayed through a rules engine to check legality, and are reproduced in [Appendix C](#appendix-c-recorded-game-records). They are single games, so they illustrate how the engine behaves and do not measure its strength.

#### 13.5.1 Game against KataGo

The source is a 20-second clip of the GUI's move-history panel scrolling from move 1 to move 216. The board itself is not visible, and neither are KataGo's settings (visit limit or HumanSL profile), the engine's colour, or the result.

| | |
|---|---|
| Length | 216 moves (108 per side). The record ends at `216. W S8` with no result shown |
| Passes | None |
| Colours | Not stated in the clip. The first four moves are identical to the game against a human below, where the engine is White, and White's replies (K8, M10) are the same, so the engine is taken to be **White** and KataGo **Black**. This is an inference |
| Captures (reconstructed) | Black about 20 stones, White about 19 |
| Stones on the board at move 216 (reconstructed) | Black 89, White 88 |

What the record shows:

- **A centre-first game.** The game opened at tengen (`K10`) and stayed in the middle: 94 of the first 100 plies (47 of 50 per side) are within five points of the centre. Neither side played on the third line or lower until move 106 (White H3) and 107 (Black J3); the first second-line moves came at 114 (White J2) and 125 (Black M2).
- **No first-line play at all.** In 216 moves neither side played on the first line, and only 14 moves (8 Black, 6 White) were on the first two lines. The mean distance from the edge is 5.25 lines for Black and 5.24 for White, so the two sides spread over the board in the same way.
- **Fewer local answers from the engine.** The Black player answered within two points of the previous move 50 times out of 107 (47%); White did so 28 times out of 108 (26%). With the colour inference above, the engine plays locally about half as often as KataGo and more often moves elsewhere.
- **An even exchange.** Captures were nearly equal (about 20 against 19) and stone counts were 89 against 88 at the end of the record. No leader can be read from the move record alone, which agrees with the earlier informal games.
- **No pass.** The pass rule (from move 100, contested points only) produced no pass in 216 moves. White played 8 moves after move 200, 2 of them on the first two lines.

Reconstructed position after move 216 (`X` Black, `O` White, a lower-case letter marks the last move; dead stones are not removed):

```
    A B C D E F G H J K L M N O P Q R S T
19  . . . . . . . . . . . . . . . . . . .  19
18  . . . . . . . . X O . . . . X . . . .  18
17  . . . . . . . X . X X . . . . . . . .  17
16  . . . . . . . . X X X X . X X X . . .  16
15  . . . X . . X O X X . X X . X . . . .  15
14  . . O O . X X X O X X X O X O . . . .  14
13  . O X X O O O O O O X X O O . . O . .  13
12  . X O O O O O O O X X X X X . O X X .  12
11  . X O O O O O O X X X X X X O . O . .  11
10  . X O O O O X X X X X O X X X O O . .  10
 9  . . O O O O X X X X O O O X X X O . .  9
 8  . O . O O O X X O O O O O O X . X o .  8
 7  . X O . O X X X . O O O X . O X O . .  7
 6  . . X O X X O O O O O O O X X . . . .  6
 5  . . X X X X X O O O O O X . X . . . .  5
 4  . . . . . X O . O O O X . X . . . . .  4
 3  . . . . . . . O O O X . X . . . . . .  3
 2  . . . . . . . . O . O X . . . . . . .  2
 1  . . . . . . . . . . . . . . . . . . .  1
    A B C D E F G H J K L M N O P Q R S T
```

*Reconstruction caveat.* The moves were read from low-resolution frames. One move, 197 (`Black C13`), is illegal in the reconstructed position (it would be suicide), so some earlier move or capture was probably misread or entered differently; the capture and stone counts above are therefore approximate. Moves 76 and 116 were never fully visible in any frame and were read from partial frames (116 was corrected from the first reading because the original was impossible).

#### 13.5.2 Game against a human

The source is a 328-second screen recording of the GUI (recorded on 23 August 2026) with the human playing Black (`--human-color B`) and the engine White.

| | |
|---|---|
| Length | 278 moves played. The **last move (278, `W B6`) was the engine's**, and it was Black's turn when the human stopped playing and shut the game off. The human did not resign: the recording's last frame shows the GUI status "Black resigned", but the game was shut off by the human, so there is no result |
| Passes | None |
| Captures (reconstructed) | White about 23 stones, Black about 10 |
| Stones on the board at the end (reconstructed, dead stones included) | White 128, Black 116 |

What the record shows:

- **Same opening as the KataGo game.** Moves 1 to 4 are `B K10, W K8, B K12, W M10` in both games, and the engine's reply at move 8 (`L9`) is also the same. The engine's output is therefore repeatable, which is what [section 13.3](#133-katago-later) assumes. The game departs at move 5 (Black `H10` here, `G10` against KataGo).
- **The same centre-first style.** 45 of the engine's first 50 moves are within five points of the centre; its first third-line move was K3 at move 74 and its first second-line move was K18 at move 118.
- **Much more edge play than the KataGo game.** White played 18 of its 139 moves on the first line (13%) and 34 (24%) on the first two lines; Black played 24 (17%) and 45 (32%). After move 200 White played 24 of its 39 moves on the first two lines (62%), against 26 of 39 for Black.
- **The game was long and the board nearly full.** It reached move 278 with no pass. The game was stopped by the human, not finished: there was no resignation, no pass and no score.

*Reconstruction caveat.* Eight moves that the text reader missed were read by eye. Two moves at the end (265 `Black T9` and 276 `White T9`) do not fit the reconstructed position, so the final stone counts and capture counts are approximate.

#### 13.5.3 What the two games show

1. **The engine is deterministic.** Same position, same reply (first four moves and move 8 in both games). Repeated games from the same start will repeat unless the opening or the opponent varies.
2. **First-move behaviour is still unobserved.** The engine played White in both games, so it has never been seen choosing a first move on an empty board (the star-point openings of [section 12](#12-candidate-generation-and-the-endgamepass-rule)).
3. **No game reached a pass.** Neither the 216-move nor the 278-move record contains a pass, so the territory-based pass rule was not exercised in a real game.
4. **Late-game edge play is heavy in the longer game.** See 3.2; the KataGo record has none because it stops at move 216 with few late edge moves.
5. **Strength is still unmeasured.** Neither game has a result: the human stopped the game against them at move 278 without finishing it, and the KataGo record ends where the recording ends.

---

## 14. Conventions: orientation, coordinates, colors

- **Internal orientation (`move.py`, `play.py`):** row 0 is the **top** row, matching SGF coordinates and what you see on the GUI board.
- **Model orientation:** `sgfmill` (and therefore the training data) uses row 0 = **bottom**. The model also takes absolute row/column as input, so it must see boards in that frame. `move.py` converts at the model boundary (`BoardView` and `mirror_point`), and `model_config.json` records `"orientation": "sgfmill_row0_bottom"`.
- **SGF coordinates:** first letter = column (left→right), second = row (top→bottom). `aa` is the top-left corner.
- **Human coordinates:** columns `A-T` **skipping `I`**, rows numbered from the bottom (`A19` = top-left, `T1` = bottom-right).
- **Colors:** `B`/`W` in text; `1`/`2` in arrays (`BLACK = 1`, `WHITE = 2`, `EMPTY = 0`).
- **Move numbers** are 1-based and count passes. `move_number` passed to the engine is the number of the move about to be played.

---

## 15. Verification tools and what has been tested

### Run these on your own data first

The model's inputs depend on how your CSVs were produced, which the code cannot see. Two checkers compare **exactly what training used** with **what `move.py` feeds the model**:

```
python move.py --verify-evidence    # board_evidence: key-by-key match rate against move.py's definitions
python move.py --verify-features    # the arrays the model trains on vs the arrays inference produces
```

Both accept `--c-models ...` and `--sgf-dir ...`. `--verify-features` prints `"identical": true` when training and inference agree exactly; otherwise it lists which spatial planes and which numeric indices differ and by how much:

| Differing index | Likely cause |
|---|---|
| spatial planes (any) | a different orientation or plane definition in precomputed `spatial_representation` |
| numeric 0-4 | a different position encoding or orientation |
| numeric 5-12 | `board_evidence` definitions differ (for example the "nearby" radius) |
| numeric 13 | `move_number` meaning differs (0- vs 1-based) |
| numeric 14-16 | zone boundaries differ |

`move.py` assumes: "nearby" = the 8 surrounding points, "near empty" = at least one other empty point among them, zone edge = within 4 lines of the edge (`corner` if near two edges, `side` if near one), `captured_count` = stones captured by the move. If your generator differs, change `nearby_radius` / `zone_edge` in `PARAMETERS` or `board_evidence()`.

### Tested so far (on synthetic data only)

- The NumPy backprop was **gradient-checked** against numerical gradients; every parameter group agrees to about 1e-7.
- `move.py`'s rules engine matched `sgfmill` on 3,070 random moves; suicide is rejected.
- **End-to-end parity:** the engine's alignment output equals the model's own output on stored training arrays (max difference 1.8e-7), after fixing an orientation mismatch between training and inference.
- Saved weights reproduce eval-mode outputs exactly (including batch-norm statistics).
- The GUI was run under a virtual display: human moves, AI replies, analysis, resign, SGF export, move limit, new game during AI thinking, AI-vs-AI, and the pass rule.
- A scripted GTP session (`boardsize`, `play`, `genmove`, `showboard`, `undo`) runs correctly.
- Component checks on constructed positions (pair scores, state vectors, initiative, intent signals, finished-position passing, dame handling).

**Not tested:** playing strength, or anything on real labelled data. All checks above used small synthetic games and random labels.

---

## 16. Known limitations

- **No lookahead or sacrifice search.** Deception and sacrifice are only approximated by the intent signals; there is no opponent-reply search. A "bad" move that works as a sacrifice is not recognized as such.
- **Hand-set parameters.** The context matrices, gains, thresholds and the 0.4/0.3/0.3 mix are untuned. `static_gain` in particular needs calibration against real games.
- **Training/inference distribution shift.** `A` is trained only on moves that were actually played, then used to rank moves nobody played. V1's optional pairwise mode, which trained on comparisons between candidate moves, addressed this; V3 has no equivalent.
- **Evidence assumptions** (§15) until verified against your data.
- **Dead stones are not recognized.** The pass rule treats a region as contested if it touches both colors, so dead groups that remain on the board keep the AI playing there; remove them by hand.
- **If the human passes while contested points remain, the AI keeps playing.** It only accepts the end once nothing contested is left.
- `--position` input has no history, so the opponent-intent term is off.
- Numeric slots 17-30 are always zero (reserved); there is no komi or scoring.
- The AI averages alignment over **all** principles in the vocabulary unless `--principles` is given.
- **Heavy edge play in long games.** In the recorded game against a human, 62% of the engine's moves after move 200 were on the first two lines (24 of 39), and neither recorded game reached a pass ([section 13.5](#135-results-so-far-recorded-games)).
- **First moves on an empty board are unobserved in play.** In both recorded games the engine played White ([section 13.5.3](#1353-what-the-two-games-show)).

---

## 17. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `Weights file is missing [...]; retrain with the current train.py` | Weights were saved by an older `train.py` without batch-norm statistics. Retrain |
| `Model config X=… does not match train.py` | Architecture constants in `train.py` differ from the ones the model was trained with. Restore them or retrain |
| `Weight shape mismatch for …` | Same cause as above |
| `--verify-features` reports differences | See the table in §15 |
| Warning "The model was trained without board_evidence…" | None of your training rows had `board_evidence`, so features 5-16 were all zero. `move.py` zeroes them at inference to match. Add `board_evidence` and retrain for better results |
| The AI keeps playing in the late game | Dead stones or contested regions remain (§12, §16); an atari anywhere disables the pass rule |
| `ModuleNotFoundError: tkinter` | Install `python3-tk` (Linux) or a Python build that includes Tk |
| Move labels look mirrored | They should not after the orientation fix; if a CSV or tool you use writes coordinates with row 0 at the bottom, convert them (§14) |

---

## Appendix A: Design brief (the blueprint)

This section records the author's original design, in the author's words (lightly edited for typos only).

### A.1 The blueprint

> **Yin** is inward, soft, passive, receptive.
> **Yang** is outward, hard, active, expressive.
>
> For any stone (s) of the same color on the board, remember: **WHAT THE ENEMY APPEARS TO WANT AND WHAT IT ACTUALLY WANTS CAN BE DIFFERENT.**
>
> Since this ML model does not have a human brain it might not really understand deception and sacrifice – concepts even humans cannot always decipher.
>
> **Game plan:** the model asks questions like the following before making a move:
>
> 1. Are the moves made by the opponent suddenly concentrating on a different part of the board without giving any importance to the others?
> 2. Are there any stones that are unprotected but the opponent is not making a move?
> 3. Are the liabilities (liberties) of the set of stones the opponent appears to capture also the liabilities of another set of stones?
> 4. etc.
>
> All of these might be intentional or unintentional. It is a fifty-fifty possibility since we do not know our opponent's brain.
>
> So what do we do? The stone we place to save a chain of stones is **yang** and the ones being saved are **yin**. However, from a broader perspective, the ones being saved also save the protector indirectly, and hence yin and yang are balanced.
>
> A bad move disrupts yin and yang. But sometimes a bad move is a good move.
>
> Yin and yang are not static. They are **contextual**.
>
> The **contextual vector** is determined on the basis of:
> 1. own weakness
> 2. enemy weakness
> 3. local urgency
> 4. board openness
> 5. game phase
> 6. board shape
>
> Yin and yang is not just symmetry and stuff. It is about making a move that brings us **one step closer to equilibrium**.
>
> Let's say `x` candidates, `n` strategic openings, `z` own stones, `l` opponent stones, `d(z,l)` the function showing stone concentration. Then we need to make a suitable move
> `φ(d(z,l), x, n)`
> which satisfies `Y` (the function for yin-yang balance).
>
> For a position `S` and candidate moves `m`:
> **`F(S,m) = A(S,m) + R(S)·Y(S,m)`**
> where `A` is the function of learned alignment, `R` is the relevance of yin-yang and `Y` is the yin-yang evaluation.

### A.2 How the blueprint maps to the code

| Blueprint idea | Where it lives |
|---|---|
| Yin = inward/soft/passive/receptive, Yang = outward/hard/active/expressive | Each of the six state dimensions is a **yang-vs-yin ratio** in [-1, 1]: -1 is the pure yin pole, +1 the pure yang pole (`state_vector`) |
| Appearance vs real intent | `intent_signals`: a feint probability `p`, an *apparent* target region and a *hidden* target region |
| Q1: sudden concentration elsewhere | `concentration` signal: shift of the opponent's recent-move centroid × tightness of the recent moves |
| Q2: unprotected stones the opponent ignores | `neglect_own` signal (and `neglect_enemy`, a diagnostic) |
| Q3: shared liberties between the stones they seem to attack and another set | `shared_liberties` signal |
| "Fifty-fifty" | The feint probability starts near 50% at moderate signal levels; **uncertainty `4p(1-p)` raises `R`**, so Yin-Yang matters more when intent is ambiguous |
| Protector = yang, protected = yin, mutually balanced | `pair_score`: yang = safety the saved chain gains, yin = safety the protector gets back, reciprocity = min/max |
| Contextual, not static | Context vector `c(S)` → per-dimension **targets** `τ(c)` and **weights** `w(c)` (two small matrices in `PARAMETERS`) |
| The six context factors | `context_vector`: own weakness, enemy weakness, local urgency, board openness, game phase, board shape |
| "One step closer to equilibrium" | `static_progress`: how much a move reduces the distance between the position's state and the context's target; overshooting counts against the move |
| `d(z,l)` stone concentration | Overlap (histogram intersection) of the two colors' blurred stone-density maps. Used as the *board shape* context factor and to size the strategic-openings quota |
| `φ(d(z,l), x, n)` | `generate_candidates` / `candidate_pool` |
| `F(S,m) = A + R·Y` | `MoveEngine.choose_move` |
| "A bad move can be a good move" / sacrifice | **Not searched yet** (no opponent-reply lookahead). See [limitations](#16-known-limitations) |

### A.3 Working agreements and design decisions

**Decided by the author**

- No PyTorch: the network is implemented from scratch in NumPy (own forward and backward pass). V3 is the author's own logic, not a port of the V1 code.
- Keep the author's formatting style across all files: compact one-line `if`s, single quotes, no comments, CRLF line endings.
- The assistant should not make design decisions without asking; the author later delegated the remaining open decisions ("change whichever you feel need to be changed").
- Scope of `Y`: fix V1's overlapping dimensions, add the protector/protected pairing, add opponent-intent signals, make it contextual/adaptive, with a Taoist reading and a mathematical formalization. The author approved the resulting structure.

**Decided by the assistant under that delegation** (change any of these in `PARAMETERS` or the code)

- `A` is recentered with `tanh` instead of a linear clip; the static term is scored **relative to the median candidate**; `static_gain = 20`.
- `Y = 0.4·static + 0.3·pair + 0.3·intent`.
- A move that saves a chain but gets nothing back for the protector scores 0 on `pair` (it is neither rewarded nor penalized).
- The shared-liberties signal compares the threatened groups' liberties with *weak enemy* groups' liberties.
- Passing is rule-based (empty-region territory rule), not model-scored. See §12.
- The model always sees boards in the orientation it was trained in (§14).
- `train.py` also records `evidence_features`, `target_min`, `target_max` and `orientation` in `model_config.json`.

---

## Appendix B: The V2 source in detail

This appendix documents the V2 code as it was provided (`network.py`, `train.py`, `move.py`, `play.py`). V2 is kept for reference; V3 replaces it. Statements below come from reading the source, except those in [B.6](#b6-findings-from-reading-and-running-the-attached-code), which say how each one was established.

### B.1 Overview and files

V2 is a NumPy-only prototype. It trains two networks together and ranks moves by combining their outputs. There is no Yin–Yang scoring, no candidate generator, no GUI and no GTP front end.

| File | Role |
|---|---|
| `network.py` | Hand-written layers (`GoConv2D` through im2col/col2im, `GoReLU`, `GoResBlock`, `GoDense`, `GoEmbedding`, `GoGlobalAvgPool2D`), the two networks, the loss functions, the `board_to_spatial_planes` encoder and `NumPyAdam` |
| `train.py` | Loads the CSV files, trains both networks in one loop, writes `model_weights.npz` |
| `move.py` | `AncientGoOracle`: loads the weights and ranks a list of legal points; `extract_yin_yang_context` converts a `board_evidence` dictionary into the 31-slot numeric vector |
| `play.py` | `GoBoardEngine` (rules) and a console loop in which the oracle plays both colors for 100 moves |

### B.2 Networks

```
PolicyNet   (469,826 parameters)

6 planes (19×19) ─► 3×3 conv, 64 ch ─► ReLU ─► 6 residual blocks ─┬─► 1×1 conv (1 ch) ─► 361 point logits ─┐
                                                                  │                                        ├─► 362 logits
                                                                  └─► flatten (64·19·19) ─► dense ─► pass logit ─┘

CandidateNet   (553,075 parameters)

7 planes ─► 3×3 conv, 64 ch ─► ReLU ─► 6 residual blocks ─► global average pool (64) ──┐
31 numeric slots ─► dense 64 ─► ReLU ─► dense 64 ─► ReLU (64) ─────────────────────────┼─► concatenate (160) ─┬─► alignment head: dense ─► 281
principle id ─► embedding (281 × 32) ────────────────────────────────────────────────┘                      ├─► confidence head: dense ─► 281
                                                                                                           └─► value head: dense ─► 1
```

- A residual block is convolution, ReLU, convolution, add the input, ReLU.
- There is no batch normalization and no dropout. Weights use He initialization and the embedding uses N(0, 0.01²). Parameters are held as float64 in memory and saved as float32.
- Each head is a single dense layer with one output per principle (281). The labelled principle is picked out with a mask during training and by index at inference. The principle id reaches the heads only through the embedding.
- The value head has no loss and is not used at inference.
- The confidence head has no sigmoid, so its output is unbounded.

### B.3 Inputs

**Spatial planes** (`board_to_spatial_planes`): stones are coded `1` (black), `-1` (white), `0` (empty).

| Plane | Content |
|---|---|
| 0 | Stones of the player to move |
| 1 | Opponent stones |
| 2 | Constant plane: 1.0 when Black is to move, 0.0 when White is |
| 3, 4, 5 | Never filled (always zero) |
| 6 | The candidate point (candidate network only; the policy network receives 6 planes) |

**Numeric slots** (31, of which 0–17 are filled from `board_evidence`; 18–30 are always zero). Slots 0–11 are raw, un-normalized counts.

| Slot | Source |
|---|---|
| 0 | `friendly_neighbour_count` |
| 1 | `enemy_neighbour_count` |
| 2 | `nearby_friendly_count` |
| 3 | `nearby_enemy_count` |
| 4 | `strong_enemy_group_count` |
| 5 | `enemy_atari_group_count` |
| 6 | `friendly_atari_group_count` |
| 7 | `near_empty` (0 or 1) |
| 8 | `captured_count` |
| 9, 10 | `own_group_after.size`, `own_group_after.liberties` |
| 11 | `ko_candidate` (0 or 1) |
| 12 | `move_number / 200` |
| 13 | `total_moves / 200` |
| 14, 15 | `stone_counts_before` black, white (÷ 180) |
| 16, 17 | `stone_counts_after` black, white (÷ 180) |

The sample label files carry exactly these `board_evidence` keys (plus `zone`, `stage`, `move_point` and `captured_coordinates_sgf`, which V2 ignores). The slot layout differs from V3's, so V2 and V3 features and weights are not interchangeable.

### B.4 Data and training

**Principle registry.** `PRINCIPLE_TO_IDX` assigns indices 0–280 to codes `P-T{t:02d}-{p:03d}` for `t = 1…9` and `p = 1…34`, stopping at 281 entries. That gives `P-T01-001` … `P-T08-034` plus `P-T09-001` … `P-T09-009`. Rows whose code is not in this registry are skipped without a message.

**Candidate dataset** (`principles_followed.csv`, `principles_violated.csv`, and an optional `principles_review_queue.csv`):

- Columns read: `principle`, `position_before` (361 characters, row-major from the top-left; `X` black, `O` white, `.` empty), `color`, `coordinate_sgf`, `board_evidence` (JSON), `confidence` (default 0.5) and, for the review-queue file only, `annotation_status`.
- Alignment target: `1.0` for the followed file, `0.0` for the violated file; for the review-queue file `1.0` if `annotation_status == "followed"`, else `0.0`.
- Each row gives a one-hot mask over the 281 principles, so only the labelled principle contributes to the loss.

**Policy dataset** (`CSV/moves_all.csv`): `position_before`, `color` and `coordinate_sgf`. The target is `row·19 + col`, or `361` (pass) when there is no valid coordinate. The legal mask is all ones.

**Loop** (`train.py`):

| Setting | Value |
|---|---|
| Epochs / batch size | 5 / 30 |
| Optimizer | Adam, learning rate 1e-3, β = (0.9, 0.999), ε = 1e-8, no weight decay |
| Gradient clipping | Global norm, 5.0 |
| Policy loss | Softmax cross-entropy over 362 outputs |
| Candidate loss | Sum of squared errors over labelled entries ÷ number of labels, for alignment plus the same for confidence (weight 1 : 1) |
| Batches | Taken in file order (no shuffling). Each step uses one policy batch and one candidate batch; an epoch has `max(policy batches, candidate batches)` steps |
| Seed / validation | No seed is set and there is no validation split. Only training losses are printed |
| Output | `model_weights.npz`, with keys `policy_layer_{i}_p_{j}` and `candidate_layer_{i}_p_{j}`. No config or vocabulary is written |

### B.5 Inference and the play loop

`AncientGoOracle.seek_best_move(board, player, legal_coords, tactical_context, principle_code)`:

1. Run the policy network on the position to get 362 logits.
2. Build one 7-plane input per legal point and run all of them through the candidate network as a single batch, with the numeric vector from `extract_yin_yang_context(tactical_context)` repeated for every candidate.
3. For each point take the alignment and confidence of the chosen principle and compute

```
harmony = alignment × confidence + 0.1 × policy_logit[point]
```

4. Sort by `harmony` and return the best point and the full ranking. It returns `"PASS"` only when the list of legal points is empty.

The default principle is `P-T05-007` (named "Begin with corner anchors" in the sample label files). If the principle code is not in the registry, index 0 is used. If `model_weights.npz` is missing, the oracle prints a warning and runs with randomly initialized weights.

`play.py` implements a rules engine (`GoBoardEngine`: recursive liberty search, suicide rejected unless the move captures, captures resolved, no ko rule) and a loop that asks the oracle for a move for each side, 100 times, printing each move. The `tactical_context` it passes contains only `move_number`, `total_moves` (fixed at 100) and `near_empty` (true before move 20).

### B.6 Findings from reading and running the attached code

These were established while preparing this document. Items 1 and 2 were checked by running code; the others are read from the source.

1. **Convolution and dense layers receive no gradient from the optimizer.**
   *How established:* a script mirroring the candidate-network step of `train.py` (reduced to one residual block to keep it fast, synthetic batch) was run through one backward pass and one `NumPyAdam.step()`. Only 1 of 17 parameter tensors changed (the embedding table). The same check on the policy network changed 0 of 10 tensors.
   *Cause:* `GoConv2D.backward` and `GoDense.backward` assign `self.grads[0] = ...`, which replaces the array. `train.py` hands the optimizer the original gradient arrays collected before training and zero-fills them each step, so the optimizer only ever sees zeros. `GoEmbedding.backward` writes into its array in place, which is why the embedding alone trains. The layers' own `grads` do hold the correct values (the check printed non-zero norms for them).
   *Consequence:* in a saved checkpoint every convolution, dense layer and head is still at its random initialization, and only the principle embeddings differ. The alignment, confidence and policy outputs therefore do not reflect the training data. Writing gradients in place (`self.grads[0][...] = ...`) in `GoConv2D` and `GoDense` would restore the flow.
2. **The registry does not cover every principle code in the sample labels.**
   *How established:* all codes in the sample files were checked against the registry. 2,474 of the 7,860 followed rows (31.5%) use `P-T03-045` or `P-T03-062`, which lie outside it (group `T03` stops at 034), so they are skipped. All 33,257 violated rows are covered. Of the usable rows only about 14% are *followed*, so the labels lean heavily toward *violated*.
3. **Batches are not shuffled.** The followed file is read first, so within an epoch the early batches are all target 1 and the later ones all target 0.
4. **Training and play see different features.** Training rows carry 18 filled slots and the real game length in `total_moves` (183 and 231 in the sample). `play.py` supplies three (`move_number`, a fixed `total_moves` of 100, `near_empty`) and leaves the rest at zero.
5. **The policy and review-queue data were not provided.** `CSV/moves_all.csv` and `CSV/principles_review_queue.csv` were not among the files given for this documentation. If they are absent at training time the loaders return nothing and the policy network is not trained at all.
6. **The confidence output is unbounded.** It has no sigmoid but is trained toward targets in [0, 1] and used as a multiplier of alignment, so a negative confidence flips the sign of the alignment term.
7. **Cost.** Every legal point (up to 361) goes through a six-block, 64-channel network in one float64 batch. This was not benchmarked here.
8. **No ko rule and no pass logic.** The play loop can repeat positions, and a side passes only when it has no legal move.

---

## Appendix C: Recorded game records

Move lists read from the two recordings described in [section 13.5](#135-results-so-far-recorded-games). Black plays the odd-numbered moves and White the even ones; coordinates use the GUI's human notation (columns `A–T` without `I`, rows from the bottom). The engine is White in both games (in the KataGo game this is inferred).

**How they were read.** The KataGo clip shows only the move-history panel, so the moves were read from its frames, first with OCR and then by eye, and replayed through a rules engine. The human game was read from the history panel of the screen recording the same way. The reconstructions are consistent except at the points noted in sections 13.5.1 and 13.5.2 (moves 197, 265 and 276); a few lines that were cut off at the bottom of a frame were read from their upper half.

### C.1 Game against KataGo (216 moves, no result shown)

```
1.K10 2.K8 3.K12 4.M10 5.G10 6.J10 7.H10 8.L9 9.K11 10.K13
11.L13 12.L11 13.L12 14.J12 15.J13 16.J8 17.J9 18.H11 19.J11 20.H9
21.G9 22.H13 23.H8 24.J14 25.L10 26.K7 27.K14 28.K15 29.L14 30.M12
31.M14 32.L7 33.M13 34.F10 35.M11 36.N11 37.N12 38.K6 39.N10 40.N9
41.O11 42.O10 43.O9 44.O8 45.P10 46.P9 47.P8 48.M7 49.Q9 50.M8
51.M9 52.J6 53.H7 54.N8 55.N7 56.M10 57.M12 58.L6 59.M6 60.N6
61.O6 62.M5 63.N5 64.K5 65.L5 66.L4 67.G8 68.H6 69.H9 70.M9
71.J10 72.G11 73.L11 74.M6 75.N11 76.F9 77.P9 78.G6 79.F7 80.L5
81.G7 82.H5 83.F6 84.G12 85.F5 86.F8 87.G5 88.E10 89.E8 90.J5
91.E9 92.K4 93.H4 94.J4 95.P15 96.L15 97.L16 98.L8 99.K9 100.N13
101.O10 102.P11 103.O12 104.G4 105.O14 106.H3 107.J3 108.O5 109.P5 110.K3
111.P6 112.J13 113.O4 114.J2 115.F4 116.Q10 117.Q11 118.N4 119.M4 120.Q12
121.N3 122.M3 123.L3 124.R11 125.M2 126.F12 127.E12 128.D10 129.E11 130.C10
131.D11 132.M15 133.M16 134.D9 135.L17 136.E7 137.C8 138.D8 139.D7 140.E9
141.E6 142.L2 143.E5 144.O13 145.D5 146.D6 147.C5 148.C7 149.C6 150.C9
151.B7 152.B8 153.B10 154.C11 155.D12 156.G13 157.H14 158.F13 159.P18 160.K16
161.K17 162.E13 163.J16 164.N14 165.J15 166.P14 167.N15 168.L15 169.M15 170.H15
171.K15 172.O15 173.O16 174.R10 175.O14 176.E8 177.R12 178.H12 179.G14 180.R13
181.F14 182.F11 183.S12 184.C12 185.D13 186.D14 187.C13 188.R9 189.B12 190.C14
191.B11 192.B13 193.D13 194.D12 195.D15 196.E11 197.C13 198.D11 199.G15 200.O15
201.P16 202.E12 203.Q16 204.J17 205.O14 206.J3 207.K16 208.Q8 209.H17 210.P7
211.J18 212.K18 213.Q7 214.R7 215.R8 216.S8
```

### C.2 Game against a human (278 moves, stopped by the human, human resigns)

```
1.K10 2.K8 3.K12 4.M10 5.H10 6.J11 7.L11 8.L9 9.J9 10.N10
11.H9 12.L12 13.J12 14.G10 15.H11 16.J8 17.H12 18.J10 19.K11 20.J10
21.J11 22.K13 23.K14 24.H8 25.G9 26.G8 27.G11 28.J7 29.F10 30.J6
31.H7 32.K6 33.H6 34.L6 35.L7 36.K5 37.L8 38.M8 39.K7 40.M7
41.M6 42.N9 43.J10 44.L7 45.G10 46.F8 47.M5 48.K4 49.L13 50.M13
51.J13 52.N7 53.M12 54.K7 55.N6 56.O10 57.N11 58.J5 59.M11 60.E8
61.E9 62.O7 63.P7 64.O08 65.O6 66.L5 67.L4 68.N8 69.P8 70.P10
71.Q10 72.O09 73.P9 74.K3 75.P11 76.F7 77.O11 78.J4 79.J3 80.J14
81.J2 82.L14 83.K2 84.H13 85.L3 86.G12 87.H5 88.E10 89.H3 90.F11
91.F9 92.O12 93.O13 94.D8 95.P12 96.F6 97.N12 98.K15 99.H4 100.K16
101.D9 102.N13 103.C8 104.D10 105.D7 106.G7 107.E7 108.J15 109.E6 110.K17
111.E5 112.L16 113.F5 114.H15 115.G5 116.M16 117.G6 118.K18 119.G4 120.M15
121.K9 122.N15 123.L10 124.J17 125.L17 126.G15 127.F15 128.K13 129.F14 130.L18
131.M17 132.M18 133.F13 134.N17 135.O17 136.H16 137.H17 138.O15 139.O14 140.G14
141.F12 142.K14 143.F16 144.L12 145.G16 146.K19 147.G19 148.L13 149.T19 150.J18
151.S18 152.H18 153.S19 154.G17 155.R19 156.L17 157.R18 158.L15 159.R17 160.P15
161.S17 162.P14 163.T17 164.P13 165.N14 166.Q13 167.O12 168.R13 169.Q12 170.O16
171.R11 172.J19 173.R12 174.E11 175.S12 176.S13 177.T12 178.Q14 179.T13 180.E12
181.F17 182.R10 183.F18 184.C10 185.Q11 186.Q9 187.F19 188.E13 189.Q19 190.B10
191.Q17 192.D12 193.C11 194.C9 195.B9 196.C7 197.B8 198.L19 199.H19 200.E14
201.E15 202.C12 203.B12 204.B11 205.A11 206.A12 207.A13 208.A10 209.A9 210.A12
211.B13 212.N5 213.O5 214.M4 215.N4 216.M3 217.M2 218.L2 219.N3 220.G13
221.D14 222.H14 223.D13 224.N2 225.C13 226.M1 227.M19 228.N19 229.O18 230.Q8
231.P19 232.P6 233.P18 234.Q7 235.Q6 236.P8 237.R6 238.L3 239.S6 240.R9
241.T6 242.S10 243.N18 244.T10 245.O19 246.R7 247.M19 248.S9 249.S11 250.N19
251.S14 252.T11 253.T14 254.L4 255.N5 256.B7 257.A8 258.A7 259.A6 260.R8
261.S8 262.S7 263.T8 264.T7 265.T9 266.M2 267.N1 268.O1 269.P3 270.L1
271.C2 272.K1 273.D4 274.J1 275.D16 276.T9 277.B18 278.B6
(game stopped by the human; Black to move)
```

---

## Acknowledgements

SGF handling uses [`sgfmill`](https://github.com/mattheww/sgfmill). KataGo was used as an informal comparison opponent during testing
