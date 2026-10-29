import numpy as np
from move import AncientGoOracle, extract_yin_yang_context

class GoBoardEngine:
    """Lightweight Go rules and state engine for legal move filtering."""
    def __init__(self, board_size=19):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8) # 1: Black, -1: White, 0: Empty
        self.current_player = 'B'
        self.move_history = []

    def get_liberties(self, r, c, visited=None):
        """Calculates liberties for a connected stone group."""
        if visited is None:
            visited = set()
        
        visited.add((r, c))
        color = self.board[r, c]
        liberties = set()

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] == 0:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    sub_libs, _ = self.get_liberties(nr, nc, visited)
                    liberties.update(sub_libs)

        return liberties, visited

    def is_legal(self, r, c, player):
        """Verifies move legality under standard Go rules."""
        if self.board[r, c] != 0:
            return False

        color = 1 if player == 'B' else -1
        self.board[r, c] = color

        # Check self-liberties
        libs, group = self.get_liberties(r, c)
        if len(libs) > 0:
            self.board[r, c] = 0
            return True

        # Check if placement captures enemy stones
        enemy_color = -color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == enemy_color:
                e_libs, _ = self.get_liberties(nr, nc)
                if len(e_libs) == 0:
                    self.board[r, c] = 0
                    return True

        self.board[r, c] = 0
        return False

    def get_legal_moves(self, player):
        """Returns list of valid (r, c) coordinates."""
        legal_coords = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c, player):
                    legal_coords.append((r, c))
        return legal_coords

    def apply_move(self, r, c, player):
        """Places a stone and resolves captures."""
        color = 1 if player == 'B' else -1
        enemy_color = -color
        self.board[r, c] = color

        # Execute captures
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == enemy_color:
                e_libs, e_group = self.get_liberties(nr, nc)
                if len(e_libs) == 0:
                    for gr, gc in e_group:
                        self.board[gr, gc] = 0

        self.move_history.append((r, c, player))
        self.current_player = 'W' if player == 'B' else 'B'


def run_classical_game_loop():
    """Main execution loop driven by rule masking and AncientGoOracle."""
    engine = GoBoardEngine(board_size=19)
    oracle = AncientGoOracle(weights_path="model_weights.npz")

    print("=== Ancient Go V2 Game Engine Started ===")

    max_moves = 100
    for move_num in range(1, max_moves + 1):
        active_player = engine.current_player
        legal_moves = engine.get_legal_moves(active_player)

        if not legal_moves:
            print(f"Move {move_num}: No legal moves available for {active_player}. Passing.")
            engine.current_player = 'W' if active_player == 'B' else 'B'
            continue

        # Basic context structure matching training evidence format
        tactical_context = {
            "move_number": move_num,
            "total_moves": max_moves,
            "near_empty": move_num < 20
        }

        # Query Oracle to pick best candidate according to principle rules
        chosen_move, rankings = oracle.seek_best_move(
            current_board=engine.board,
            active_player=active_player,
            legal_coords=legal_moves,
            tactical_context=tactical_context,
            principle_code="P-T05-007"
        )

        if chosen_move == "PASS":
            print(f"Move {move_num} ({active_player}): PASS")
            engine.current_player = 'W' if active_player == 'B' else 'B'
        else:
            r, c = chosen_move
            engine.apply_move(r, c, active_player)
            top_harmony = rankings[0]['harmony_score'] if rankings else 0.0
            print(f"Move {move_num} ({active_player}): Placed stone at ({r}, {c}) | Harmony Score: {top_harmony:.4f}")

    print("\n=== Game Simulation Complete ===")


if __name__ == "__main__":
    run_classical_game_loop()