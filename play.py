import argparse
import json
import random
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sgfmill import boards

import move
import train as training


BOARD_SIZE = training.BOARD_SIZE
BOARD_COLOR = "#d8a24a"
LINE_COLOR = "#4c3219"
PANEL_COLOR = "#f3ead7"
ACCENT_COLOR = "#8f1d2c"
TEXT_COLOR = "#241b16"


class GoGame:
    def __init__(self):
        self.board = boards.Board(BOARD_SIZE)
        self.current_color = "b"
        self.move_number = 1
        self.consecutive_passes = 0
        self.position_history = {training.board_hash(self.board)}
        self.moves = []
        self.game_over = False

    def point_from_sgf(self, coordinate):
        if str(coordinate).lower() == "pass":
            return None
        return training.sgf_coordinate_to_row_col(str(coordinate).lower())

    def apply(self, color, point, move_sgf=None, source="human"):
        color = str(color).lower()
        if color not in {"b", "w"}:
            raise ValueError("Color must be b or w")
        if color != self.current_color:
            raise ValueError("It is not this color's turn")
        if self.game_over:
            raise ValueError("The game is already over")

        before = self.board.copy()
        if point is None:
            after = before.copy()
            coordinate = "pass"
            self.consecutive_passes += 1
        else:
            after = before.copy()
            try:
                after.play(point[0], point[1], color)
            except Exception as error:
                raise ValueError("That move is not legal") from error
            if training.board_hash(after) in self.position_history:
                raise ValueError("That move repeats a previous board position")
            coordinate = move_sgf or training.coordinate_sgf(point)
            self.consecutive_passes = 0

        captured = self.captured_points(before, after, color)
        self.board = after
        self.position_history.add(training.board_hash(after))
        self.moves.append(
            {
                "move_number": self.move_number,
                "color": color.upper(),
                "coordinate_sgf": coordinate,
                "coordinate_human": human_coordinate(point),
                "captured_count": len(captured),
                "captured_coordinates": [training.coordinate_sgf(value) for value in captured],
                "source": source,
            }
        )
        self.move_number += 1
        self.current_color = "w" if color == "b" else "b"
        if self.consecutive_passes >= 2:
            self.game_over = True
        return self.moves[-1]

    def captured_points(self, before, after, color):
        opponent = "w" if color == "b" else "b"
        before_points = {
            point for point in training.all_points(BOARD_SIZE) if training.board_value(before, point[0], point[1]) == opponent
        }
        after_points = {
            point for point in training.all_points(BOARD_SIZE) if training.board_value(after, point[0], point[1]) == opponent
        }
        return sorted(before_points - after_points)

    def legal_point(self, coordinate):
        point = self.point_from_sgf(coordinate)
        if point is None:
            return None
        simulated = self.board.copy()
        try:
            simulated.play(point[0], point[1], self.current_color)
        except Exception as error:
            raise ValueError("The selected AI move is not legal on the live board") from error
        if training.board_hash(simulated) in self.position_history:
            raise ValueError("The selected AI move repeats a previous board position")
        return point


def human_coordinate(point):
    if point is None:
        return "pass"
    row, column = point
    columns = "ABCDEFGHIJKLMNOPQRS"
    return f"{columns[column]}{BOARD_SIZE - row}"


def parse_principles(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_human_color(value):
    color = str(value).lower()
    if color == "random":
        return random.choice(["b", "w"])
    if color not in {"b", "w"}:
        raise ValueError("Human color must be B, W, or random")
    return color


class GoInterface:
    def __init__(self, root, arguments):
        self.root = root
        self.arguments = arguments
        self.game = GoGame()
        self.engine = None
        self.ai_busy = False
        self.last_analysis = None
        self.selected_candidate = None
        self.cell_size = 34
        self.margin = 38
        self.canvas_size = self.margin * 2 + self.cell_size * (BOARD_SIZE - 1)
        self.principles = parse_principles(arguments.principles)
        self.setup_window()
        self.draw_board()
        self.root.after(50, self.load_engine)

    def setup_window(self):
        self.root.title("Principle-Guided Go")
        self.root.configure(bg=PANEL_COLOR)
        self.root.minsize(1050, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        board_frame = ttk.Frame(outer)
        board_frame.pack(side="left", fill="both", expand=False)

        self.canvas = tk.Canvas(
            board_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            background=BOARD_COLOR,
            highlightthickness=1,
            highlightbackground=LINE_COLOR,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.board_click)

        side = ttk.Frame(outer, padding=(18, 0, 0, 0))
        side.pack(side="left", fill="both", expand=True)

        self.title_label = ttk.Label(side, text="Principle-Guided Go", font=("TkDefaultFont", 17, "bold"))
        self.title_label.pack(anchor="w", pady=(0, 12))

        self.status_var = tk.StringVar(value="Loading model...")
        self.status_label = ttk.Label(side, textvariable=self.status_var, wraplength=300)
        self.status_label.pack(anchor="w", fill="x", pady=(0, 12))

        controls = ttk.LabelFrame(side, text="Game")
        controls.pack(fill="x", pady=(0, 12))

        self.turn_var = tk.StringVar(value="Turn: Black")
        ttk.Label(controls, textvariable=self.turn_var).pack(anchor="w", padx=8, pady=(8, 4))
        self.move_var = tk.StringVar(value="Move: 1")
        ttk.Label(controls, textvariable=self.move_var).pack(anchor="w", padx=8, pady=2)

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x", padx=8, pady=8)
        self.pass_button = ttk.Button(button_row, text="Pass", command=self.pass_move)
        self.pass_button.pack(side="left", padx=(0, 5))
        self.resign_button = ttk.Button(button_row, text="Resign", command=self.resign)
        self.resign_button.pack(side="left", padx=5)
        self.new_button = ttk.Button(button_row, text="New Game", command=self.new_game)
        self.new_button.pack(side="left", padx=5)

        analysis_frame = ttk.LabelFrame(side, text="AI candidate analysis")
        analysis_frame.pack(fill="both", expand=True, pady=(0, 12))
        self.candidate_list = tk.Listbox(analysis_frame, height=12, width=42, font=("TkFixedFont", 10))
        self.candidate_list.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.candidate_list.yview)
        scrollbar.pack(side="right", fill="y", padx=8, pady=8)
        self.candidate_list.configure(yscrollcommand=scrollbar.set)
        self.candidate_list.bind("<<ListboxSelect>>", self.select_candidate)

        self.play_candidate_button = ttk.Button(side, text="Play Selected Candidate", command=self.play_selected_candidate)
        self.play_candidate_button.pack(fill="x", pady=(0, 8))

        moves_frame = ttk.LabelFrame(side, text="Move history")
        moves_frame.pack(fill="both", expand=True, pady=(0, 12))
        self.move_list = tk.Listbox(moves_frame, height=8, width=42, font=("TkFixedFont", 10))
        self.move_list.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        move_scrollbar = ttk.Scrollbar(moves_frame, orient="vertical", command=self.move_list.yview)
        move_scrollbar.pack(side="right", fill="y", padx=8, pady=8)
        self.move_list.configure(yscrollcommand=move_scrollbar.set)

        self.save_button = ttk.Button(side, text="Save SGF", command=self.save_sgf)
        self.save_button.pack(fill="x")

    def load_engine(self):
        try:
            self.engine = move.MoveEngine(self.arguments.checkpoint)
            self.status_var.set(self.ready_status())
            self.refresh_controls()
            if self.is_ai_turn():
                self.root.after(250, self.request_ai_move)
        except Exception as error:
            self.status_var.set("Model could not be loaded")
            messagebox.showerror("Model loading error", str(error), parent=self.root)
            self.refresh_controls()

    def ready_status(self):
        if self.arguments.mode == "ai_vs_ai":
            return "AI vs AI ready"
        return f"Your color: {self.arguments.human_color.upper()}"

    def is_ai_turn(self):
        if self.game.game_over:
            return False
        if self.arguments.mode == "ai_vs_ai":
            return True
        return self.game.current_color != self.arguments.human_color

    def refresh_controls(self):
        color_name = "Black" if self.game.current_color == "b" else "White"
        self.turn_var.set(f"Turn: {color_name}")
        self.move_var.set(f"Move: {self.game.move_number}")
        state = "Game over" if self.game.game_over else self.ready_status()
        if not self.ai_busy and self.engine is not None and not self.game.game_over:
            self.status_var.set(f"{state} — {color_name} to play")
        self.pass_button.configure(state="disabled" if self.ai_busy or self.game.game_over or self.is_ai_turn() else "normal")
        self.resign_button.configure(state="disabled" if self.ai_busy or self.game.game_over or self.is_ai_turn() else "normal")
        self.play_candidate_button.configure(state="normal" if self.selected_candidate and not self.ai_busy else "disabled")

    def draw_board(self):
        self.canvas.delete("all")
        left = self.margin
        top = self.margin
        right = self.margin + self.cell_size * (BOARD_SIZE - 1)
        bottom = self.margin + self.cell_size * (BOARD_SIZE - 1)
        for index in range(BOARD_SIZE):
            x = left + index * self.cell_size
            y = top + index * self.cell_size
            self.canvas.create_line(x, top, x, bottom, fill=LINE_COLOR)
            self.canvas.create_line(left, y, right, y, fill=LINE_COLOR)

        columns = "ABCDEFGHJKLMNOPQRST"
        for index, label in enumerate(columns):
            x = left + index * self.cell_size
            self.canvas.create_text(x, top - 20, text=label, fill=TEXT_COLOR, font=("TkDefaultFont", 9, "bold"))
            self.canvas.create_text(x, bottom + 20, text=label, fill=TEXT_COLOR, font=("TkDefaultFont", 9, "bold"))
        for index in range(BOARD_SIZE):
            y = top + index * self.cell_size
            label = str(BOARD_SIZE - index)
            self.canvas.create_text(left - 20, y, text=label, fill=TEXT_COLOR, font=("TkDefaultFont", 9, "bold"))
            self.canvas.create_text(right + 20, y, text=label, fill=TEXT_COLOR, font=("TkDefaultFont", 9, "bold"))

        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for row, column in star_points:
            x = left + column * self.cell_size
            y = top + row * self.cell_size
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=LINE_COLOR, outline=LINE_COLOR)

        for row, column in training.all_points(BOARD_SIZE):
            value = training.board_value(self.game.board, row, column)
            if value not in {"b", "w"}:
                continue
            x = left + column * self.cell_size
            y = top + row * self.cell_size
            radius = self.cell_size * 0.43
            fill = "#101010" if value == "b" else "#f7f4ed"
            outline = "#050505" if value == "b" else "#686868"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, width=1)

        if self.game.moves:
            last = self.game.moves[-1]["coordinate_sgf"]
            point = self.game.point_from_sgf(last)
            if point is not None:
                row, column = point
                x = left + column * self.cell_size
                y = top + row * self.cell_size
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#d62728", width=2)

    def board_click(self, event):
        if self.engine is None or self.ai_busy or self.game.game_over or self.is_ai_turn():
            return
        column = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)
        if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
            return
        x = self.margin + column * self.cell_size
        y = self.margin + row * self.cell_size
        if abs(event.x - x) > self.cell_size * 0.48 or abs(event.y - y) > self.cell_size * 0.48:
            return
        self.try_apply(self.game.current_color, (row, column), source="human")

    def pass_move(self):
        if self.engine is None or self.ai_busy or self.game.game_over or self.is_ai_turn():
            return
        self.try_apply(self.game.current_color, None, source="human")

    def try_apply(self, color, point, source="human", move_sgf=None):
        try:
            record = self.game.apply(color, point, move_sgf=move_sgf, source=source)
        except Exception as error:
            messagebox.showerror("Illegal move", str(error), parent=self.root)
            return False
        self.move_list.insert("end", f"{record['move_number']:>3}. {record['color']} {record['coordinate_human']}")
        self.move_list.see("end")
        self.candidate_list.delete(0, "end")
        self.selected_candidate = None
        self.draw_board()
        self.refresh_controls()
        if self.game.game_over:
            self.status_var.set("Game over — both players passed")
        elif self.game.move_number > self.arguments.max_moves:
            self.game.game_over = True
            self.status_var.set(f"Game stopped at move limit {self.arguments.max_moves}")
        elif self.is_ai_turn():
            self.root.after(120, self.request_ai_move)
        return True

    def request_ai_move(self):
        if self.engine is None or self.ai_busy or self.game.game_over or not self.is_ai_turn():
            return
        self.ai_busy = True
        self.refresh_controls()
        self.status_var.set("AI is considering a move...")
        position = training.board_string(self.game.board)
        color = self.game.current_color.upper()
        move_number = self.game.move_number
        stage = training.stage_for(move_number, max(self.arguments.max_moves, 1))
        start_hash = training.board_hash(self.game.board)

        def worker():
            try:
                result = self.engine.choose_move(
                    position,
                    color,
                    move_number=move_number,
                    stage=stage,
                    candidate_limit=self.arguments.candidate_limit,
                    principle_codes=self.principles,
                )
                payload = {"result": result, "start_hash": start_hash, "color": color}
            except Exception as error:
                payload = {"error": str(error), "start_hash": start_hash, "color": color}
            self.root.after(0, lambda: self.finish_ai_move(payload))

        threading.Thread(target=worker, daemon=True).start()

    def finish_ai_move(self, payload):
        self.ai_busy = False
        if payload.get("error"):
            self.refresh_controls()
            messagebox.showerror("AI move error", payload["error"], parent=self.root)
            return
        if training.board_hash(self.game.board) != payload["start_hash"] or self.game.current_color != payload["color"].lower():
            self.refresh_controls()
            return
        result = payload["result"]
        self.last_analysis = result
        self.populate_candidates(result.get("candidates", []))
        selected = None
        for candidate in result.get("candidates", []):
            try:
                point = self.game.legal_point(candidate["move_sgf"])
                selected = (candidate, point)
                break
            except Exception:
                continue
        if selected is None:
            self.refresh_controls()
            messagebox.showerror("AI move error", "No legal non-repeating candidate was available", parent=self.root)
            return
        candidate, point = selected
        self.selected_candidate = candidate
        self.try_apply(self.game.current_color, point, source="ai", move_sgf=candidate["move_sgf"])

    def populate_candidates(self, candidates):
        self.candidate_list.delete(0, "end")
        for index, candidate in enumerate(candidates):
            text = (
                f"{index + 1:>2}. {candidate['move_human']:<5} "
                f"final {candidate['final_score']:+.3f}  "
                f"align {candidate['alignment_score']:+.3f}  "
                f"Y/Y {candidate['yin_yang_score']:+.3f}"
            )
            self.candidate_list.insert("end", text)

    def select_candidate(self, event=None):
        if self.last_analysis is None:
            return
        selection = self.candidate_list.curselection()
        if not selection:
            return
        index = selection[0]
        candidates = self.last_analysis.get("candidates", [])
        if index < len(candidates):
            self.selected_candidate = candidates[index]
        self.refresh_controls()

    def play_selected_candidate(self):
        if not self.selected_candidate or self.game.game_over:
            return
        candidate = self.selected_candidate
        try:
            point = self.game.legal_point(candidate["move_sgf"])
        except Exception as error:
            messagebox.showerror("Illegal candidate", str(error), parent=self.root)
            return
        self.try_apply(self.game.current_color, point, source="selected_ai", move_sgf=candidate["move_sgf"])

    def resign(self):
        if self.game.game_over:
            return
        color = "Black" if self.game.current_color == "b" else "White"
        if not messagebox.askyesno("Resign", f"Resign as {color}?", parent=self.root):
            return
        self.game.game_over = True
        self.status_var.set(f"{color} resigned")
        self.refresh_controls()

    def new_game(self):
        if not messagebox.askyesno("New game", "Start a new game?", parent=self.root):
            return
        self.game = GoGame()
        self.last_analysis = None
        self.selected_candidate = None
        self.move_list.delete(0, "end")
        self.candidate_list.delete(0, "end")
        self.draw_board()
        self.refresh_controls()
        if self.is_ai_turn():
            self.root.after(250, self.request_ai_move)

    def sgf_text(self):
        nodes = []
        for record in self.game.moves:
            coordinate = "" if record["coordinate_sgf"] == "pass" else record["coordinate_sgf"]
            nodes.append(f";{record['color']}[{coordinate}]")
        return "(;GM[1]FF[4]CA[UTF-8]SZ[19]PB[Principle AI]PW[Human]RU[Chinese]" + "".join(nodes) + ")"

    def save_sgf(self):
        path = self.arguments.output_sgf
        if not path:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save Go game",
                defaultextension=".sgf",
                filetypes=[("SGF files", "*.sgf"), ("All files", "*.*")],
            )
        if not path:
            return
        Path(path).write_text(self.sgf_text(), encoding="utf-8")
        if self.last_analysis is not None and self.arguments.output_json:
            Path(self.arguments.output_json).write_text(json.dumps(self.last_analysis, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved SGF: {path}")

    def close(self):
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Play graphical 19x19 Go using the principle-guided V1 move selector.")
    parser.add_argument("--checkpoint", default="checkpoints/alignment_model.pt")
    parser.add_argument("--mode", choices=["human_vs_ai", "ai_vs_ai"], default="human_vs_ai")
    parser.add_argument("--human-color", choices=["B", "W", "b", "w", "random"], default="random")
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--principles", default=None)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--output-sgf", default=None)
    parser.add_argument("--output-json", default=None)
    arguments = parser.parse_args()
    arguments.human_color_mode = arguments.human_color.lower()
    arguments.human_color = resolve_human_color(arguments.human_color)
    root = tk.Tk()
    GoInterface(root, arguments)
    root.mainloop()


if __name__ == "__main__":
    main()
