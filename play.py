import argparse
import json
import queue
import random
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import move
import train as training

BOARD_SIZE = training.BOARD_SIZE
BOARD_COLOR = '#d8a24a'
LINE_COLOR = '#4c3219'
PANEL_COLOR = '#f3ead7'
ACCENT_COLOR = '#8f1d2c'
TEXT_COLOR = '#241b16'
COLOR_NAMES = {move.BLACK: 'Black', move.WHITE: 'White'}
COLOR_LETTERS = {move.BLACK: 'B', move.WHITE: 'W'}

class GoGame:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_color = move.BLACK
        self.move_number = 1
        self.consecutive_passes = 0
        self.position_history = {self.board.tobytes()}
        self.moves = []
        self.game_over = False
        self.result = None

    def check(self, point):
        if self.board[point] != move.EMPTY: raise ValueError('That point is already occupied')
        outcome = move.play_stone(self.board, point[0], point[1], self.current_color)
        if outcome is None: raise ValueError('That move would be suicide')
        if outcome[0].tobytes() in self.position_history: raise ValueError('That move repeats a previous board position')
        return outcome

    def apply(self, color, point, source='human'):
        if self.game_over: raise ValueError('The game is already over')
        if color != self.current_color: raise ValueError("It is not this color's turn")
        if point is None: after, captured = self.board.copy(), []
        else: after, captured = self.check(point)
        self.consecutive_passes = self.consecutive_passes + 1 if point is None else 0
        self.board = after
        self.position_history.add(after.tobytes())
        self.moves.append({'move_number': self.move_number, 'color': COLOR_LETTERS[color], 'color_value': color, 'point': point, 'coordinate_sgf': move.coordinate_sgf(point), 'coordinate_human': move.coordinate_human(point), 'captured_count': len(captured), 'captured_coordinates': [move.coordinate_sgf(value) for value in captured], 'source': source})
        self.move_number += 1
        self.current_color = 3 - color
        if self.consecutive_passes >= 2: self.game_over = True
        return self.moves[-1]

    def history(self): return [(record['color_value'], record['coordinate_sgf']) for record in self.moves]

def resolve_human_color(value):
    color = str(value).lower()
    if color == 'random': return random.choice([move.BLACK, move.WHITE])
    if color not in {'b', 'w'}: raise ValueError('Human color must be B, W, or random')
    return move.BLACK if color == 'b' else move.WHITE

def parse_principles(value):
    if not value: return None
    return [item.strip() for item in value.split(',') if item.strip()]

class GoInterface:
    def __init__(self, root, arguments):
        self.root = root
        self.arguments = arguments
        self.game = GoGame()
        self.game_id = 0
        self.engine = None
        self.ai_busy = False
        self.results = queue.Queue()
        self.last_analysis = None
        self.analysis_key = None
        self.selected_candidate = None
        self.cell_size = 34
        self.margin = 38
        self.canvas_size = self.margin * 2 + self.cell_size * (BOARD_SIZE - 1)
        self.principles = parse_principles(arguments.principles)
        self.setup_window()
        self.draw_board()
        self.root.after(50, self.load_engine)

    def setup_window(self):
        self.root.title('Principle-Guided Go')
        self.root.configure(bg=PANEL_COLOR)
        self.root.minsize(1050, 820)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill='both', expand=True)
        board_frame = ttk.Frame(outer)
        board_frame.pack(side='left', fill='both', expand=False)
        self.canvas = tk.Canvas(board_frame, width=self.canvas_size, height=self.canvas_size, background=BOARD_COLOR, highlightthickness=1, highlightbackground=LINE_COLOR)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.board_click)
        side = ttk.Frame(outer, padding=(18, 0, 0, 0))
        side.pack(side='left', fill='both', expand=True)
        ttk.Label(side, text='Principle-Guided Go', font=('TkDefaultFont', 17, 'bold')).pack(anchor='w', pady=(0, 12))
        self.status_var = tk.StringVar(value='Loading model...')
        ttk.Label(side, textvariable=self.status_var, wraplength=330).pack(anchor='w', fill='x', pady=(0, 12))
        controls = ttk.LabelFrame(side, text='Game')
        controls.pack(fill='x', pady=(0, 12))
        self.turn_var = tk.StringVar(value='Turn: Black')
        ttk.Label(controls, textvariable=self.turn_var).pack(anchor='w', padx=8, pady=(8, 4))
        self.move_var = tk.StringVar(value='Move: 1')
        ttk.Label(controls, textvariable=self.move_var).pack(anchor='w', padx=8, pady=2)
        button_row = ttk.Frame(controls)
        button_row.pack(fill='x', padx=8, pady=8)
        self.pass_button = ttk.Button(button_row, text='Pass', command=self.pass_move)
        self.pass_button.pack(side='left', padx=(0, 5))
        self.resign_button = ttk.Button(button_row, text='Resign', command=self.resign)
        self.resign_button.pack(side='left', padx=5)
        self.new_button = ttk.Button(button_row, text='New Game', command=self.new_game)
        self.new_button.pack(side='left', padx=5)
        analysis_frame = ttk.LabelFrame(side, text='AI candidate analysis')
        analysis_frame.pack(fill='both', expand=True, pady=(0, 8))
        self.intent_var = tk.StringVar(value='')
        ttk.Label(analysis_frame, textvariable=self.intent_var, wraplength=330).pack(anchor='w', padx=8, pady=(8, 0))
        list_row = ttk.Frame(analysis_frame)
        list_row.pack(fill='both', expand=True)
        self.candidate_list = tk.Listbox(list_row, height=12, width=46, font=('TkFixedFont', 10), exportselection=False)
        self.candidate_list.pack(side='left', fill='both', expand=True, padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(list_row, orient='vertical', command=self.candidate_list.yview)
        scrollbar.pack(side='right', fill='y', padx=8, pady=8)
        self.candidate_list.configure(yscrollcommand=scrollbar.set)
        self.candidate_list.bind('<<ListboxSelect>>', self.select_candidate)
        self.detail_var = tk.StringVar(value='')
        ttk.Label(analysis_frame, textvariable=self.detail_var, wraplength=330).pack(anchor='w', padx=8, pady=(0, 8))
        action_row = ttk.Frame(side)
        action_row.pack(fill='x', pady=(0, 8))
        self.analyze_button = ttk.Button(action_row, text='Analyze Position', command=lambda: self.request_engine(False))
        self.analyze_button.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.play_candidate_button = ttk.Button(action_row, text='Play Selected Candidate', command=self.play_selected_candidate)
        self.play_candidate_button.pack(side='left', fill='x', expand=True, padx=(4, 0))
        moves_frame = ttk.LabelFrame(side, text='Move history')
        moves_frame.pack(fill='both', expand=True, pady=(0, 12))
        self.move_list = tk.Listbox(moves_frame, height=8, width=46, font=('TkFixedFont', 10))
        self.move_list.pack(side='left', fill='both', expand=True, padx=(8, 0), pady=8)
        move_scrollbar = ttk.Scrollbar(moves_frame, orient='vertical', command=self.move_list.yview)
        move_scrollbar.pack(side='right', fill='y', padx=8, pady=8)
        self.move_list.configure(yscrollcommand=move_scrollbar.set)
        self.save_button = ttk.Button(side, text='Save SGF', command=self.save_sgf)
        self.save_button.pack(fill='x')

    def load_engine(self):
        try:
            parameters = json.loads(Path(self.arguments.parameters).read_text(encoding='utf-8')) if self.arguments.parameters else None
            self.engine = move.MoveEngine(self.arguments.weights, self.arguments.config, self.arguments.vocab, parameters)
            self.status_var.set(self.ready_status())
            self.refresh_controls()
            if self.is_ai_turn(): self.root.after(250, lambda: self.request_engine(True))
        except Exception as error:
            self.status_var.set('Model could not be loaded')
            messagebox.showerror('Model loading error', str(error), parent=self.root)
            self.refresh_controls()

    def ready_status(self):
        if self.arguments.mode == 'ai_vs_ai': return 'AI vs AI ready'
        return f'Your color: {COLOR_NAMES[self.arguments.human_color]}'

    def is_ai_turn(self):
        if self.game.game_over: return False
        if self.arguments.mode == 'ai_vs_ai': return True
        return self.game.current_color != self.arguments.human_color

    def position_key(self): return (self.game.board.tobytes(), self.game.current_color)

    def refresh_controls(self):
        color_name = COLOR_NAMES[self.game.current_color]
        self.turn_var.set(f'Turn: {color_name}')
        self.move_var.set(f'Move: {self.game.move_number}')
        if not self.ai_busy and self.engine is not None and not self.game.game_over: self.status_var.set(f'{self.ready_status()} — {color_name} to play')
        human_turn = self.engine is not None and not self.ai_busy and not self.game.game_over and not self.is_ai_turn()
        self.pass_button.configure(state='normal' if human_turn else 'disabled')
        self.resign_button.configure(state='normal' if human_turn else 'disabled')
        self.analyze_button.configure(state='normal' if human_turn else 'disabled')
        self.play_candidate_button.configure(state='normal' if human_turn and self.selected_candidate and self.analysis_key == self.position_key() else 'disabled')

    def draw_board(self):
        self.canvas.delete('all')
        left = top = self.margin
        right = bottom = self.margin + self.cell_size * (BOARD_SIZE - 1)
        for index in range(BOARD_SIZE):
            x = left + index * self.cell_size
            y = top + index * self.cell_size
            self.canvas.create_line(x, top, x, bottom, fill=LINE_COLOR)
            self.canvas.create_line(left, y, right, y, fill=LINE_COLOR)
            self.canvas.create_text(x, top - 20, text=move.COLUMN_LETTERS[index], fill=TEXT_COLOR, font=('TkDefaultFont', 9, 'bold'))
            self.canvas.create_text(x, bottom + 20, text=move.COLUMN_LETTERS[index], fill=TEXT_COLOR, font=('TkDefaultFont', 9, 'bold'))
            self.canvas.create_text(left - 20, y, text=str(BOARD_SIZE - index), fill=TEXT_COLOR, font=('TkDefaultFont', 9, 'bold'))
            self.canvas.create_text(right + 20, y, text=str(BOARD_SIZE - index), fill=TEXT_COLOR, font=('TkDefaultFont', 9, 'bold'))
        for row, column in [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]:
            x = left + column * self.cell_size
            y = top + row * self.cell_size
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=LINE_COLOR, outline=LINE_COLOR)
        for row, column in np.argwhere(self.game.board != move.EMPTY).tolist():
            x = left + column * self.cell_size
            y = top + row * self.cell_size
            radius = self.cell_size * 0.43
            black = self.game.board[row, column] == move.BLACK
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='#101010' if black else '#f7f4ed', outline='#050505' if black else '#686868', width=1)
        if self.game.moves and self.game.moves[-1]['point'] is not None:
            row, column = self.game.moves[-1]['point']
            x = left + column * self.cell_size
            y = top + row * self.cell_size
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline='#d62728', width=2)

    def board_click(self, event):
        if self.engine is None or self.ai_busy or self.game.game_over or self.is_ai_turn(): return
        column = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)
        if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE): return
        x = self.margin + column * self.cell_size
        y = self.margin + row * self.cell_size
        if abs(event.x - x) > self.cell_size * 0.48 or abs(event.y - y) > self.cell_size * 0.48: return
        self.try_apply(self.game.current_color, (row, column), source='human')

    def pass_move(self):
        if self.engine is None or self.ai_busy or self.game.game_over or self.is_ai_turn(): return
        self.try_apply(self.game.current_color, None, source='human')

    def try_apply(self, color, point, source='human'):
        try: record = self.game.apply(color, point, source=source)
        except ValueError as error:
            messagebox.showerror('Illegal move', str(error), parent=self.root)
            return False
        captures = f"  x{record['captured_count']}" if record['captured_count'] else ''
        self.move_list.insert('end', f"{record['move_number']:>3}. {record['color']} {record['coordinate_human']}{captures}")
        self.move_list.see('end')
        if source != 'ai': self.clear_analysis()
        self.draw_board()
        self.refresh_controls()
        if self.game.game_over: self.status_var.set('Game over — both players passed')
        elif self.game.move_number > self.arguments.max_moves:
            self.game.game_over = True
            self.refresh_controls()
            self.status_var.set(f'Game stopped at move limit {self.arguments.max_moves}')
        elif self.is_ai_turn(): self.root.after(120, lambda: self.request_engine(True))
        return True

    def request_engine(self, play):
        if self.engine is None or self.ai_busy or self.game.game_over: return
        if play and not self.is_ai_turn(): return
        if not play and self.is_ai_turn(): return
        self.ai_busy = True
        self.refresh_controls()
        self.status_var.set('AI is considering a move...' if play else 'Analyzing position...')
        board = self.game.board.copy()
        color = self.game.current_color
        move_number = self.game.move_number
        history = self.game.history()
        key = self.position_key()
        game_id = self.game_id

        def worker():
            try: payload = {'result': self.engine.choose_move(board, color, move_number=move_number, history=history, candidate_limit=self.arguments.candidate_limit, openings=self.arguments.openings, principle_codes=self.principles)}
            except Exception as error: payload = {'error': str(error)}
            payload.update({'play': play, 'key': key, 'game_id': game_id})
            self.results.put(payload)

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self.poll_engine)

    def poll_engine(self):
        try: payload = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll_engine)
            return
        self.finish_engine(payload)

    def finish_engine(self, payload):
        self.ai_busy = False
        if payload['game_id'] != self.game_id or payload['key'] != self.position_key():
            self.refresh_controls()
            if self.is_ai_turn(): self.root.after(120, lambda: self.request_engine(True))
            return
        if payload.get('error'):
            self.refresh_controls()
            messagebox.showerror('AI move error', payload['error'], parent=self.root)
            return
        result = payload['result']
        self.last_analysis = result
        self.analysis_key = payload['key']
        self.populate_candidates(result['candidates'], result['intent'])
        if not payload['play']:
            self.refresh_controls()
            self.status_var.set('Analysis ready — select a candidate to play it')
            return
        selected = None
        for candidate in result['candidates']:
            point = training.coordinate_to_point(candidate['move_sgf'])
            try:
                if point is not None: self.game.check(point)
            except ValueError: continue
            selected = (candidate, point)
            break
        if selected is None:
            reason = {'settled': 'no contested points left', 'no_legal_move': 'no legal move'}.get(result.get('pass_reason'), 'no legal non-repeating candidate')
            if self.try_apply(self.game.current_color, None, source='ai'): self.status_var.set(f'AI passed — {reason}')
            return
        self.show_candidate(selected[0])
        self.try_apply(self.game.current_color, selected[1], source='ai')

    def clear_analysis(self):
        self.candidate_list.delete(0, 'end')
        self.selected_candidate = None
        self.analysis_key = None
        self.intent_var.set('')
        self.detail_var.set('')

    def populate_candidates(self, candidates, intent):
        self.candidate_list.delete(0, 'end')
        for index, candidate in enumerate(candidates): self.candidate_list.insert('end', f"{index + 1:>2}. {candidate['move_human']:<5} F{candidate['final_score']:+.3f} A{candidate['alignment_score']:+.3f} Y{candidate['yin_yang_score']:+.3f}")
        if intent is None: self.intent_var.set('Intent: not enough opponent moves yet')
        else: self.intent_var.set(f"Intent: feint probability {intent['probability']:.2f} (concentration {intent['concentration']:.2f}, neglected {intent['neglect_own']:.2f}, shared liberties {intent['shared_liberties']:.2f})")
        self.selected_candidate = None
        self.detail_var.set('')

    def show_candidate(self, candidate):
        self.selected_candidate = candidate
        components = candidate['components']
        captures = f" · captures {candidate['captured_count']}" if candidate['captured_count'] else ''
        self.detail_var.set(f"{candidate['move_human']}: static {components['static']:+.3f}  pair {components['pair']:+.3f}  intent {components['intent']:+.3f}  R {candidate['yin_yang_relevance']:.2f}{captures}")

    def select_candidate(self, event=None):
        if self.last_analysis is None: return
        selection = self.candidate_list.curselection()
        if not selection: return
        candidates = self.last_analysis['candidates']
        if selection[0] < len(candidates): self.show_candidate(candidates[selection[0]])
        self.refresh_controls()

    def play_selected_candidate(self):
        if not self.selected_candidate or self.game.game_over or self.is_ai_turn() or self.analysis_key != self.position_key(): return
        self.try_apply(self.game.current_color, training.coordinate_to_point(self.selected_candidate['move_sgf']), source='selected_ai')

    def resign(self):
        if self.game.game_over: return
        color_name = COLOR_NAMES[self.game.current_color]
        if not messagebox.askyesno('Resign', f'Resign as {color_name}?', parent=self.root): return
        self.game.game_over = True
        self.game.result = 'W+R' if self.game.current_color == move.BLACK else 'B+R'
        self.refresh_controls()
        self.status_var.set(f'{color_name} resigned')

    def new_game(self):
        if not messagebox.askyesno('New game', 'Start a new game?', parent=self.root): return
        self.game_id += 1
        self.game = GoGame()
        self.last_analysis = None
        self.move_list.delete(0, 'end')
        self.clear_analysis()
        self.draw_board()
        self.refresh_controls()
        if self.is_ai_turn(): self.root.after(250, lambda: self.request_engine(True))

    def player_names(self):
        if self.arguments.mode == 'ai_vs_ai': return 'Principle AI', 'Principle AI'
        return ('Human', 'Principle AI') if self.arguments.human_color == move.BLACK else ('Principle AI', 'Human')

    def sgf_text(self):
        black, white = self.player_names()
        nodes = ''.join(f";{record['color']}[{'' if record['coordinate_sgf'] == 'pass' else record['coordinate_sgf']}]" for record in self.game.moves)
        result = f'RE[{self.game.result}]' if self.game.result else ''
        return f'(;GM[1]FF[4]CA[UTF-8]SZ[{BOARD_SIZE}]PB[{black}]PW[{white}]RU[Chinese]{result}{nodes})'

    def save_sgf(self):
        path = self.arguments.output_sgf or filedialog.asksaveasfilename(parent=self.root, title='Save Go game', defaultextension='.sgf', filetypes=[('SGF files', '*.sgf'), ('All files', '*.*')])
        if not path: return
        Path(path).write_text(self.sgf_text(), encoding='utf-8')
        if self.last_analysis is not None and self.arguments.output_json: Path(self.arguments.output_json).write_text(json.dumps(self.last_analysis, indent=2), encoding='utf-8')
        self.status_var.set(f'Saved SGF: {path}')

    def close(self): self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description='Play graphical 19x19 Go using the principle-guided V2 move selector.')
    parser.add_argument('--weights', default='model_weights.npz')
    parser.add_argument('--config', default='model_config.json')
    parser.add_argument('--vocab', default='principle_vocab.json')
    parser.add_argument('--parameters', default=None)
    parser.add_argument('--mode', choices=['human_vs_ai', 'ai_vs_ai'], default='human_vs_ai')
    parser.add_argument('--human-color', choices=['B', 'W', 'b', 'w', 'random'], default='random')
    parser.add_argument('--candidate-limit', type=int, default=24)
    parser.add_argument('--openings', type=int, default=4)
    parser.add_argument('--principles', default=None)
    parser.add_argument('--max-moves', type=int, default=300)
    parser.add_argument('--output-sgf', default=None)
    parser.add_argument('--output-json', default=None)
    arguments = parser.parse_args()
    arguments.human_color = resolve_human_color(arguments.human_color)
    root = tk.Tk()
    GoInterface(root, arguments)
    root.mainloop()

if __name__ == '__main__':
    main()