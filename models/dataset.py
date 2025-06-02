import os
import numpy as np
import time
try:
    from .utils import board_to_tensor, result_to_value
except ImportError:
    from utils import board_to_tensor, result_to_value
import torch
import torch.nn as nn
from torch.utils.data import Dataset  # Add Dataset
from torch import zeros
import chess
import chess.pgn
from tqdm import tqdm
import io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
DATA_DIR = "data"

def load_pgn(file_path, max_games=None):
    games = []
    try:
        with open(file_path, 'r', encoding='utf-8') as pgn_file:
            game_count = 0
            while max_games is None or game_count < max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
                game_count += 1
    except Exception as e:
        print(f"Error loading: {file_path}: {e}")
    return games

def load_lichess_dataset(data_dir, max_files=None, max_games_per_file=None):
    files = [file for file in os.listdir(data_dir) if file.endswith(".pgn")]

    if max_files is not None:
        files = files[:min(len(files), max_files)]

    all_games = []
    for file in tqdm(files, desc="Loading PGN files"):
        file_path = os.path.join(data_dir, file)
        games = load_pgn(file_path, max_games_per_file)
        all_games.extend(games)
        print(f"Loaded {len(games)} games from {file}. Total games: {len(all_games)}")

    return all_games

class ChessDataset(Dataset):
    def __init__(self, data_dir, max_files=None, max_games_per_file=None, max_positions=None):
        self.positions = []
        self.labels = []

        games = load_lichess_dataset(data_dir, max_files, max_games_per_file)
        positions_added = 0

        print("Extracting positions from games...")
        for game in tqdm(games):
            result = game.headers.get("Result", "*")
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                if board.fullmove_number < 5:
                    continue

                position_tensor = board_to_tensor(board)
                label = result_to_value(result, board.turn)

                self.positions.append(position_tensor)
                self.labels.append(torch.tensor([label], dtype=torch.float32))

                positions_added += 1
                if max_positions and positions_added >= max_positions:
                    break

            if max_positions and positions_added >= max_positions:
                break

        print(f"Dataset created with {len(self.positions)} positions")

    def __len__(self):  # Properly indented as part of ChessDataset
        return len(self.positions)

    def __getitem__(self, index):  # Properly indented as part of ChessDataset
        return self.positions[index], self.labels[index]
