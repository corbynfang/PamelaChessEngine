import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import random
import time

from models import ChessAI
from utils import board_to_tensor, result_to_value

class SimpleChessDataset(Dataset):
    def __init__(self, pgn_path, max_games=500, stockfish_path=None):
        self.positions = []
        self.evaluations = []

        print(f"🔍 Loading games from: {pgn_path}")
        self._load_games_simple(pgn_path, max_games, stockfish_path)
        print(f"✅ Loaded {len(self.positions)} positions")

    def _load_games_simple(self, pgn_path, max_games, stockfish_path):
        """Simple, reliable game loading"""
        pgn_files = self._get_pgn_files(pgn_path)
        if not pgn_files:
            print("❌ No PGN files found!")
            return

        # Initialize Stockfish once
        engine = None
        if stockfish_path and os.path.exists(stockfish_path):
            try:
                print(f"🔧 Starting Stockfish: {stockfish_path}")
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print("✅ Stockfish ready")
            except Exception as e:
                print(f"⚠️ Stockfish failed: {e}")

        total_games = 0

        for pgn_file in pgn_files:
            if total_games >= max_games:
                break

            print(f"📖 Processing: {os.path.basename(pgn_file)}")
            file_games = self._process_single_file(pgn_file, max_games - total_games, engine)
            total_games += file_games

            print(f"   ✅ Loaded {file_games} games, total: {total_games}")

        if engine:
            try:
                engine.quit()
            except:
                pass

    def _process_single_file(self, pgn_file, max_games, engine):
        """Process one PGN file reliably"""
        games_loaded = 0

        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                while games_loaded < max_games:
                    try:
                        game = chess.pgn.read_game(f)
                        if not game:
                            break  # End of file

                        # Quick validation
                        result = game.headers.get("Result", "*")
                        if result == "*":
                            continue

                        moves = list(game.mainline_moves())
                        if len(moves) < 15:
                            continue

                        # Extract positions from this game
                        positions_added = self._extract_from_game(game, engine)
                        if positions_added > 0:
                            games_loaded += 1

                        # Progress update
                        if games_loaded % 50 == 0:
                            print(f"      Games: {games_loaded}, Positions: {len(self.positions)}")

                    except Exception as e:
                        # Skip problematic games
                        continue

        except Exception as e:
            print(f"❌ Error reading {pgn_file}: {e}")

        return games_loaded

    def _extract_from_game(self, game, engine):
        """Extract training positions from one game"""
        board = game.board()
        moves = list(game.mainline_moves())
        result = game.headers.get("Result")

        positions_added = 0

        # Sample positions every 4 moves
        for move_idx in range(0, len(moves), 4):
            try:
                # Play moves to get to position
                temp_board = game.board()
                for i in range(move_idx):
                    temp_board.push(moves[i])

                # Get evaluation
                if engine and move_idx % 8 == 0:  # Use engine sparingly
                    evaluation = self._get_engine_eval(temp_board, engine)
                else:
                    evaluation = self._get_result_eval(result, temp_board)

                # Add to dataset
                position_tensor = board_to_tensor(temp_board)
                self.positions.append(position_tensor)
                self.evaluations.append(evaluation)
                positions_added += 1

                # Don't take too many positions per game
                if positions_added >= 8:
                    break

            except Exception as e:
                continue

        return positions_added

    def _get_engine_eval(self, board, engine):
        """Get Stockfish evaluation with timeout"""
        try:
            # Quick evaluation with timeout
            info = engine.analyse(board, chess.engine.Limit(time=0.1, depth=8))
            score = info["score"].white()

            if score.is_mate():
                return 1.0 if score.mate() > 0 else -1.0
            else:
                cp = score.score()
                return max(-1.0, min(1.0, cp / 400.0))

        except Exception as e:
            return 0.0

    def _get_result_eval(self, result, board):
        """Simple evaluation based on game result"""
        base_eval = result_to_value(result, board.turn == chess.WHITE)
        return base_eval * 0.7  # Lower confidence

    def _get_pgn_files(self, pgn_path):
        """Get list of PGN files"""
        if os.path.isfile(pgn_path) and pgn_path.endswith('.pgn'):
            return [pgn_path]
        elif os.path.isdir(pgn_path):
            files = []
            for f in os.listdir(pgn_path):
                if f.endswith('.pgn'):
                    files.append(os.path.join(pgn_path, f))
            return files
        return []

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx].float(), torch.tensor(self.evaluations[idx], dtype=torch.float32)


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()

        # Better optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.003,
            weight_decay=0.01
        )

        # Fixed scheduler - remove verbose
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.7
            # Removed verbose=True
        )

        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for positions, targets in tqdm(train_loader, desc="Training"):
            positions = positions.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(positions).squeeze()
            loss = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for positions, targets in val_loader:
                positions = positions.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(positions).squeeze()
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, save_path):
        print(f"🚀 Training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # Manual verbose output
            if new_lr != old_lr:
                print(f"🔄 Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

            print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={new_lr:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(save_path)
                print(f"✅ Best model saved!")

    def _save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)


def train_chess_ai(data_path="pgn", epochs=30, max_games=500, stockfish_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Device: {device}")

    # Load dataset
    print("📚 Creating dataset...")
    dataset = SimpleChessDataset(data_path, max_games, stockfish_path)

    if len(dataset) == 0:
        print("❌ No data loaded!")
        return None

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders with better settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Larger batch size
        shuffle=True,
        drop_last=True  # Consistent batch sizes
    )
    val_loader = DataLoader(val_dataset, batch_size=64)

    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model and trainer
    model = ChessAI()  # Fresh model
    trainer = Trainer(model, device)  # Use fixed Trainer

    # Train
    save_path = "trained_models/chess_ai.pth"
    trainer.train(train_loader, val_loader, epochs, save_path)

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Simple Chess AI Training")
    parser.add_argument("--data", default="pgn", help="PGN directory or file path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--max_games", type=int, default=500, help="Maximum games to process")
    parser.add_argument("--stockfish", help="Path to Stockfish executable")

    args = parser.parse_args()

    print("🧠 Simple Chess AI Training")
    print("=" * 40)

    model_path = train_chess_ai(
        data_path=args.data,
        epochs=args.epochs,
        max_games=args.max_games,
        stockfish_path=args.stockfish
    )

    if model_path:
        print(f"🎉 Model saved to: {model_path}")
    else:
        print("❌ Training failed")


if __name__ == "__main__":
    main()
