import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import chess
import chess.pgn
from pathlib import Path
from tqdm import tqdm
import argparse
import glob
import os
import json
import random
from collections import defaultdict

from models import create_chess_model
from utils import board_to_tensor, result_to_value, chess_knowledge

class SmartChessDataset(Dataset):
    """Dataset that learns real openings and endgames from PGN files"""

    def __init__(self, pgn_path, max_games=3000, min_elo=1200,
                 learn_openings=True, learn_endgames=True, augment_data=True):
        self.positions = []
        self.evaluations = []
        self.sources = []
        self.phase_labels = []  # Track game phase: opening, middlegame, endgame

        # Statistics for analysis
        self.opening_stats = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0})
        self.endgame_stats = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0})

        # Load and analyze PGN files
        self._analyze_pgn_files(pgn_path, max_games, min_elo, learn_openings, learn_endgames)

        # Add learned knowledge to training data
        if learn_openings:
            self._add_learned_openings()

        if learn_endgames:
            self._add_learned_endgames()

        # Data augmentation
        if augment_data and len(self.positions) > 0:
            self._smart_augmentation()

        print(f"🎯 Final dataset: {len(self.positions)} positions")
        self._print_analysis()

    def _get_pgn_files(self, pgn_path):
        """Get all PGN files"""
        pgn_files = []

        if os.path.isfile(pgn_path):
            if pgn_path.endswith('.pgn'):
                pgn_files = [pgn_path]
        elif os.path.isdir(pgn_path):
            for root, dirs, files in os.walk(pgn_path):
                for file in files:
                    if file.endswith('.pgn'):
                        pgn_files.append(os.path.join(root, file))

        return pgn_files

    def _analyze_pgn_files(self, pgn_path, max_games, min_elo, learn_openings, learn_endgames):
        """Analyze PGN files to extract real chess knowledge"""
        print(f"🔍 Analyzing PGN files for chess patterns...")

        pgn_files = self._get_pgn_files(pgn_path)
        if not pgn_files:
            print(f"⚠️  No PGN files found in {pgn_path}")
            return

        games_processed = 0
        games_per_file = max_games // len(pgn_files) if pgn_files else max_games

        for pgn_file in pgn_files:
            print(f"📖 Analyzing {os.path.basename(pgn_file)}...")
            file_games = 0

            try:
                with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                    while file_games < games_per_file and games_processed < max_games:
                        game = chess.pgn.read_game(f)
                        if not game:
                            break

                        if not self._is_quality_game(game, min_elo):
                            continue

                        result = game.headers.get("Result", "*")
                        if result == "*":
                            continue

                        # Analyze this game for patterns
                        self._analyze_single_game(game, result, learn_openings, learn_endgames)

                        file_games += 1
                        games_processed += 1

                        if games_processed % 100 == 0:
                            print(f"   Analyzed {games_processed} games...")

            except Exception as e:
                print(f"❌ Error in {pgn_file}: {e}")

        print(f"✅ Analyzed {games_processed} games")

    def _analyze_single_game(self, game, result, learn_openings, learn_endgames):
        """Extract knowledge from a single game"""
        board = game.board()
        moves = list(game.mainline_moves())

        if len(moves) < 10:
            return

        # Determine game phases
        opening_moves = min(15, len(moves) // 4)
        endgame_start = max(len(moves) - 15, len(moves) * 3 // 4)

        for move_idx, move in enumerate(moves):
            position_before = board.fen()
            board.push(move)
            position_after = board.fen()

            # OPENING ANALYSIS
            if move_idx < opening_moves and learn_openings:
                self._record_opening_pattern(position_before, move, result, move_idx)

                # Add opening positions to training data
                if move_idx >= 3:  # Skip first few moves
                    tensor = board_to_tensor(board)
                    evaluation = self._calculate_opening_evaluation(board, result, move_idx)

                    self.positions.append(tensor)
                    self.evaluations.append(evaluation)
                    self.sources.append('opening_real')
                    self.phase_labels.append('opening')

            # MIDDLEGAME ANALYSIS
            elif opening_moves <= move_idx < endgame_start:
                # Sample some middlegame positions
                if move_idx % 3 == 0:  # Every 3rd move
                    tensor = board_to_tensor(board)
                    evaluation = self._calculate_middlegame_evaluation(board, result, move_idx, len(moves))

                    self.positions.append(tensor)
                    self.evaluations.append(evaluation)
                    self.sources.append('middlegame_real')
                    self.phase_labels.append('middlegame')

            # ENDGAME ANALYSIS
            elif move_idx >= endgame_start and learn_endgames:
                self._record_endgame_pattern(board, result)

                # Add endgame positions to training data
                tensor = board_to_tensor(board)
                evaluation = self._calculate_endgame_evaluation(board, result, move_idx, len(moves))

                self.positions.append(tensor)
                self.evaluations.append(evaluation)
                self.sources.append('endgame_real')
                self.phase_labels.append('endgame')

    def _record_opening_pattern(self, position_fen, move, result, move_idx):
        """Record opening move patterns and their success rates"""
        # Create a simpler position key (first few moves)
        board_temp = chess.Board()
        fen_parts = position_fen.split()

        # Use just piece positions for pattern matching
        position_key = fen_parts[0]  # Just piece positions
        move_key = f"{position_key}:{move.uci()}"

        # Record the outcome
        if result == "1-0":
            if move_idx % 2 == 0:  # White's move
                self.opening_stats[move_key]['wins'] += 1
            else:  # Black's move, white won
                self.opening_stats[move_key]['losses'] += 1
        elif result == "0-1":
            if move_idx % 2 == 0:  # White's move, black won
                self.opening_stats[move_key]['losses'] += 1
            else:  # Black's move
                self.opening_stats[move_key]['wins'] += 1
        else:  # Draw
            self.opening_stats[move_key]['draws'] += 1

        self.opening_stats[move_key]['total'] += 1

    def _record_endgame_pattern(self, board, result):
        """Record endgame patterns and outcomes"""
        # Get material signature
        piece_count = chess.popcount(board.occupied)
        if piece_count > 8:  # Not a true endgame
            return

        # Create material signature
        white_pieces = []
        black_pieces = []

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_char = chess.piece_symbol(piece.piece_type).upper()
                if piece.color == chess.WHITE:
                    white_pieces.append(piece_char)
                else:
                    black_pieces.append(piece_char)

        white_pieces.sort()
        black_pieces.sort()
        material_key = f"{''.join(white_pieces)}_vs_{''.join(black_pieces)}"

        # Record outcome
        if result == "1-0":
            self.endgame_stats[material_key]['wins'] += 1
        elif result == "0-1":
            self.endgame_stats[material_key]['losses'] += 1
        else:
            self.endgame_stats[material_key]['draws'] += 1

        self.endgame_stats[material_key]['total'] += 1

    def _calculate_opening_evaluation(self, board, result, move_idx):
        """Calculate evaluation for opening positions"""
        base_eval = result_to_value(result, board.turn == chess.WHITE)

        # Opening positions should have lower confidence initially
        confidence = min(0.7, move_idx / 15.0)

        # Add some opening-specific bonuses
        opening_bonus = 0.0

        # Bonus for piece development
        developed_pieces = 0
        for square in [chess.B1, chess.C1, chess.F1, chess.G1]:  # White pieces
            piece = board.piece_at(square)
            if not piece or piece.piece_type == chess.KING:
                developed_pieces += 1

        for square in [chess.B8, chess.C8, chess.F8, chess.G8]:  # Black pieces
            piece = board.piece_at(square)
            if not piece or piece.piece_type == chess.KING:
                developed_pieces += 1

        opening_bonus += (developed_pieces / 8.0) * 0.1

        # Bonus for center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        center_control = 0
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                center_control += 1

        opening_bonus += (center_control / 4.0) * 0.1

        final_eval = base_eval * confidence + opening_bonus
        return max(-1.0, min(1.0, final_eval))

    def _calculate_middlegame_evaluation(self, board, result, move_idx, total_moves):
        """Calculate evaluation for middlegame positions"""
        base_eval = result_to_value(result, board.turn == chess.WHITE)

        # Full confidence in middlegame
        confidence = 1.0

        # Game progress adjustment
        progress = move_idx / total_moves
        time_bonus = (1.0 - progress) * 0.1  # Slight bonus for earlier positions

        final_eval = base_eval * confidence + time_bonus
        return max(-1.0, min(1.0, final_eval))

    def _calculate_endgame_evaluation(self, board, result, move_idx, total_moves):
        """Calculate evaluation for endgame positions"""
        base_eval = result_to_value(result, board.turn == chess.WHITE)

        # High confidence in endgame positions
        confidence = 1.0

        # Distance to end bonus/penalty
        moves_from_end = total_moves - move_idx
        if moves_from_end < 10:  # Very close to end
            confidence = 1.2  # Higher confidence

        # Material advantage in endgame is more important
        material_bonus = self._calculate_material_advantage(board) * 0.2

        final_eval = base_eval * confidence + material_bonus
        return max(-1.0, min(1.0, final_eval))

    def _calculate_material_advantage(self, board):
        """Calculate material advantage"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }

        white_material = black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        material_diff = white_material - black_material
        return max(-1.0, min(1.0, material_diff / 10.0))

    def _add_learned_openings(self):
        """Add high-success opening patterns to training data"""
        print("📚 Adding successful opening patterns...")

        added_count = 0
        min_games = 5  # Minimum games to consider a pattern

        for move_pattern, stats in self.opening_stats.items():
            if stats['total'] < min_games:
                continue

            # Calculate success rate
            win_rate = stats['wins'] / stats['total']
            draw_rate = stats['draws'] / stats['total']

            # Skip poor patterns
            if win_rate < 0.3:
                continue

            try:
                # Reconstruct position and move
                position_key, move_uci = move_pattern.split(':')
                board = chess.Board()
                board.set_board_fen(position_key + " w - - 0 1")  # Assume white to move

                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)

                    tensor = board_to_tensor(board)
                    # Evaluation based on success rate
                    evaluation = (win_rate - 0.5) * 2.0 + draw_rate * 0.1
                    evaluation = max(-0.5, min(0.5, evaluation))

                    self.positions.append(tensor)
                    self.evaluations.append(evaluation)
                    self.sources.append('opening_learned')
                    self.phase_labels.append('opening')

                    added_count += 1

                    if added_count >= 500:  # Limit learned openings
                        break

            except:
                continue

        print(f"✅ Added {added_count} learned opening patterns")

    def _add_learned_endgames(self):
        """Add endgame patterns based on actual game outcomes"""
        print("🏁 Adding learned endgame patterns...")

        added_count = 0
        min_games = 3

        for material_pattern, stats in self.endgame_stats.items():
            if stats['total'] < min_games:
                continue

            # Calculate pattern success
            win_rate = stats['wins'] / stats['total']
            draw_rate = stats['draws'] / stats['total']

            # Generate training positions for this material pattern
            try:
                white_pieces, black_pieces = material_pattern.split('_vs_')
                white_list = list(white_pieces)
                black_list = list(black_pieces)

                # Generate a few positions with this material
                for _ in range(min(10, max(2, stats['total'] // 2))):
                    board = self._generate_material_position(white_list, black_list)
                    if board:
                        tensor = board_to_tensor(board)

                        # Evaluation based on historical success
                        if win_rate > 0.7:
                            evaluation = 0.8
                        elif win_rate > 0.5:
                            evaluation = 0.4
                        elif draw_rate > 0.5:
                            evaluation = 0.0
                        else:
                            evaluation = -0.4

                        # Adjust for side to move
                        if board.turn == chess.BLACK:
                            evaluation = -evaluation

                        self.positions.append(tensor)
                        self.evaluations.append(evaluation)
                        self.sources.append('endgame_learned')
                        self.phase_labels.append('endgame')

                        added_count += 1

            except:
                continue

        print(f"✅ Added {added_count} learned endgame patterns")

    def _generate_material_position(self, white_pieces, black_pieces):
        """Generate position with specific material"""
        for _ in range(20):  # Max attempts
            try:
                board = chess.Board.empty()
                placed = set()

                # Place pieces
                for piece_char in white_pieces:
                    piece_type = {'K': chess.KING, 'Q': chess.QUEEN, 'R': chess.ROOK,
                                'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}.get(piece_char)
                    if piece_type:
                        available = [sq for sq in chess.SQUARES if sq not in placed]
                        if available:
                            square = random.choice(available)
                            board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                            placed.add(square)

                for piece_char in black_pieces:
                    piece_type = {'K': chess.KING, 'Q': chess.QUEEN, 'R': chess.ROOK,
                                'B': chess.BISHOP, 'N': chess.KNIGHT, 'P': chess.PAWN}.get(piece_char)
                    if piece_type:
                        available = [sq for sq in chess.SQUARES if sq not in placed]
                        if available:
                            square = random.choice(available)
                            board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
                            placed.add(square)

                board.turn = random.choice([chess.WHITE, chess.BLACK])

                if board.is_valid() and not board.is_game_over():
                    return board
            except:
                continue
        return None

    def _smart_augmentation(self):
        """Smart data augmentation based on game phases"""
        print("🔄 Smart data augmentation...")
        original_count = len(self.positions)

        # Augment based on phase importance
        phase_weights = {'opening': 0.3, 'middlegame': 0.2, 'endgame': 0.4}

        for i, (position, evaluation, phase) in enumerate(zip(self.positions, self.evaluations, self.phase_labels)):
            if random.random() < phase_weights.get(phase, 0.2):
                # Horizontal flip
                flipped = position.flip(-1)
                self.positions.append(flipped)
                self.evaluations.append(evaluation)
                self.sources.append(f"{self.sources[i]}_flip")
                self.phase_labels.append(phase)

        print(f"✅ Augmented: {original_count} -> {len(self.positions)} positions")

    def _is_quality_game(self, game, min_elo):
        """Filter for quality games"""
        try:
            white_elo = game.headers.get("WhiteElo", "0")
            black_elo = game.headers.get("BlackElo", "0")

            if white_elo.isdigit() and black_elo.isdigit():
                if int(white_elo) < min_elo or int(black_elo) < min_elo:
                    return False

            move_count = len(list(game.mainline_moves()))
            return move_count >= 20  # Reasonable game length
        except:
            return True

    def _print_analysis(self):
        """Print analysis of learned patterns"""
        from collections import Counter

        print("\n📊 Dataset Analysis:")

        # Source distribution
        source_counts = Counter(self.sources)
        for source, count in source_counts.items():
            percentage = (count / len(self.sources)) * 100
            print(f"   {source}: {count} ({percentage:.1f}%)")

        # Phase distribution
        phase_counts = Counter(self.phase_labels)
        print("\n🎯 Game Phase Distribution:")
        for phase, count in phase_counts.items():
            percentage = (count / len(self.phase_labels)) * 100
            print(f"   {phase}: {count} ({percentage:.1f}%)")

        # Top opening patterns
        print(f"\n📚 Learned {len(self.opening_stats)} opening patterns")
        if self.opening_stats:
            top_openings = sorted(self.opening_stats.items(),
                                key=lambda x: x[1]['total'], reverse=True)[:5]
            print("   Top patterns:")
            for pattern, stats in top_openings:
                win_rate = stats['wins'] / stats['total'] * 100
                print(f"     {pattern.split(':')[1]}: {stats['total']} games, {win_rate:.1f}% win rate")

        # Endgame patterns
        print(f"\n🏁 Learned {len(self.endgame_stats)} endgame patterns")
        if self.endgame_stats:
            top_endgames = sorted(self.endgame_stats.items(),
                                key=lambda x: x[1]['total'], reverse=True)[:5]
            print("   Top patterns:")
            for pattern, stats in top_endgames:
                win_rate = stats['wins'] / stats['total'] * 100
                print(f"     {pattern}: {stats['total']} games, {win_rate:.1f}% win rate")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx].float(), torch.tensor(self.evaluations[idx], dtype=torch.float32)

# Use the same trainer as before...
class SimpleChessTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=7, factor=0.7
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for positions, targets in tqdm(train_loader, desc="Training", leave=False):
            positions = positions.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(positions).squeeze()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

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

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, save_path):
        print(f"🚀 Starting training for {epochs} epochs...")
        print("=" * 50)

        patience_counter = 0
        max_patience = 12

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"LR={current_lr:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(save_path, epoch, train_loss, val_loss)
                patience_counter = 0
                print(f"✅ New best model saved! Val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"🛑 Early stopping")
                    break

        print("=" * 50)
        print(f"🎉 Training complete! Best validation loss: {self.best_val_loss:.4f}")

    def save_model(self, path, epoch, train_loss, val_loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

def train_smart_chess_ai(data_path, epochs=50, batch_size=64, max_games=3000,
                        min_elo=1400, learn_openings=True, learn_endgames=True):
    """Train AI that learns real patterns from PGN files"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Load smart dataset that learns from PGN
    dataset = SmartChessDataset(
        data_path,
        max_games=max_games,
        min_elo=min_elo,
        learn_openings=learn_openings,
        learn_endgames=learn_endgames,
        augment_data=True
    )

    if len(dataset) == 0:
        raise ValueError("No training data loaded!")

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    print(f"📊 Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Model and trainer
    model = create_chess_model()
    trainer = SimpleChessTrainer(model, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")

    # Train
    save_path = "trained_models/smart_chess_ai.pth"
    trainer.train(train_loader, val_loader, epochs, save_path)

    return save_path

def main():
    parser = argparse.ArgumentParser(description="Smart Chess AI Training - Learns from Real Games")
    parser.add_argument("--data", default="data/pgn", help="PGN file or directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_games", type=int, default=3000)
    parser.add_argument("--min_elo", type=int, default=1400)
    parser.add_argument("--no_openings", action="store_true", help="Don't learn opening patterns")
    parser.add_argument("--no_endgames", action="store_true", help="Don't learn endgame patterns")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Path not found: {args.data}")
        return

    print("🧠 Smart Chess AI Training")
    print("Will learn real opening moves and endgame patterns from your PGN files!")
    print("=" * 60)

    model_path = train_smart_chess_ai(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_games=args.max_games,
        min_elo=args.min_elo,
        learn_openings=not args.no_openings,
        learn_endgames=not args.no_endgames
    )

    print(f"🎉 Smart chess AI saved to: {model_path}")

if __name__ == "__main__":
    main()
