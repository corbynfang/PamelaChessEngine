import os
import torch
import chess
import chess.pgn
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import time

def board_to_tensor(board):
    """Convert chess board to tensor representation"""
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_map = {
        (chess.PAWN, chess.WHITE): 0, (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.WHITE): 1, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.WHITE): 2, (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.WHITE): 3, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.WHITE): 4, (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.WHITE): 5, (chess.KING, chess.BLACK): 11,
    }
    
    for square, piece in board.piece_map().items():
        channel = piece_map[(piece.piece_type, piece.color)]
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        tensor[channel, rank, file] = 1.0
    
    return tensor

def preprocess_pgn_files(pgn_dir, output_dir, max_games=10000, max_positions=100000, 
                        min_elo=1200, sample_rate=0.2, chunk_size=10000):
    """Preprocess PGN files into efficient chunks"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    positions = []
    results = []
    positions_added = 0
    games_processed = 0
    chunk_num = 0
    
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    print(f"Found {len(pgn_files)} PGN files")
    
    stats = {
        'games_processed': 0,
        'positions_extracted': 0,
        'games_skipped_elo': 0,
        'games_skipped_result': 0,
        'chunks_created': 0
    }
    
    def save_chunk():
        nonlocal chunk_num, positions, results
        if positions:
            chunk_file = output_dir / f"chunk_{chunk_num:04d}.pkl"
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'positions': positions,
                    'results': results,
                    'count': len(positions)
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved chunk {chunk_num} with {len(positions)} positions")
            chunk_num += 1
            positions = []
            results = []
            stats['chunks_created'] += 1
    
    start_time = time.time()
    
    for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
        file_path = os.path.join(pgn_dir, pgn_file)
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            while games_processed < max_games and positions_added < max_positions:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Check ELO requirements
                    if min_elo > 0:
                        try:
                            white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                            black_elo = int(game.headers.get("BlackElo", "0") or "0")
                            if white_elo < min_elo or black_elo < min_elo:
                                stats['games_skipped_elo'] += 1
                                continue
                        except ValueError:
                            stats['games_skipped_elo'] += 1
                            continue
                    
                    # Parse result
                    result_str = game.headers.get("Result", "*")
                    if result_str == "1-0":
                        result = 1.0
                    elif result_str == "0-1":
                        result = -1.0
                    elif result_str == "1/2-1/2":
                        result = 0.0
                    else:
                        stats['games_skipped_result'] += 1
                        continue
                    
                    games_processed += 1
                    stats['games_processed'] += 1
                    
                    # Extract positions from game
                    board = game.board()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        board.push(move)
                        move_count += 1
                        
                        # Skip opening moves and sample positions
                        if move_count < 10:
                            continue
                        
                        if np.random.random() > sample_rate:
                            continue
                        
                        # Skip terminal positions
                        if board.is_checkmate() or board.is_stalemate():
                            continue
                        
                        # Skip positions with too few pieces (likely endgame)
                        if len(board.piece_map()) < 10:
                            continue
                        
                        tensor = board_to_tensor(board)
                        positions.append(tensor)
                        
                        # Adjust result based on whose turn it is
                        adjusted_result = result if board.turn == chess.WHITE else -result
                        results.append(torch.tensor([adjusted_result], dtype=torch.float32))
                        
                        positions_added += 1
                        stats['positions_extracted'] += 1
                        
                        # Save chunk when it reaches the size limit
                        if len(positions) >= chunk_size:
                            save_chunk()
                        
                        if positions_added >= max_positions:
                            break
                    
                    if games_processed % 1000 == 0:
                        elapsed = time.time() - start_time
                        print(f"Progress: {games_processed} games, {positions_added} positions, "
                              f"{elapsed:.1f}s elapsed")
                
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue
        
        if positions_added >= max_positions:
            break
    
    # Save remaining positions
    save_chunk()
    
    # Save metadata
    metadata = {
        'stats': stats,
        'parameters': {
            'max_games': max_games,
            'max_positions': max_positions,
            'min_elo': min_elo,
            'sample_rate': sample_rate,
            'chunk_size': chunk_size
        },
        'total_chunks': chunk_num,
        'total_positions': positions_added,
        'processing_time': time.time() - start_time
    }
    
    with open(output_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Games processed: {stats['games_processed']}")
    print(f"Games skipped (ELO): {stats['games_skipped_elo']}")
    print(f"Games skipped (result): {stats['games_skipped_result']}")
    print(f"Positions extracted: {stats['positions_extracted']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Processing time: {time.time() - start_time:.1f} seconds")
    print(f"Output directory: {output_dir}")
    
    return metadata

class ChunkedChessDataset(torch.utils.data.Dataset):
    """Dataset that loads from preprocessed chunks"""
    
    def __init__(self, chunk_dir):
        self.chunk_dir = Path(chunk_dir)
        
        # Load metadata
        with open(self.chunk_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Find all chunk files
        self.chunk_files = sorted(list(self.chunk_dir.glob("chunk_*.pkl")))
        print(f"Found {len(self.chunk_files)} chunk files")
        
        # Load chunk sizes for indexing
        self.chunk_sizes = []
        self.cumulative_sizes = [0]
        
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
                size = chunk_data['count']
                self.chunk_sizes.append(size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
        
        self.total_size = self.cumulative_sizes[-1]
        print(f"Total dataset size: {self.total_size}")
        
        # Cache for currently loaded chunk
        self.current_chunk_idx = -1
        self.current_chunk_data = None
    
    def _load_chunk(self, chunk_idx):
        """Load a specific chunk into memory"""
        if chunk_idx != self.current_chunk_idx:
            with open(self.chunk_files[chunk_idx], 'rb') as f:
                self.current_chunk_data = pickle.load(f)
            self.current_chunk_idx = chunk_idx
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Find which chunk contains this index
        chunk_idx = 0
        while chunk_idx < len(self.cumulative_sizes) - 1:
            if idx < self.cumulative_sizes[chunk_idx + 1]:
                break
            chunk_idx += 1
        
        # Load the chunk if not already loaded
        self._load_chunk(chunk_idx)
        
        # Get the local index within the chunk
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        return (self.current_chunk_data['positions'][local_idx], 
                self.current_chunk_data['results'][local_idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PGN files for efficient training")
    parser.add_argument("--pgn_dir", type=str, required=True, help="Directory containing PGN files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--max_games", type=int, default=20000, help="Maximum games to process")
    parser.add_argument("--max_positions", type=int, default=200000, help="Maximum positions to extract")
    parser.add_argument("--min_elo", type=int, default=1200, help="Minimum player ELO")
    parser.add_argument("--sample_rate", type=float, default=0.2, help="Position sampling rate")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Positions per chunk file")
    
    args = parser.parse_args()
    
    print("Preprocessing Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    preprocess_pgn_files(
        pgn_dir=args.pgn_dir,
        output_dir=args.output_dir,
        max_games=args.max_games,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size
    )