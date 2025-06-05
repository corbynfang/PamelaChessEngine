import torch
import chess
import random
import json
from pathlib import Path
import time

class ChessKnowledge:
    def __init__(self):
        self.opening_data = {}
        try:
            self._load_opening_data()
        except Exception as e:
            print(f"⚠️ Couldn't load opening book: {e}")
            self.opening_data = {}

    def _load_opening_data(self):
        books_file = Path("books/comprehensive_openings.json")
        if books_file.exists():
            try:
                with open(books_file) as f:
                    data = json.load(f)
                    self.opening_data = data.get("book_moves", {})
                    print(f"📖 Loaded {len(self.opening_data)} opening positions")
            except Exception as e:
                print(f"⚠️ Error loading opening book: {e}")
                self.opening_data = {}
        else:
            print("📖 No opening book found, using pure engine play")
            self.opening_data = {}

# Global instance
chess_knowledge = ChessKnowledge()

def board_to_tensor(board):
    pieces_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    tensor = torch.zeros(13, 8, 8)

    # Channel 12: Turn indicator (1 if white to move, 0 if black)
    if board.turn == chess.WHITE:
        tensor[12].fill_(1.0)

    # Channels 0-11: Piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_offset + pieces_to_index[piece.piece_type]
            file_idx = chess.square_file(square)
            rank_idx = 7 - chess.square_rank(square)
            tensor[piece_idx][rank_idx][file_idx] = 1.0

    return tensor

def get_best_move(board, model, depth=3, max_time=15.0):
    start_time = time.time()

    # Adjust thinking time based on position complexity
    if board.fullmove_number <= 8:
        max_time = 2.0  # Quick opening decisions
    elif board.fullmove_number <= 15:
        max_time = 4.0  # Still relatively quick
    elif chess.popcount(board.occupied) <= 12:
        max_time = 15.0  # More time for complex endgames
    else:
        max_time = 8.0  # Standard middlegame time

    # Extended opening book usage
    if board.fullmove_number <= 12:
        book_move = get_opening_book_move(board)
        if book_move:
            think_time = time.time() - start_time
            print(f"📖 Book move: {book_move.uci()} ({think_time:.3f}s)")
            return book_move

    # Basic opening principles if no book move
    if board.fullmove_number <= 8:  # Reduced from 6
        principled_move = get_opening_principle_move(board, model)
        if principled_move:
            think_time = time.time() - start_time
            print(f"📚 Opening principle: {principled_move.uci()} ({think_time:.3f}s)")
            return principled_move

    # Adjust search depth based on game complexity
    legal_moves_count = len(list(board.legal_moves))
    piece_count = chess.popcount(board.occupied)

    if piece_count <= 10:
        actual_depth = min(depth + 2, 7)  # Much deeper in endgame
    elif piece_count <= 16:
        actual_depth = min(depth + 1, 6)  # Deeper in late middlegame
    elif legal_moves_count > 35:
        actual_depth = max(depth - 1, 3)  # Faster when many options
    else:
        actual_depth = depth

    # Iterative deepening with time control
    try:
        best_move = None
        for current_depth in range(2, actual_depth + 1):
            if time.time() - start_time > max_time * 0.8:
                break

            move = minimax_search(board, model, current_depth, start_time, max_time)
            if move:
                best_move = move

        think_time = time.time() - start_time
        print(f"🧠 Engine move: {best_move.uci() if best_move else 'None'} (depth={actual_depth}, {think_time:.1f}s)")
        return best_move

    except Exception as e:
        print(f"❌ Search error: {e}")
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

def get_opening_principle_move(board, model):
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        piece = board.piece_at(move.from_square)
        if not piece:
            continue

        move_uci = move.uci()
        from_square = move.from_square
        to_square = move.to_square

        # Avoid moving king early (except castling)
        if piece.piece_type == chess.KING and not board.is_castling(move):
            continue

        # Avoid retreating pieces in opening without good reason
        if board.fullmove_number <= 10:
            if piece.piece_type != chess.PAWN and not board.is_capture(move):
                # Simple retreat detection
                if (piece.color == chess.WHITE and chess.square_rank(to_square) < chess.square_rank(from_square)) or \
                   (piece.color == chess.BLACK and chess.square_rank(to_square) > chess.square_rank(from_square)):
                    continue

        # Excellent moves (top priority) - return immediately
        if move_uci in ['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f1c4', 'c1f4',
                       'e7e5', 'd7d5', 'g8f6', 'b8c6', 'f1e2', 'c8f5']:
            return move

        # Castling is always good
        if board.is_castling(move):
            return move

    # If no excellent moves, find good developing moves
    good_moves = []
    for move in legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            if move.to_square in [chess.F3, chess.C3, chess.F6, chess.C6,
                                chess.C4, chess.F4, chess.E2, chess.D3]:
                good_moves.append(move)

    if good_moves:
        return evaluate_moves_with_model(board, model, good_moves)

    return None

def evaluate_moves_with_model(board, model, moves):
    if len(moves) == 1:
        return moves[0]

    best_move = None
    best_eval = float('-inf')

    for move in moves:
        try:
            board.push(move)
            with torch.no_grad():
                tensor = board_to_tensor(board).unsqueeze(0)
                device = next(model.parameters()).device
                tensor = tensor.to(device)
                eval_score = model(tensor).item()
                if board.turn == chess.BLACK:
                    eval_score = -eval_score

                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            board.pop()
        except:
            board.pop()
            continue

    return best_move

def get_opening_book_move(board):
    """Get move from opening book if available"""
    try:
        position_key = board.fen()
        if position_key in chess_knowledge.opening_data:
            moves_data = chess_knowledge.opening_data[position_key]
            if moves_data:
                best_move = None
                best_weight = 0
                for move_uci, data in moves_data.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            weight = data.get("weight", 0) if isinstance(data, dict) else data
                            if weight > best_weight:
                                best_weight = weight
                                best_move = move
                    except:
                        continue
                return best_move
    except Exception as e:
        print(f"⚠️ Opening book error: {e}")
    return None

def minimax_search(board, model, depth, start_time=None, max_time=15.0):
    """Enhanced minimax with better evaluation and time management"""

    def evaluate_position(board):
        try:
            model.eval()
            with torch.no_grad():
                tensor = board_to_tensor(board).unsqueeze(0)
                device = next(model.parameters()).device
                tensor = tensor.to(device)
                eval_score = model(tensor).item()

                # Add positional bonuses
                eval_score += get_positional_bonus(board)

                return eval_score if board.turn == chess.WHITE else -eval_score
        except Exception as e:
            return evaluate_material_balance(board)

    def minimax(board, depth, alpha, beta, maximizing):
        # Time check
        if start_time and time.time() - start_time > max_time:
            return evaluate_position(board)

        # Terminal conditions
        if depth == 0 or board.is_game_over():
            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    return 10000 if board.turn == chess.WHITE else -10000
                elif result == "0-1":
                    return -10000 if board.turn == chess.WHITE else 10000
                else:
                    return 0
            return evaluate_position(board)

        # Get smart ordered moves
        moves = order_moves_smart(board)

        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                eval_score = minimax(board, depth-1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = minimax(board, depth-1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break
            return min_eval

    # Find the best move with filtered candidates
    moves = filter_bad_moves(board, order_moves_smart(board))
    if not moves:
        return None

    best_move = None
    best_value = float('-inf')

    for move in moves:
        board.push(move)
        value = minimax(board, depth-1, float('-inf'), float('inf'), False)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move

    return best_move

def filter_bad_moves(board, moves):
    """Filter out obviously bad moves"""
    if board.fullmove_number > 6:  # Only filter after opening
        return moves

    filtered_moves = []
    for move in moves:
        piece = board.piece_at(move.from_square)

        # Don't move king early (except castling)
        if piece and piece.piece_type == chess.KING and not board.is_castling(move):
            continue

        # Don't move same piece twice in opening without good reason
        if board.fullmove_number <= 10:
            # This is a simplified check - in practice you'd want more sophisticated logic
            pass

        filtered_moves.append(move)

    return filtered_moves if filtered_moves else moves

def get_positional_bonus(board):
    bonus = 0.0
    piece_count = chess.popcount(board.occupied)

    # Center control bonus
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                bonus += 0.1
            else:
                bonus -= 0.1

    # King safety is critical in opening/middlegame
    if piece_count > 16:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        # Heavily penalize exposed kings
        if white_king == chess.E1 and piece_count > 24:
            bonus -= 0.4
        if black_king == chess.E8 and piece_count > 24:
            bonus += 0.4

        # Bonus for castled kings
        if white_king in [chess.G1, chess.C1]:
            bonus += 0.3
        if black_king in [chess.G8, chess.C8]:
            bonus -= 0.3

        # Penalize king walks in opening
        if white_king not in [chess.E1, chess.G1, chess.C1] and piece_count > 20:
            bonus -= 0.5
        if black_king not in [chess.E8, chess.G8, chess.C8] and piece_count > 20:
            bonus += 0.5

    return bonus

def order_moves_smart(board):
    """Enhanced move ordering for better search efficiency"""
    moves = list(board.legal_moves)
    if not moves:
        return []

    scored_moves = []

    for move in moves:
        score = 0

        # Winning captures (MVV-LVA)
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            moving = board.piece_at(move.from_square)
            if captured and moving:
                score += captured.piece_type * 10 - moving.piece_type

        # Promotions
        if move.promotion:
            score += move.promotion * 8

        # Checks
        board.push(move)
        if board.is_check():
            score += 5
        board.pop()

        # Castling
        if board.is_castling(move):
            score += 6

        # Good piece development in opening
        if board.fullmove_number <= 10:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Developing to good squares
                if move.to_square in [chess.F3, chess.C3, chess.F6, chess.C6,
                                    chess.C4, chess.F4, chess.C5, chess.F5]:
                    score += 3

        scored_moves.append((move, score))

    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, _ in scored_moves]

def result_to_value(result, is_white_turn):
    if result == '1-0':
        return 1.0 if is_white_turn else -1.0
    elif result == '0-1':
        return -1.0 if is_white_turn else 1.0
    else:
        return 0.0

def evaluate_material_balance(board):
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
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

    return (white_material - black_material) / 40.0

# Keep your existing test functions...
def test_utils():
    print("🧪 Testing utils...")
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"✅ board_to_tensor shape: {tensor.shape}")
    assert tensor.shape == (13, 8, 8)

    moves = order_moves_smart(board)
    print(f"✅ order_moves_smart returned {len(moves)} moves")
    assert len(moves) == 20

    print("🎉 All utils tests passed!")

if __name__ == "__main__":
    test_utils()
