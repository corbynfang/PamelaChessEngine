import torch
import chess
import random
import json
from pathlib import Path

class ChessKnowledge:
    def __init__(self):
        self.opening_data = {}
        self._load_opening_data()

    def _load_opening_data(self):
        books_file = Path("books/comprehensive_openings.json")
        if books_file.exists():
            with open(books_file) as f:
                data = json.load(f)
                self.opening_data = data.get("book_moves", {})

chess_knowledge = ChessKnowledge()

def board_to_tensor(board):
    """Convert chess board to tensor"""
    pieces_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    tensor = torch.zeros(13, 8, 8)

    if board.turn == chess.WHITE:
        tensor[12].fill_(1.0)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_offset + pieces_to_index[piece.piece_type]
            file_idx = chess.square_file(square)
            rank_idx = 7 - chess.square_rank(square)
            tensor[piece_idx][rank_idx][file_idx] = 1.0

    return tensor

def get_best_move(board, model, depth=3):
    """Main function - optimized for speed and strength"""
    import time
    start_time = time.time()

    # Opening book (instant moves)
    if board.fullmove_number <= 3:
        book_move = get_opening_book_move(board)
        if book_move:
            print(f"📖 Book: {book_move.uci()} (0.0s)")
            return book_move

    # Adjust depth based on complexity
    legal_moves_count = len(list(board.legal_moves))
    piece_count = chess.popcount(board.occupied)

    if piece_count <= 8:
        actual_depth = min(depth + 1, 4)  # Deeper in endgame
    elif legal_moves_count > 35:
        actual_depth = max(depth - 1, 2)  # Faster when many options
    else:
        actual_depth = depth

    # Search for best move
    move = minimax_search(board, model, actual_depth)

    think_time = time.time() - start_time
    print(f"🧠 Move: {move.uci() if move else 'None'} (depth={actual_depth}, {think_time:.1f}s)")

    return move

def get_opening_book_move(board):
    """Get move from opening book"""
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
    return None

def minimax_search(board, model, depth):
    """Optimized minimax search"""

    def evaluate_position(board):
        model.eval()
        with torch.no_grad():
            tensor = board_to_tensor(board).unsqueeze(0)
            device = next(model.parameters()).device
            tensor = tensor.to(device)
            eval_score = model(tensor).item()
            return eval_score if board.turn == chess.WHITE else -eval_score

    def minimax(board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return evaluate_position(board)

        moves = order_moves(board)

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

    # Find best move
    moves = order_moves(board)
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

def order_moves(board):
    """Order moves for better search efficiency"""
    moves = list(board.legal_moves)

    scored_moves = []
    for move in moves:
        score = 0

        # Captures
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                score += captured.piece_type * 10

        # Checks
        board.push(move)
        if board.is_check():
            score += 5
        board.pop()

        scored_moves.append((move, score))

    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, _ in scored_moves]

def result_to_value(result, is_white_turn):
    """Convert game result to evaluation"""
    if result == '1-0':
        return 1.0 if is_white_turn else -1.0
    elif result == '0-1':
        return -1.0 if is_white_turn else 1.0
    else:
        return 0.0
