import os
import chess
import torch
import berserk
from dotenv import load_dotenv

from models import create_chess_model
from utils import get_best_move

class ChessBot:
    def __init__(self):
        load_dotenv()

        # Lichess setup
        self.client = berserk.Client(berserk.TokenSession(os.getenv('LICHESS_API_TOKEN')))

        # Load trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_chess_model().to(self.device)

        checkpoint = torch.load("trained_models/smart_chess_ai.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("✅ Chess Bot ready!")

    def get_move(self, board):
        """Get move with smart depth selection"""
        if board.fullmove_number <= 6:
            depth = 3  # Good opening depth
        elif chess.popcount(board.occupied) <= 8:
            depth = 4  # Deep endgame analysis
        else:
            depth = 3  # Standard middlegame depth

        return get_best_move(board, self.model, depth)

    def start(self):
        """Start the bot and handle events"""
        print("🎮 Waiting for challenges...")

        try:
            for event in self.client.bots.stream_incoming_events():
                if event['type'] == 'challenge':
                    self.handle_challenge(event['challenge'])
                elif event['type'] == 'gameStart':
                    self.play_game(event['game']['id'])
        except KeyboardInterrupt:
            print("\n👋 Bot stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")

    def handle_challenge(self, challenge):
        """Accept challenges"""
        try:
            challenger = challenge['challenger']['name']
            self.client.bots.accept_challenge(challenge['id'])
            print(f"✅ Accepted challenge from {challenger}")
        except Exception as e:
            print(f"❌ Challenge error: {e}")

    def play_game(self, game_id):
        """Play a single game"""
        print(f"🎯 Starting game: {game_id}")

        try:
            bot_id = self.client.account.get()['id']
            our_color = None

            for event in self.client.bots.stream_game_state(game_id):

                if event['type'] == 'gameFull':
                    # Game started - determine our color
                    our_color = 'white' if event['white']['id'] == bot_id else 'black'
                    opponent = event['black']['name'] if our_color == 'white' else event['white']['name']

                    print(f"🎨 Playing {our_color.upper()} vs {opponent}")

                    # Handle initial position
                    board = self.create_board_from_moves(event['state']['moves'])
                    if self.is_our_turn(board, our_color):
                        self.make_move(game_id, board)

                elif event['type'] == 'gameState':
                    # Game state update - opponent moved
                    board = self.create_board_from_moves(event['moves'])
                    if our_color and self.is_our_turn(board, our_color):
                        self.make_move(game_id, board)

                elif event['type'] == 'gameFinish':
                    # Game ended
                    winner = event.get('winner', 'draw')
                    print(f"🎉 Game finished - Winner: {winner}")
                    break

        except Exception as e:
            print(f"❌ Game error: {e}")

    def create_board_from_moves(self, moves_str):
        """Create board position from move string"""
        board = chess.Board()
        if moves_str:
            for move_uci in moves_str.split():
                try:
                    board.push(chess.Move.from_uci(move_uci))
                except:
                    break  # Invalid move, stop parsing
        return board

    def is_our_turn(self, board, our_color):
        """Check if it's our turn to move"""
        if board.is_game_over():
            return False

        white_to_move = (board.turn == chess.WHITE)
        we_are_white = (our_color == 'white')

        return white_to_move == we_are_white

    def make_move(self, game_id, board):
        """Make our move"""
        try:
            # Get the best move from our AI
            move = self.get_move(board)

            if move and move in board.legal_moves:
                # Send move to Lichess
                self.client.bots.make_move(game_id, move.uci())

                # Display move info
                move_san = board.san(move)  # Algebraic notation
                move_num = board.fullmove_number
                pieces = chess.popcount(board.occupied)

                print(f"♟️  Move {move_num}: {move_san} ({move.uci()}) - {pieces} pieces")
            else:
                print("❌ No valid move found!")

        except Exception as e:
            print(f"❌ Move error: {e}")

def main():
    """Main function"""
    print("🚀 Starting Chess Bot")
    print("=" * 30)

    bot = ChessBot()
    bot.start()

if __name__ == "__main__":
    main()
