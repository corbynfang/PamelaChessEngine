import os
import chess
import torch
import berserk
from dotenv import load_dotenv

from models import ChessAI
from utils import get_best_move

class ChessBot:
    def __init__(self):
        load_dotenv()

        # Lichess setup
        self.client = berserk.Client(berserk.TokenSession(os.getenv('LICHESS_API_TOKEN')))

        # Load trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessAI().to(self.device)

        # Load checkpoint
        try:
            checkpoint = torch.load("trained_models/chess_ai.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✅ Loaded trained ChessAI model")

            if 'epoch' in checkpoint:
                print(f"📊 Model from epoch {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"📊 Best validation loss: {checkpoint['best_val_loss']:.4f}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Using untrained model")

        print("✅ Chess Bot ready!")

    def get_move(self, board):
        piece_count = chess.popcount(board.occupied)
        move_count = board.fullmove_number

        # Enhanced depth selection
        if move_count <= 8:
            depth = 4
        elif piece_count <= 12:
            depth = 6  # Deep endgame search
        elif piece_count <= 20:
            depth = 5  # Late middlegame
        else:
            depth = 4  # Opening/early middlegame

        print(f"🧠 Thinking (depth={depth}, pieces={piece_count})...")
        return get_best_move(board, self.model, depth)

    def start(self):
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
        try:
            challenger = challenge['challenger']['name']
            variant = challenge.get('variant', {}).get('name', 'standard')

            if variant != 'Standard':
                print(f"❌ Declined {challenger} (variant: {variant})")
                self.client.bots.decline_challenge(challenge['id'])
                return

            self.client.bots.accept_challenge(challenge['id'])
            print(f"✅ Accepted challenge from {challenger}")

        except Exception as e:
            print(f"❌ Challenge error: {e}")

    def play_game(self, game_id):
        print(f"🎯 Starting game: {game_id}")

        try:
            bot_id = self.client.account.get()['id']
            our_color = None

            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    our_color = 'white' if event['white']['id'] == bot_id else 'black'
                    opponent = event['black']['name'] if our_color == 'white' else event['white']['name']
                    print(f"🎨 Playing {our_color.upper()} vs {opponent}")

                    board = self.create_board_from_moves(event['state']['moves'])
                    if self.is_our_turn(board, our_color):
                        self.make_move(game_id, board)

                elif event['type'] == 'gameState':
                    board = self.create_board_from_moves(event['moves'])
                    if not board.is_game_over() and our_color and self.is_our_turn(board, our_color):
                        self.make_move(game_id, board)

                elif event['type'] == 'gameFinish':
                    print(f"🎉 Game finished!")
                    break

        except Exception as e:
            print(f"❌ Game error: {e}")

    def create_board_from_moves(self, moves_str):
        board = chess.Board()
        if moves_str:
            for move_uci in moves_str.split():
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        break
                except:
                    break
        return board

    def is_our_turn(self, board, our_color):
        if board.is_game_over():
            return False
        return (board.turn == chess.WHITE) == (our_color == 'white')

    def make_move(self, game_id, board):
        try:
            move = self.get_move(board)

            if move and move in board.legal_moves:
                self.client.bots.make_move(game_id, move.uci())
                move_san = board.san(move)
                print(f"♟️ Played: {move_san} ({move.uci()})")
            else:
                print("❌ No valid move found!")

        except Exception as e:
            print(f"❌ Move error: {e}")

def main():
    load_dotenv()  # Load .env file

    print("🚀 Starting ChessAI Bot")
    print("=" * 40)

    # Check if model exists
    if not os.path.exists("trained_models/chess_ai.pth"):
        print("❌ Model file not found!")
        return

    # Check for API token
    if not os.getenv('LICHESS_API_TOKEN'):
        print("❌ API token not found!")
        return

    bot = ChessBot()
    bot.start()

if __name__ == "__main__":
    main()
