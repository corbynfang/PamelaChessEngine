import subprocess
import sys
from pathlib import Path
import urllib.request
import json
import zipfile
import io

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch",
        "chess",
        "berserk",
        "python-dotenv",
        "tqdm",
        "numpy"
    ]

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package}")
        except:
            print(f"✗ Failed to install {package}")

def setup_directories():
    """Create necessary directories"""
    dirs = ["books", "tablebases", "trained_models", "data"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")

def create_basic_opening_book():
    """Create a basic opening book from common openings"""
    import chess
    import chess.polyglot

    # Create a simple opening book with common openings
    openings = {
        # Starting position -> common first moves
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
            ("e2e4", 100),  # King's Pawn
            ("d2d4", 95),   # Queen's Pawn
            ("g1f3", 85),   # Reti Opening
            ("c2c4", 80),   # English Opening
        ],
        # After 1.e4
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
            ("e7e5", 100),  # Open Game
            ("c7c5", 95),   # Sicilian Defense
            ("e7e6", 80),   # French Defense
            ("c7c6", 75),   # Caro-Kann Defense
        ],
        # After 1.d4
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": [
            ("d7d5", 100),  # Queen's Gambit
            ("g8f6", 95),   # Indian Defenses
            ("f7f5", 70),   # Dutch Defense
            ("e7e6", 85),   # French-style setup
        ],
    }

    # Create opening knowledge JSON (simpler format)
    opening_data = {
        "openings": {},
        "principles": {
            "control_center": 0.1,
            "develop_pieces": 0.08,
            "king_safety": 0.12,
            "early_queen": -0.05
        }
    }

    # Convert to our format
    for position_fen, moves in openings.items():
        opening_data["openings"][position_fen] = {}
        for move_uci, weight in moves:
            opening_data["openings"][position_fen][move_uci] = {
                "weight": weight,
                "evaluation": 0.05 + (weight / 2000.0)  # Small positive bias
            }

    # Save opening data
    with open("books/opening_book.json", "w") as f:
        json.dump(opening_data, f, indent=2)

    print("✓ Created basic opening book")

def download_alternative_books():
    """Download alternative opening book sources"""

    # Try to download from lichess opening database
    try:
        print("Downloading Lichess opening data...")
        url = "https://explorer.lichess.ovh/masters?play="

        # Create a simple opening database from common positions
        opening_positions = [
            "",  # Starting position
            "e2e4",  # After 1.e4
            "e2e4,e7e5",  # After 1.e4 e5
            "d2d4",  # After 1.d4
            "d2d4,d7d5",  # After 1.d4 d5
        ]

        lichess_data = {}

        for position in opening_positions[:2]:  # Just get a few to avoid rate limiting
            try:
                full_url = url + position
                response = urllib.request.urlopen(full_url)
                if response.status == 200:
                    import json
                    data = json.loads(response.read().decode())
                    if 'moves' in data:
                        lichess_data[position] = data['moves']
                        print(f"✓ Got data for position: {position or 'start'}")
            except Exception as e:
                print(f"Could not get lichess data for {position}: {e}")
                continue

        # Save lichess data
        if lichess_data:
            with open("books/lichess_openings.json", "w") as f:
                json.dump(lichess_data, f, indent=2)
            print("✓ Downloaded Lichess opening data")

    except Exception as e:
        print(f"Could not download from Lichess: {e}")

def create_comprehensive_opening_knowledge():
    """Create comprehensive opening knowledge base"""

    opening_knowledge = {
        "book_moves": {
            # Format: "fen_position": {"move_uci": {"weight": int, "eval": float}}
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": {
                "e2e4": {"weight": 100, "eval": 0.1},
                "d2d4": {"weight": 95, "eval": 0.08},
                "g1f3": {"weight": 85, "eval": 0.05},
                "c2c4": {"weight": 80, "eval": 0.06}
            },
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": {
                "e7e5": {"weight": 100, "eval": 0.0},
                "c7c5": {"weight": 95, "eval": 0.02},
                "e7e6": {"weight": 80, "eval": -0.01},
                "c7c6": {"weight": 75, "eval": 0.01}
            },
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": {
                "d7d5": {"weight": 100, "eval": 0.0},
                "g8f6": {"weight": 95, "eval": 0.01},
                "f7f5": {"weight": 70, "eval": -0.02},
                "e7e6": {"weight": 85, "eval": 0.0}
            },
            # After 1.e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": {
                "g1f3": {"weight": 100, "eval": 0.08},
                "f1c4": {"weight": 85, "eval": 0.05},
                "d2d3": {"weight": 70, "eval": 0.02},
                "f2f4": {"weight": 60, "eval": 0.0}
            },
            # After 1.d4 d5
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2": {
                "c2c4": {"weight": 100, "eval": 0.1},
                "g1f3": {"weight": 90, "eval": 0.05},
                "c1f4": {"weight": 75, "eval": 0.03},
                "e2e3": {"weight": 80, "eval": 0.02}
            }
        },
        "opening_principles": {
            "center_control_bonus": 0.08,
            "piece_development_bonus": 0.05,
            "early_castling_bonus": 0.1,
            "early_queen_penalty": -0.1,
            "repeated_moves_penalty": -0.05
        }
    }

    with open("books/comprehensive_openings.json", "w") as f:
        json.dump(opening_knowledge, f, indent=2)

    print("✓ Created comprehensive opening knowledge")

def download_chess_resources():
    """Download and create chess resources"""

    print("Creating opening resources...")

    # Create basic opening book
    create_basic_opening_book()

    # Create comprehensive opening knowledge
    create_comprehensive_opening_knowledge()

    # Try to get additional data from Lichess
    download_alternative_books()

    # Create endgame knowledge (enhanced version)
    endgame_knowledge = {
        "basic_endgames": {
            # King + Queen vs King
            "KQ_vs_K": {"result": "win", "difficulty": "easy", "eval": 0.95, "moves_to_mate": 10},
            # King + Rook vs King
            "KR_vs_K": {"result": "win", "difficulty": "medium", "eval": 0.85, "moves_to_mate": 16},
            # King + 2 Rooks vs King
            "KRR_vs_K": {"result": "win", "difficulty": "easy", "eval": 0.98, "moves_to_mate": 8},
            # King + 2 Bishops vs King
            "KBB_vs_K": {"result": "win", "difficulty": "hard", "eval": 0.75, "moves_to_mate": 30},
            # King + Bishop + Knight vs King
            "KBN_vs_K": {"result": "win", "difficulty": "very_hard", "eval": 0.65, "moves_to_mate": 35},
            # King + Pawn vs King
            "KP_vs_K": {"result": "depends", "difficulty": "medium", "eval": 0.2, "notes": "depends on pawn position"},
            # Drawn endgames
            "K_vs_K": {"result": "draw", "difficulty": "trivial", "eval": 0.0},
            "KB_vs_K": {"result": "draw", "difficulty": "easy", "eval": 0.0},
            "KN_vs_K": {"result": "draw", "difficulty": "easy", "eval": 0.0},
            "KNN_vs_K": {"result": "draw", "difficulty": "easy", "eval": 0.0, "notes": "cannot force mate"}
        },
        "pawn_endgames": {
            "opposition_rule": "Key squares and opposition determine outcome",
            "passed_pawn_bonus": 0.3,
            "king_activity_bonus": 0.2
        },
        "piece_values_endgame": {
            "queen": 9.5,
            "rook": 5.1,
            "bishop": 3.3,
            "knight": 3.2,
            "pawn": 1.0,
            "king": 4.0  # King is active in endgame
        }
    }

    with open("tablebases/endgame_knowledge.json", "w") as f:
        json.dump(endgame_knowledge, f, indent=2)
    print("✓ Created enhanced endgame knowledge")

def create_sample_pgn():
    """Create a sample PGN file for testing"""
    sample_games = '''[Event "Sample Game 1"]
[Site "Training"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Sample Game 2"]
[Site "Training"]
[Date "2024.01.01"]
[Round "2"]
[White "Player2"]
[Black "Player1"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 c6 7. Qc2 Nbd7 8. Bd3 Nh5 9. Bxe7 Qxe7 0-1

[Event "Sample Game 3"]
[Site "Training"]
[Date "2024.01.01"]
[Round "3"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]

1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 6. e4 e5 7. Nc3 Nc6 8. a3 a5 9. Rb1 h6 1/2-1/2
'''

    with open("data/sample_games.pgn", "w") as f:
        f.write(sample_games)

    print("✓ Created sample PGN file")

if __name__ == "__main__":
    print("🚀 Setting up Chess Bot (Fixed Version)...")
    print("=" * 50)

    print("\n📦 Installing packages...")
    install_requirements()

    print("\n📁 Creating directories...")
    setup_directories()

    print("\n📚 Creating chess resources...")
    download_chess_resources()

    print("\n📄 Creating sample data...")
    create_sample_pgn()

    print("\n✅ Setup complete!")
    print("\nFiles created:")
    print("  📖 books/opening_book.json - Basic opening moves")
    print("  📖 books/comprehensive_openings.json - Detailed opening knowledge")
    print("  📖 books/lichess_openings.json - Lichess data (if available)")
    print("  🏁 tablebases/endgame_knowledge.json - Endgame evaluations")
    print("  🎮 data/sample_games.pgn - Sample training games")

    print("\nNext steps:")
    print("1. Add more PGN files to data/ folder (optional)")
    print("2. Run: python train.py --data data/sample_games.pgn --epochs 20")
    print("3. Create .env file with: LICHESS_API_TOKEN=your_token_here")
    print("4. Run: python lichess.py")
