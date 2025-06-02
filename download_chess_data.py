import urllib.request
import os
import zipfile
import gzip
import shutil
from pathlib import Path

def download_chess_resources():
    """Download opening books and endgame tables"""

    # Create directories
    books_dir = Path("books")
    tablebases_dir = Path("tablebases")
    books_dir.mkdir(exist_ok=True)
    tablebases_dir.mkdir(exist_ok=True)

    print("Downloading chess resources...")

    # Download opening books
    opening_books = [
        {
            "name": "Human.bin",
            "url": "https://github.com/niklasf/python-chess/raw/master/data/opening-books/Human.bin",
            "path": books_dir / "human.bin"
        },
        {
            "name": "Computer.bin",
            "url": "https://github.com/niklasf/python-chess/raw/master/data/opening-books/Computer.bin",
            "path": books_dir / "computer.bin"
        }
    ]

    for book in opening_books:
        if not book["path"].exists():
            print(f"Downloading {book['name']}...")
            try:
                urllib.request.urlretrieve(book["url"], book["path"])
                print(f"✓ Downloaded {book['name']}")
            except Exception as e:
                print(f"✗ Failed to download {book['name']}: {e}")

    # Download basic 3-4-5 piece tablebases (smaller size)
    tablebase_files = [
        "KQvK.rtbw", "KQvK.rtbz",  # King + Queen vs King
        "KRvK.rtbw", "KRvK.rtbz",  # King + Rook vs King
        "KPvK.rtbw", "KPvK.rtbz",  # King + Pawn vs King
        "KBBvK.rtbw", "KBBvK.rtbz", # King + 2 Bishops vs King
        "KBNvK.rtbw", "KBNvK.rtbz", # King + Bishop + Knight vs King
    ]

    # Note: Syzygy tablebases are very large (hundreds of GB for complete sets)
    # For this example, we'll create a simple endgame knowledge base instead
    print("Creating endgame knowledge base...")
    create_endgame_knowledge(tablebases_dir)

    print("Chess resources setup complete!")

def create_endgame_knowledge(tablebases_dir):
    """Create a simple endgame knowledge base"""
    endgame_knowledge = {
        # Format: (white_pieces, black_pieces): evaluation_function
        "basic_endgames": {
            "KQ_vs_K": {"result": "win", "difficulty": "easy"},
            "KR_vs_K": {"result": "win", "difficulty": "medium"},
            "KBB_vs_K": {"result": "win", "difficulty": "hard"},
            "KBN_vs_K": {"result": "win", "difficulty": "very_hard"},
            "KP_vs_K": {"result": "depends", "difficulty": "medium"},
            "K_vs_K": {"result": "draw", "difficulty": "trivial"}
        }
    }

    import json
    with open(tablebases_dir / "endgame_knowledge.json", "w") as f:
        json.dump(endgame_knowledge, f, indent=2)

if __name__ == "__main__":
    download_chess_resources()
