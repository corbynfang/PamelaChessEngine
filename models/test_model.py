import torch
import chess
import os
from train import ChessModel, board_to_tensor

def test_model(model_path):
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Test with a simple position
    board = chess.Board()
    tensor = board_to_tensor(board).unsqueeze(0).to(device)

    with torch.no_grad():
        evaluation = model(tensor)

    print(f"Starting position evaluation: {evaluation.item()}")

if __name__ == "__main__":
    model_name = input("Enter the model filename (e.g., final_chess_model.pth): ")
    model_path = os.path.join("..", "trained_models", model_name)
    test_model(model_path)
