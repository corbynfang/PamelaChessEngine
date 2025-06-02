import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAI(nn.Module):
    """Single, optimized chess evaluation model"""
    def __init__(self, input_channels=13):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Layer 1: 13 -> 64 channels
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 128 -> 128 channels (deeper features)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling - reduces 8x8 to 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Evaluation head
        self.evaluator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        evaluation = self.evaluator(features)
        return evaluation

# Simple factory function
def create_chess_model():
    return ChessAI()
