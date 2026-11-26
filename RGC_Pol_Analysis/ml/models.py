"""Models, adapted from the upstream ml/ folder.

Both output 2 numbers: P and log(sigma_P^2) (heteroscedastic uncertainty). The output layer is linear, so
negative P is possible (the upstream sigmoid sits only inside the SE attention block).

  cnn        PolarizationCNN from ml/train_cnn_v2.py (3 conv blocks, adaptive pooling that keeps the left/right
             peak positions, MLP trunk); in_channels configurable; heads P and logvar instead of P and Q
  inception  CNNArchitectureModel from ml/pol_cnn.py (inception block, residual blocks, optional SE block,
             global pooling); heads P and logvar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -14.0, 2.0      # sigma_P between ~0.001 and ~2.7


class PolarizationCNN(nn.Module):
    def __init__(self, in_channels=2, width=1):
        super().__init__()
        c1, c2, c3 = 32 * width, 64 * width, 128 * width
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=9, padding=4), nn.BatchNorm1d(c1), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=7, padding=3), nn.BatchNorm1d(c2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2), nn.BatchNorm1d(c3), nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 32, 128), nn.GELU())
        self.p_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.trunk(self.features(x))
        return self.p_head(h).squeeze(-1), self.logvar_head(h).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)


class InceptionBlock(nn.Module):
    def __init__(self, cin, c1, c2, c3, c4):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(cin, c1, 1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(cin, c2[0], 1), nn.ReLU(), nn.Conv1d(c2[0], c2[1], 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(cin, c3[0], 1), nn.ReLU(), nn.Conv1d(c3[0], c3[1], 5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(cin, c4, 1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        return x + self.bn(self.conv2(F.relu(self.conv1(x))))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=2):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(-1)))))
        return x * y.unsqueeze(-1)


class InceptionResNet(nn.Module):
    def __init__(self, in_channels=2, num_residual_blocks=3, use_se_block=True):
        super().__init__()
        c1, c2, c3, c4 = 64, (32, 96), (32, 160), 32
        ch = c1 + c2[1] + c3[1] + c4
        self.inception = InceptionBlock(in_channels, c1, c2, c3, c4)
        self.res = nn.ModuleList(ResidualBlock(ch) for _ in range(num_residual_blocks))
        self.se = SEBlock(ch) if use_se_block else None
        self.fc = nn.Linear(ch, 32)
        self.p_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.inception(x)
        for r in self.res:
            x = r(x)
        if self.se is not None:
            x = self.se(x)
        h = F.relu(self.fc(x.mean(-1)))
        return self.p_head(h).squeeze(-1), self.logvar_head(h).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)


def build_model(name, in_channels):
    if name == "cnn":
        return PolarizationCNN(in_channels)
    if name == "inception":
        return InceptionResNet(in_channels)
    raise ValueError(f"unknown model {name}")
