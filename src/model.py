import torch
import torch.nn as nn

NUM_CHANNELS = 12
TOTAL_CLASSES = 50

# model class
class EMGHybridModel(nn.Module):
    def __init__(self, input_channels=NUM_CHANNELS, num_classes=TOTAL_CLASSES):
        super(EMGHybridModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 512, kernel_size=3, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.lstm = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for param_name, param in m.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return x