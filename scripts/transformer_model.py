import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=2, num_layers=2, hidden_dim=64):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=128,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        batch_size, seq_len, _ = src.size()
        src = self.input_layer(src) + self.positional_encoding[:, :seq_len, :]
        transformer_output = self.transformer(src, src)
        output = self.output_layer(transformer_output[:, -1, :])
        return output