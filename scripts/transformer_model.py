import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=2, num_layers=2, hidden_dim=64, dropout=0.1, max_len=1000):
        super(TransformerModel, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Sinusoidal positional encoding
        self.positional_encoding = self.create_positional_encoding(hidden_dim, max_len)

        # Transformer encoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=256,  # Increase feedforward network dimension
            dropout=dropout,     # Add Dropout
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting

    def create_positional_encoding(self, hidden_dim, max_len):
        """Create sinusoidal positional encoding"""
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, hidden_dim)

    def forward(self, src):
        batch_size, seq_len, _ = src.size()

        # Input mapping + positional encoding
        src = self.input_layer(src) + self.positional_encoding[:, :seq_len, :].to(src.device)
        src = self.dropout(src)  # Add Dropout

        # Transformer processing
        transformer_output = self.transformer(src, src)

        # Residual connection (input added to output)
        transformer_output = transformer_output + src  # Residual connection

        # Output layer (only use the output from the last time step)
        output = self.output_layer(transformer_output[:, -5, :])  # Shape: (batch_size, output_dim)
        # output = self.output_layer(transformer_output[:, -1, :]) 
        # print("output:",output)
        return output

# import torch
# import torch.nn as nn

# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, output_dim, nhead=2, num_layers=2, hidden_dim=64):
#         super(TransformerModel, self).__init__()
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
#         self.transformer = nn.Transformer(
#             d_model=hidden_dim,
#             nhead=nhead,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=128,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(hidden_dim, output_dim)

#     def forward(self, src):
#         batch_size, seq_len, _ = src.size()
#         src = self.input_layer(src) + self.positional_encoding[:, :seq_len, :]
#         transformer_output = self.transformer(src, src)
#         output = self.output_layer(transformer_output[:, -1, :])
#         return output