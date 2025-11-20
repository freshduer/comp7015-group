import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, pretrained_embeddings=None, bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if self.use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch size, sent len, emb dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: [batch size, sent len, lstm_output_dim]
        
        if self.use_attention:
            # Attention mechanism
            # output: [batch size, seq len, lstm_output_dim]
            # attention weights: [batch size, seq len, 1]
            attn_weights = F.softmax(self.attention(output), dim=1)
            
            # context vector: [batch size, lstm_output_dim]
            context = torch.sum(attn_weights * output, dim=1)
            final_feature = self.dropout(context)
        else:
            if self.lstm.bidirectional:
                # Concat the final forward and backward hidden states
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1,:,:]
            final_feature = self.dropout(hidden)
            
        return self.fc(final_feature)
