import torch
import useful


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.step = 32

    def __len__(self):
        return (len(self.text) - self.seq_len) // self.step

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.seq_len
        chunk_y = self.text[start:end]
        chunk_x = useful.remove_diacritics(chunk_y)
        return useful.text_to_tensor(chunk_x), useful.text_to_tensor(chunk_y)


class CharLTSM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super(CharLTSM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)

        out, hidden = self.lstm(emb, hidden)

        out = out.reshape(-1, out.shape[2])

        logits = self.fc(out)

        return logits, hidden

