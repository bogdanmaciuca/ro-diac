# https://github.com/aleris/ReadME-RoTex-Corpus-Builder?tab=readme-ov-file

import torch

COMMON_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\
    .,!?;:-_()\"'"
DIACRITICS = "ăâîșțĂÂÎȘȚ"
VOCAB = sorted(list(set(COMMON_CHARS + DIACRITICS)))
VOCAB_SIZE = len(VOCAB)

char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}

diac_map = str.maketrans('ăâîșțĂÂÎȘȚ', 'aaistAAIST')


def remove_diacritics(text):
    return text.translate(diac_map)


def text_to_tensor(text):
    indices = [char_to_idx.get(c, 0) for c in text]
    return torch.tensor(indices, dtype=torch.long)
