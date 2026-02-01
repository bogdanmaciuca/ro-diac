import torch
import model
import useful

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 4
LR = 0.002
SEQ_LEN = 100

raw_text = open('../data/ziarul-lumina.txt').read()[:500000]

dataset = model.TextDataset(raw_text, SEQ_LEN)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)

model = model.CharLTSM(useful.VOCAB_SIZE, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'Start training on {DEVICE}...')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()

        logits, _ = model(x_batch)

        y_targets = y_batch.view(-1)

        loss = criterion(logits, y_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}')


print('\n--- Test inferenta ---')
test_str = 'Stiinta si tehnologia progreseaza (cica). Vreau sa testez cat mai bine jucaria asta pe care am facut-o acuma. Sper sa tina la orice fiinte arunc in interiorul ei.'
print(f'Input: {test_str}')

model.eval()
with torch.no_grad():
    inp = useful.text_to_tensor(test_str).unsqueeze(0).to(DEVICE)
    logits, _ = model(inp)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    restored = ''.join([useful.idx_to_char[idx.item()] for idx in preds])
    print(f'Output: {restored}')

