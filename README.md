# Ro-Diacritics-Restore

A lightweight Deep Learning model that automatically restores diacritics (ă, â, î, ș, ț) to Romanian text.

## Project Overview

* **Goal:** Convert "standard" text (ASCII) into gramatically correct Romanian text with diacritics.
* **Architecture:** Character-level Bidirectional LSTM (Long Short-Term Memory).
* **Stack:** Python 3, PyTorch, NumPy.
* **Status:** MVP (Functional Prototype).

## Setup and installation

### 1. Clone the repository
```bash
git clone https://github.com/bogdanmaciuca/ro-diac
cd ro-diacritics
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


## Usage

### 1. Prepare the data
Place a file containing Romanian text with diacritics inside a folder `data`.
```bash
mkdir data
mv your-training-data.txt ./data
```

### 2. Train the model
```bash
cd src
python3 train.py
```

### 3. Testing
Temporary: Use the `test_str` inside `train.py` to test the model after training it.


## TODO
- [ ] save model to disk and use it via a `predict.py`
- [ ] improve training efficiency
- [ ] add a realtime LSP client
