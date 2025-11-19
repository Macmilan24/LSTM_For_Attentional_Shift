import json
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --- 1. Load Data From Sentence Files ---
print("--- Training Model V2: Sentence Data ---")
data_files = {
    "cars": os.path.join("data", "sentences", "cars.sent"),
    "insect": os.path.join("data", "sentences", "insects.sent"),
    "poison": os.path.join("data", "sentences", "poisons.sent"),
    "insecticide": os.path.join("data", "sentences", "insecticide-long.sent")
}
categories = list(data_files.keys())
category_to_int = {category: i for i, category in enumerate(categories)}

sequences = []
labels = []
for category, filepath in data_files.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        sequences.extend(lines)
        labels.extend([category_to_int[category]] * len(lines))
print(f"Loaded {len(sequences)} total sentences.")

# --- 2. Data Preprocessing ---
MAX_SEQUENCE_LENGTH = 20
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(sequences)
X = tokenizer.texts_to_sequences(sequences)
X_padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y = to_categorical(labels, num_classes=len(categories))

# --- 3. Model Training ---
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(64),
    Dense(len(categories), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_padded, y, epochs=30, batch_size=32, verbose=0)

# --- 4. Save Model & Tokenizer ---
model.save('model_v2_sentences.keras')
with open('tokenizer_v2.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

print("Model V2 training complete.")