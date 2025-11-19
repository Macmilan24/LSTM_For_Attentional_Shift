import json
import numpy as np
import random
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


words_data = {
    "cars": ["car", "engine", "wheel", "drive", "speed", "road", "vehicle", "motor", "tire", "highway"],
    "insect": ["ant", "aphid", "caterpillar", "locust", "spider", "beetle", "armyworm", "beanfly", "grasshopper"],
    "poison": ["aconite", "aflatoxin", "alcohol", "ammonia", "arsenic", "aspirin", "azoxystrobi", "botulinum", "caffeine", "chlorine", "cocaine", "cyanide"],
    "insecticide": ["abamectin", "acetamiprid", "alachlor", "fosmethilan", "fenson", "ethion", "lindane", "fluvalinate", "hydramethylnon", "bendiocarb", "aldrin", "benazolin", "dichloropropene", "aldicarb", "diazinon", "dicofol", "dinoseb", "ethiofencarb", "methidathion", "trichlorophenol"],
    }

categories = list(words_data.keys())
category_to_int = {category: i for i, category in enumerate(categories)}
print(f"Categories to be learned: {categories}")

sequences = []
labels = []

NUM_SAMPLES_PER_CATEGORY = 500
MAX_SEQUENCE_LENGTH = 5

for category, word in words_data.items():
    for _ in range(NUM_SAMPLES_PER_CATEGORY):
        seq_length = random.randint(2,MAX_SEQUENCE_LENGTH)
        sequences = random.choices(word, k=seq_length)
        
        sequences.append(" ".join(sequences))
        labels.append(category_to_int[category])

print(f"successfully generated {len(sequences)} training samples.")

print("\n --- PreProcessing Data for the LSTM ---")

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(sequences)
vocab_size = len(tokenizer.word_index) + 1

X = tokenizer.texts_to_sequences(sequences)

X_padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

y = to_categorical(labels, num_classes=len(categories))

print("\n --- Building and Training the LSTM Model ---")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(32),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_padded, y, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

print("\n Saving the Trained Model")

model.save('model_v1_generated.keras')
with open('tokenizer_v1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

print("Model V1 training complete.")