import json
import numpy as np
import random
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


words_data = {
    "insect": ["Ant", "ant", "aphid", "aphids", "armyworm", "armyworms", "beetwebworm", "spider", "beanpodborer", "beanfly", "caterpillar", "locust", "grasshopper", "jassid"],
    "poison": ["aconite", "aflatoxin", "alcohol", "ammonia", "arsenic", "aspirin", "azoxystrobi", "botulinum", "caffeine", "chlorine", "cocaine", "cyanide"],
    "insecticide": ["abamectin", "acetamiprid", "alachlor", "fosmethilan", "fenson", "ethion", "lindane", "fluvalinate", "hydramethylnon", "bendiocarb", "aldrin", "benazolin", "dichloropropene", "aldicarb", "diazinon", "dicofol", "dinoseb", "ethiofencarb", "methidathion", "trichlorophenol"],
    "species": ["eusocialSpecies", "agriculturalPestSpecies", "nocturnalSpecies", "toxicPlantSpecies", "heavyMetalForm", "solventForm", "irritantClass", "fungalToxinSpecies"],
    "disease": ["parasiticInfection", "envenomation", "plantVirusSyndrome", "yieldDeclineSyndrom", "leafBlight", "necroticLeafSyndrom", "poisoning", "liverCancer", "intoxication", "irritation"]
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

model.save('lstm_topic_model.keras')

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("\nTraining complete!")
