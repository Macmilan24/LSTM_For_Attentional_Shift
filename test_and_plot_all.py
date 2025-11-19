import json
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Main Test Configuration ---
MODELS_TO_TEST = {
    "V1: Trained on Generated Data": {
        "model_path": "model_v1_generated.keras",
        "tokenizer_path": "tokenizer_v1.json",
        "max_len": 5
    },
    "V2: Trained on Sentences": {
        "model_path": "model_v2_sentences.keras",
        "tokenizer_path": "tokenizer_v2.json",
        "max_len": 20
    },
    "V3: Trained on Word Lists": {
        "model_path": "model_v3_words.keras",
        "tokenizer_path": "tokenizer_v3.json",
        "max_len": 5
    }
}
CATEGORIES = ["cars", "insect", "poison", "insecticide"]
TEST_STREAM = [
    'car', 'engine', 'wheel', 'speed', 'driving', 'road', 'vehicle', # Phase 1
    'ant', 'spider', 'locust', 'aphid', 'caterpillar', 'beetle',     # Phase 2
    'lindane',                                                      # Phase 3
    'arsenic', 'cyanide', 'botulinum', 'aconite', 'chlorine',        # Phase 4
    'motor', 'tire', 'highway'                                      # Phase 5
]

def run_simulation(model, tokenizer, max_len):
    current_sequence = []
    predictions_over_time = []
    for word in TEST_STREAM:
        current_sequence.append(word)
        if len(current_sequence) > max_len:
            current_sequence.pop(0)
        
        text_sequence = " ".join(current_sequence)
        int_sequence = tokenizer.texts_to_sequences([text_sequence])
        padded_sequence = pad_sequences(int_sequence, maxlen=max_len, padding='post')
        
        prediction = model.predict(padded_sequence, verbose=0)[0]
        predictions_over_time.append(prediction)
    return np.array(predictions_over_time)

# --- Plotting ---
fig, axes = plt.subplots(len(MODELS_TO_TEST), 1, figsize=(15, 20), sharex=True)
fig.suptitle('Comparison of LSTM Models on a Comprehensive Test', fontsize=20)

for ax, (title, config) in zip(axes, MODELS_TO_TEST.items()):
    print(f"--- Testing {title} ---")
    if not (os.path.exists(config['model_path']) and os.path.exists(config['tokenizer_path'])):
        print(f"Skipping {title}: Model or tokenizer file not found.")
        continue

    model = load_model(config['model_path'])
    with open(config['tokenizer_path']) as f:
        tokenizer = tokenizer_from_json(json.load(f))
    
    predictions = run_simulation(model, tokenizer, config['max_len'])

    for i, category in enumerate(CATEGORIES):
        ax.plot(predictions[:, i], label=category, marker='o', linestyle='-')

    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Predicted Probability')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

# Common X-axis formatting
plt.xticks(ticks=range(len(TEST_STREAM)), labels=TEST_STREAM, rotation=45, ha="right")
plt.xlabel('Time Step (Word in Stream)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('final_comparison_plot.png')
print("\nPlot saved as 'final_comparison_plot.png'")
plt.show()