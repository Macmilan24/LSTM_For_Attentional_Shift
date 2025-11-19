import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("Loading the trained model and tokenizer")
model = load_model('lstm_topic_model.keras')

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


categories = ["insect", "poison", "insecticide", "species", "disease"]
MAX_SEQUENCE_LENGTH = 5

print("Model and tokenizer loaded successfully.")


print("\n---  Defining the test word stream ---")
test_stream = [
    'ant', 'aphid', 'caterpillar', 'locust', 'spider',
    'arsenic', 'cyanide', 'cocaine', 'botulinum', 'aconite'
]
print(f"Test stream: {test_stream}")



print("\n--- Running the simulation step-by-step ---")
current_sequence = []
predictions_over_time = []

for word in test_stream:
    current_sequence.append(word)

    if len(current_sequence) > MAX_SEQUENCE_LENGTH:
        current_sequence.pop(0)

    text_sequence = " ".join(current_sequence)
    int_sequence = tokenizer.texts_to_sequences([text_sequence])
    padded_sequence = pad_sequences(int_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = model.predict(padded_sequence, verbose=0)[0]
    predictions_over_time.append(prediction)

    print(f"\n- Input Sequence: {current_sequence}")
    prediction_dict = {cat: f"{prob:.2%}" for cat, prob in zip(categories, prediction)}
    print(f"- Predictions: {prediction_dict}")


print("\n--- Generating the final plot ---")
predictions_over_time = np.array(predictions_over_time)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

for i, category in enumerate(categories):
    ax.plot(predictions_over_time[:, i], label=category, marker='o', linestyle='-')


ax.set_title('LSTM Topic Prediction Over Time (Simulated Attentional Shift)', fontsize=16)
ax.set_xlabel('Time Step (Word in Stream)', fontsize=12)
ax.set_ylabel('Predicted Probability', fontsize=12)
ax.set_xticks(range(len(test_stream)))
ax.set_xticklabels(test_stream, rotation=45, ha="right")
ax.axvline(x=4.5, color='r', linestyle='--', lw=2, label='Topic Switch')
ax.legend(title='Topic Categories', loc='upper left', bbox_to_anchor=(1.02, 1))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.set_yticklabels([f'{int(y*100)}%' for y in ax.get_yticks()])

plt.tight_layout() 

plt.savefig('lstm_attention_shift_plot.png')
print("\nPlot saved as 'lstm_attention_shift_plot.png'.")
plt.show()