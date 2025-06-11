import pandas as pd
import text2emotion as te
import numpy as np # For handling potential NaN values gracefully

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = r'data\raw_convo\conversation_analysis.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

if 'message' not in df.columns:
    print("Error: The CSV file must contain a 'message' column.")
    exit()

# Ensure the 'message' column contains strings and handle NaN values
# Convert any non-string entries to string and replace actual NaN with empty string
df['message'] = df['message'].astype(str).replace('nan', '')

# Define a function to get emotions and handle potential errors
def get_emotions_safe(text):
    if not isinstance(text, str) or not text.strip():
        # If text is not a string or is empty/whitespace, return a dictionary of zeros
        return {'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.0}
    try:
        emotions = te.get_emotion(text)
        # Ensure all expected emotion keys are present, even if their value is 0.0
        # text2emotion usually returns all 5, but this is a safeguard.
        return {
            'Happy': emotions.get('Happy', 0.0),
            'Angry': emotions.get('Angry', 0.0),
            'Surprise': emotions.get('Surprise', 0.0),
            'Sad': emotions.get('Sad', 0.0),
            'Fear': emotions.get('Fear', 0.0)
        }
    except Exception as e:
        print(f"Warning: Could not process text '{text[:50]}...' for emotion analysis. Error: {e}")
        return {'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.0}


# Apply the function to the 'message' column
df['emotion_scores'] = df['message'].apply(get_emotions_safe)

# Expand the dictionary into separate columns
df['Happy_Emotion'] = df['emotion_scores'].apply(lambda x: x['Happy'])
df['Angry_Emotion'] = df['emotion_scores'].apply(lambda x: x['Angry'])
df['Surprise_Emotion'] = df['emotion_scores'].apply(lambda x: x['Surprise'])
df['Sad_Emotion'] = df['emotion_scores'].apply(lambda x: x['Sad'])
df['Fear_Emotion'] = df['emotion_scores'].apply(lambda x: x['Fear'])

# Drop the intermediate 'emotion_scores' column if you don't need it
df = df.drop(columns=['emotion_scores'])

# Save the updated DataFrame to a new CSV file
output_file_path = r'data\raw_convo\conversation_analysis.csv'
df.to_csv(output_file_path, index=False)
print(f"Processed data with emotion scores saved to '{output_file_path}'")