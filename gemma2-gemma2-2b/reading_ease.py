import pandas as pd
import textstat

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = r'conversation_analysis.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

if 'message' not in df.columns:
    print("Error: The CSV file must contain a 'message' column.")
    exit()

# Apply textstat functions
df['flesch_reading_ease'] = df['message'].apply(textstat.flesch_reading_ease)
df['text_standard'] = df['message'].apply(textstat.text_standard)



# To overwrite the original file (use with caution):
df.to_csv(file_path, index=False)
print(f"Processed data saved to '{file_path}'")