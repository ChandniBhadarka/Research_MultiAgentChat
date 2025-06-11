import json
import csv
from textblob import TextBlob

def analyze_conversation(convo_json):
    results = []
    prev_polarity = None
    for i, turn in enumerate(convo_json):
        text = turn.get('message', '')
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        sentiment_shift = None
        if prev_polarity is not None:
            sentiment_shift = polarity - prev_polarity
        prev_polarity = polarity
        
        results.append({
            'turn_index': i,
            'speaker': turn.get('speaker', 'unknown'),
            'message': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_shift': sentiment_shift,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    return results

def main():
    # 📝 Path to your input and output files
    json_filepath = r"data\raw_convo\ai_chat_log_with_memory.json"
    csv_output_path = r'data\raw_convo\conversation_analysis.csv'

    with open(json_filepath, 'r', encoding='utf-8') as f:
        convo_json = json.load(f)

    analysis = analyze_conversation(convo_json)

    # 🧾 Save analysis to CSV
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        if not analysis:
            print("No data to write.")
            return
        
        fieldnames = list(analysis[0].keys())  # Extract keys from the first result
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis:
            writer.writerow(row)

    print(f"✅ Analysis saved to: {csv_output_path}")

if __name__ == "__main__":
    main()
