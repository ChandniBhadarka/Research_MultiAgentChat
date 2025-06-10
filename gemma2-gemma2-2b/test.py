# for emotion analysis
# import nltk
# nltk.download('omw-1.4') # Open Multilingual Wordnet
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

#for lexical check
import nltk
# nltk.download('punkt')         # For tokenization
nltk.download('averaged_perceptron_tagger') # For POS tagging
# nltk.download('tagsets')      # Optional: for understanding POS tags
# nltk.download('maxent_ne_chunker') # For named entity chunking (basic syntax/NER)
# nltk.download('words')         # For named entity chunking

import nltk
import os

# Define a custom NLTK data path.
# This is often 'C:\\Users\\YourUsername\\AppData\\Roaming\\nltk_data' on Windows
# You can verify this path by looking at the "Searched in:" list in your previous error traceback.
# We'll use a common default:
nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")

# Add this path to NLTK's data paths
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

print(f"NLTK will search for data in: {nltk.data.path}")

# Try downloading again. Use 'force=True' to ensure it redownloads if it thinks it's up-to-date.
try:
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, force=True)
    print("Successfully downloaded 'averaged_perceptron_tagger'.")
except Exception as e:
    print(f"Error downloading: {e}")

# You might also want to try downloading the 'taggers' collection as a whole
# if the specific tagger still gives issues.
# try:
#     nltk.download('taggers', download_dir=nltk_data_path, force=True)
#     print("Successfully downloaded 'taggers' collection.")
# except Exception as e:
#     print(f"Error downloading 'taggers' collection: {e}")