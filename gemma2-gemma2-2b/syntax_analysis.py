import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag # Still needed internally for chunking
from nltk.chunk import RegexpParser

# --- Configuration ---
file_path = r'data\raw_convo\conversation_analysis.csv' # CHANGE THIS to your CSV file path
output_nltk_file_path = file_path # This will overwrite the original file with new columns

# --- NLTK Functions ---

def analyze_text_nltk(text):
    """
    Performs basic chunking and lexical richness calculation using NLTK.
    POS tagging is done internally for chunking, but its output is not included.
    Returns a dictionary of results.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'nltk_chunks': [], # Retaining chunks for syntax
            'nltk_ttr': 0.0,
            'nltk_num_tokens': 0,
            'nltk_num_types': 0
        }

    # Tokenization
    tokens = word_tokenize(text.lower()) # Convert to lower for TTR consistency

    # Filter out non-alphabetic tokens for TTR (optional, but often desired for pure word count)
    alpha_tokens = [word for word in tokens if word.isalpha()]

    # POS Tagging (Internal use only, not returned as a column)
    # This step is essential for RegexpParser to work correctly
    pos_tags = pos_tag(tokens)

    # Basic Syntax: Noun Phrase Chunking
    # Define a grammar for noun phrases (NP)
    grammar = r"""
        NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
        PP: {<IN><NP>}               # Chunk prepositions followed by NP
        VP: {<VB.*><NP|PP|CLAUSE>*}  # Chunk verbs and their arguments
    """
    chunk_parser = RegexpParser(grammar)
    chunks = chunk_parser.parse(pos_tags)

    # Extract chunks in a readable format (e.g., (NP the quick brown fox))
    chunk_phrases = []
    for subtree in chunks.subtrees():
        if subtree.label() != 'S': # 'S' is the sentence root
            chunk_phrases.append((subtree.label(), " ".join([word for word, tag in subtree.leaves()])))

    # Lexical Richness (Type-Token Ratio - TTR)
    num_tokens = len(alpha_tokens)
    num_types = len(set(alpha_tokens))
    ttr = num_types / num_tokens if num_tokens > 0 else 0.0

    return {
        'nltk_chunks': str(chunk_phrases), # Convert list to string for CSV storage
        'nltk_ttr': ttr,
        'nltk_num_tokens': num_tokens,
        'nltk_num_types': num_types
    }

# --- Main Processing ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

if 'message' not in df.columns:
    print("Error: The CSV file must contain a 'message' column.")
    exit()

# Apply NLTK analysis to the 'message' column
print("Applying NLTK analysis (without explicit POS tagging output)...")
# Use .apply() with a lambda function for efficiency
nltk_results = df['message'].apply(analyze_text_nltk)

# Expand the dictionary results into separate columns
df_nltk_results = pd.DataFrame(nltk_results.tolist())
# Concatenate the new columns with the original DataFrame
df = pd.concat([df, df_nltk_results], axis=1)

# Save the updated DataFrame
df.to_csv(output_nltk_file_path, index=False)
print(f"Processed data with NLTK features (no POS tags) saved to '{output_nltk_file_path}'")