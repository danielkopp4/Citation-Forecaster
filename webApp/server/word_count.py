# word_count.py

import sys
import json

# Function to count the number of words in the input text
def count_words(text):
    words = text.split()
    return len(words)

if __name__ == "__main__":
    # Read the input text from stdin
    input_text = sys.stdin.read()
    
    # Call the count_words function to get the word count
    word_count = count_words(input_text)
    
    # Create a JSON object containing the word count
    word_count_json = json.dumps({"word_count": word_count})
    
    # Print the JSON object to stdout
    print(word_count_json)
