# word_count.py
import sys
import json

def count_words(text):
    words = text.split()
    return len(words)

if __name__ == "__main__":
    input_text = sys.stdin.read()
    word_count = count_words(input_text)
    print(json.dumps({"word_count": word_count}))