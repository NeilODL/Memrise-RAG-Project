import os
import tiktoken

# Set up tokenizer
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Folder path
folder_path = "sample_data"

# Token count dictionary
token_counts = {}

# Iterate through .txt files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tokens = enc.encode(content)
        token_counts[filename] = len(tokens)

# Print results
for file, count in token_counts.items():
    print(f"{file}: {count} tokens")
