def count_words_in_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        words = text.split()
        return len(words)


# Replace 'your_file.md' with the actual file path
md_file_path = "paper.md"
word_count = count_words_in_md(md_file_path)
print(f"Word count: {word_count}")
