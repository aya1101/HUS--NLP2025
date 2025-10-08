def load_raw_text_data(file_path: str) -> str:
    """Load raw text data from a given file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()