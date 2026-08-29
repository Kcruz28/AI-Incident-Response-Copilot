import os

def raw_data(file_path):
    dir_path = os.path.dirname(file_path)
    files_names = os.listdir(dir_path)
    files_paths = [os.path.join(dir_path, file_name) for file_name in files_names]
    return files_paths
def reading(file_path):
    total_text = ""
    with open(file_path, "r") as f:
        total_text = f.read()
    return total_text

def chunk_text(text, chunk_size=150):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


if __name__ == "__main__":
    file_path = "example.txt"
    files_paths = raw_data(file_path)
    for file_path in files_paths:
        text = reading(file_path)
        chunks = chunk_text(text)
        for chunk in chunks:
            print(chunk)