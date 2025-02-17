def read_data(path: str = "tinyshakespeare.txt"):
    with open(path, "r") as f:
        data = f.read()
    return data
