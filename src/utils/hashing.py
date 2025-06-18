import hashlib

def get_file_hash(filepath):
    """Tính hash MD5 của một file."""
    h = hashlib.md5()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest() 