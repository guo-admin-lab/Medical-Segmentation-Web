import base64

# btyes to base64
def bytes2base64(path):
    with open(path, 'rb') as f:
        stream = f.read()
    stream = base64.b64encode(stream).decode()
    return stream

