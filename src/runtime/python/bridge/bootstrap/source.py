def _jetstream_decode_source(source):
    from io import BytesIO
    from tokenize import detect_encoding

    encoding, _ = detect_encoding(BytesIO(source).readline)
    return source.decode(encoding).encode("utf-8")
