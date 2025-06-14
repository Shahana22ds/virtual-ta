import base64

def get_image_mime_type(b64str):
    header = base64.b64decode(b64str)[:12]
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    elif header.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    elif header.startswith(b'RIFF') and b'WEBP' in header:
        return 'image/webp'
    else:
        return 'application/octet-stream'  # Fallback
