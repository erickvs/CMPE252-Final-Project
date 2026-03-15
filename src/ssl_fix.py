import ssl
import certifi

def create_ssl_context():
    """
    Factory to create a default SSL context that uses the installed certifi certificates.
    This fixes SSL: CERTIFICATE_VERIFY_FAILED on macOS.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    return context