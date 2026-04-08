import logging
import requests
import requests_cache

logger = logging.getLogger(__name__)

# Try to use curl_cffi for better impersonation if available
try:
    from curl_cffi import requests as cffi_requests
    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False

# Try to install SQLite cache for requests
try:
    requests_cache.install_cache(
        'finance_cache', backend='sqlite', expire_after=900,
        backend_options={'pragmas': {'journal_mode': 'wal'}},
    )
except Exception:
    try:
        requests_cache.install_cache('finance_cache', backend='sqlite', expire_after=900)
    except Exception:
        pass

def get_session(timeout: int = 20):
    """
    Return a centralized HTTP session for API requests.
    Attempts to use curl_cffi to prevent blocking, falls back to requests.Session.
    """
    if _HAS_CFFI:
        try:
            session = cffi_requests.Session(impersonate="chrome")
            session.timeout = timeout
            return session
        except Exception as e:
            logger.debug(f"Failed to initialize curl_cffi session, falling back to standard requests: {e}")
            pass
            
    # Fallback to standard requests
    session = requests.Session()
    # Add a default User-Agent
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    session.timeout = timeout
    return session

# Create a singleton session to be shared across modules
_global_session = get_session()

def get_global_session():
    """Return the shared global HTTP session."""
    return _global_session
