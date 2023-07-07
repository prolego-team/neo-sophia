from typing import Optional
import warnings

import chromadb


_db_persist_path = None
_db_impl = None
_is_configured = False
_client = None


def configure_db(
        persist_dir: Optional[str] = '.chroma_cache',
        db_impl: Optional[str] = 'duckdb+parquet') -> None:
    """Set the Chroma DB parameters.
    
    There should never be more than one client, so the initialization
    parameters should be global constants."""
    global _db_persist_path, _db_impl, _is_configured
    if _is_configured:
        warnings.warn(
            '''Configuration of the Chroma DB is being attempted after '''
            '''the client has been created.  The new configuration is '''
            '''being IGNORED.''',
            RuntimeWarning
        )
    _db_persist_path = str(persist_dir)
    _db_impl = db_impl
    _is_configured = True


def get_inmemory_client() -> chromadb.Client:
    """Return a Chroma DB client.
    
    There should never be more than one client."""
    global _client
    if _client is not None:
        return _client
    
    if not _is_configured:
        configure_db()
    
    _client = chromadb.Client(chromadb.Settings(
        chroma_db_impl=_db_impl,
        persist_directory=_db_persist_path,
        anonymized_telemetry=False
    ))

    return _client
