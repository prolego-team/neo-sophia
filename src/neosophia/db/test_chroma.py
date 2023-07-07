from pathlib import Path

import chromadb

from neosophia.db import chroma

def test_chroma(tmp_path):
    persist_path = tmp_path / '.chroma_test'
    
    chroma.configure_db(persist_path)
    db_client = chroma.get_inmemory_client()
    assert isinstance(db_client, chromadb.api.local.LocalAPI)

    db_client2 = chroma.get_inmemory_client()
    assert db_client is db_client2
