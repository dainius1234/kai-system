from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2


def main() -> None:
    uri = os.getenv("PG_URI", "postgresql://keeper:localdev@postgres:5432/sovereign")
    print(f"initializing memu-core database at {uri}")
    conn = psycopg2.connect(uri)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id text PRIMARY KEY,
            timestamp text,
            event_type text,
            content jsonb,
            embedding vector,
            relevance float,
            pinned bool
        );
        """
    )
    conn.commit()
    conn.close()
    print("schema created")


if __name__ == "__main__":
    main()
