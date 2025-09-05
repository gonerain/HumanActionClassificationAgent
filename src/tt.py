import os
from sqlalchemy import create_engine, text
e = create_engine(os.getenv("SP_DB_URL"))
with e.connect() as conn:
    print(conn.execute(text("SELECT current_database(), version()")).fetchone())