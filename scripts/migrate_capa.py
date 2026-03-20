from src.database.db import init_db, engine
from sqlalchemy import text, inspect

insp = inspect(engine)
cols = [c["name"] for c in insp.get_columns("capa_cases")]
if "priority" not in cols:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE capa_cases ADD COLUMN priority VARCHAR(16) DEFAULT 'MEDIUM'"))
        conn.commit()
    print("Added priority column")
else:
    print("priority column already exists")

init_db()
print("DB migration complete")
