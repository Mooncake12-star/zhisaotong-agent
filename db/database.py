import sqlite3
import json
import os

DB_PATH = None
_INITIALIZED = False


def get_db_path():
    global DB_PATH
    if DB_PATH is None:
        from utils.path_tool import get_abs_path
        DB_PATH = get_abs_path("data/zst.db")
    return DB_PATH


def get_connection():
    global _INITIALIZED
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    if not _INITIALIZED:
        _init_db(conn)
        _seed_data(conn)
        _INITIALIZED = True
    return conn


def _init_db(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id          TEXT PRIMARY KEY,
            brand       TEXT,
            name        TEXT NOT NULL,
            type        TEXT,
            price       REAL,
            specs       TEXT,
            features    TEXT,
            suitable_for TEXT,
            rating      REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            city        TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS external_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            month       TEXT,
            record_data TEXT
        )
    """)
    conn.commit()


def _seed_data(conn):
    c = conn.cursor()

    # 从 JSON 导入产品（仅在首次）
    count = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if count == 0:
        from utils.path_tool import get_abs_path
        json_path = get_abs_path("data/products.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                products = json.load(f)
            for p in products:
                c.execute("""
                    INSERT INTO products (id, brand, name, type, price, specs, features, suitable_for, rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p["id"], p["brand"], p["name"], p["type"], p["price"],
                    json.dumps(p.get("specs", {}), ensure_ascii=False),
                    json.dumps(p.get("features", []), ensure_ascii=False),
                    json.dumps(p.get("suitable_for", []), ensure_ascii=False),
                    p.get("rating", 0)
                ))
            print(f"[DB] 已导入 {len(products)} 条产品数据")

    # 导入模拟使用记录（仅在首次）
    count = c.execute("SELECT COUNT(*) FROM external_records").fetchone()[0]
    if count == 0:
        sample = [
            (1, "2026-06", json.dumps({"扫地次数": 22, "平均时长": "45分钟", "耗材": "滚刷正常"}, ensure_ascii=False)),
            (1, "2026-05", json.dumps({"扫地次数": 18, "平均时长": "40分钟", "耗材": "滚刷正常"}, ensure_ascii=False)),
            (2, "2026-06", json.dumps({"扫地次数": 15, "平均时长": "30分钟", "耗材": "滤网需更换"}, ensure_ascii=False)),
        ]
        for uid, month, data in sample:
            c.execute("INSERT INTO external_records (user_id, month, record_data) VALUES (?, ?, ?)", (uid, month, data))
        print("[DB] 已导入模拟使用记录")

    conn.commit()
