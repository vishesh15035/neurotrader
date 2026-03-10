"""
Memory System — Short-term + Long-term
Short-term: last N reasoning steps in RAM
Long-term: SQLite — persists decisions, outcomes, patterns
"""
import sqlite3
import json
from datetime import datetime
from collections import deque
from pathlib import Path

class ShortTermMemory:
    def __init__(self, max_size: int = 10):
        self.buffer   = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, role: str, content: str, metadata: dict = None):
        self.buffer.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().isoformat(),
            "metadata":  metadata or {}
        })

    def get_context(self) -> list:
        return [{"role": m["role"], "content": m["content"]} for m in self.buffer]

    def get_recent(self, n: int = 3) -> list:
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()

    def summary(self) -> str:
        if not self.buffer:
            return "No recent memory."
        lines = []
        for m in list(self.buffer)[-5:]:
            ts   = m["timestamp"][:19]
            role = m["role"].upper()
            txt  = m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"]
            lines.append(f"[{ts}] {role}: {txt}")
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self, db_path: str = "data/memory.db"):
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                ticker      TEXT,
                action      TEXT,
                reasoning   TEXT,
                price       REAL,
                confidence  REAL,
                outcome     TEXT
            );
            CREATE TABLE IF NOT EXISTS patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                pattern     TEXT,
                context     TEXT,
                result      TEXT,
                score       REAL
            );
            CREATE TABLE IF NOT EXISTS market_insights (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                ticker      TEXT,
                insight     TEXT,
                tags        TEXT
            );
        """)
        self.conn.commit()

    def log_decision(self, ticker: str, action: str, reasoning: str,
                     price: float, confidence: float = 0.5):
        self.conn.execute("""
            INSERT INTO decisions (timestamp, ticker, action, reasoning, price, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), ticker, action, reasoning, price, confidence))
        self.conn.commit()

    def update_outcome(self, decision_id: int, outcome: str):
        self.conn.execute("UPDATE decisions SET outcome=? WHERE id=?", (outcome, decision_id))
        self.conn.commit()

    def log_pattern(self, pattern: str, context: str, result: str, score: float):
        self.conn.execute("""
            INSERT INTO patterns (timestamp, pattern, context, result, score)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), pattern, context, result, score))
        self.conn.commit()

    def log_insight(self, ticker: str, insight: str, tags: list):
        self.conn.execute("""
            INSERT INTO market_insights (timestamp, ticker, insight, tags)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), ticker, insight, json.dumps(tags)))
        self.conn.commit()

    def get_past_decisions(self, ticker: str = None, limit: int = 5) -> list:
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM decisions WHERE ticker=? ORDER BY timestamp DESC LIMIT ?",
                (ticker, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"id":r[0],"timestamp":r[1],"ticker":r[2],"action":r[3],
                 "reasoning":r[4],"price":r[5],"confidence":r[6],"outcome":r[7]} for r in rows]

    def get_performance_summary(self) -> dict:
        rows = self.conn.execute("SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall()
        total = self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        return {"total_decisions": total, "by_action": {r[0]: r[1] for r in rows}}

    def get_insights(self, ticker: str = None, limit: int = 3) -> list:
        if ticker:
            rows = self.conn.execute(
                "SELECT insight, tags FROM market_insights WHERE ticker=? ORDER BY timestamp DESC LIMIT ?",
                (ticker, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT insight, tags FROM market_insights ORDER BY timestamp DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"insight": r[0], "tags": json.loads(r[1])} for r in rows]
