import json
import os
import sqlite3
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    model             TEXT    NOT NULL,
    latency_ms        INTEGER NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    status            TEXT    NOT NULL,   -- ok / error
    tag               TEXT,
    request_json      TEXT,
    response_text     TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS calls_ts ON calls(ts);
"""

COLUMNS = ("id", "ts", "model", "latency_ms", "prompt_tokens", "completion_tokens",
           "cost_usd", "status", "tag", "request_json", "response_text", "error")


def default_path() -> str:
    return os.environ.get("LLM_TRACE_DB", "traces.db")


@dataclass
class Call:
    model: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    status: str = "ok"
    tag: str | None = None
    request: dict | None = None
    response_text: str | None = None
    error: str | None = None
    ts: float | None = None


class TraceStore:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def record(self, c: Call) -> int:
        cur = self.conn.execute(
            "INSERT INTO calls (ts, model, latency_ms, prompt_tokens, completion_tokens,"
            " cost_usd, status, tag, request_json, response_text, error)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (c.ts or time.time(), c.model, c.latency_ms, c.prompt_tokens, c.completion_tokens,
             c.cost_usd, c.status, c.tag, json.dumps(c.request) if c.request else None,
             c.response_text, c.error),
        )
        self.conn.commit()
        return cur.lastrowid

    def recent(self, limit: int = 20, model: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM calls"
        args: list = []
        if model:
            sql += " WHERE model = ?"
            args.append(model)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(limit)
        return self.conn.execute(sql, args).fetchall()

    def summary(self, since_ts: float | None = None) -> list[sqlite3.Row]:
        where = "WHERE ts >= ?" if since_ts else ""
        args = [since_ts] if since_ts else []
        # no percentile functions in sqlite, the CLI computes p95 from latencies()
        return self.conn.execute(
            f"""
            SELECT model,
                   count(*)                                   AS calls,
                   sum(status = 'error')                      AS errors,
                   round(avg(CASE WHEN status = 'ok' THEN latency_ms END)) AS avg_ms,
                   max(latency_ms)                                        AS max_ms,
                   coalesce(sum(prompt_tokens), 0)            AS prompt_tokens,
                   coalesce(sum(completion_tokens), 0)        AS completion_tokens,
                   sum(cost_usd)                              AS cost_usd
            FROM calls {where}
            GROUP BY model
            ORDER BY cost_usd DESC NULLS LAST
            """,
            args,
        ).fetchall()

    def latencies(self, model: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT latency_ms FROM calls WHERE model = ? AND status = 'ok' ORDER BY latency_ms", (model,))
        return [r[0] for r in rows]

    def close(self):
        self.conn.close()
