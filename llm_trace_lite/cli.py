import argparse
import sys
import time
from datetime import datetime

from .store import TraceStore, default_path


def _fmt_cost(v) -> str:
    return "-" if v is None else f"${v:.6f}"


def _table(head: list[str], rows: list[list]) -> str:
    """Plain text table; numbers right-aligned, everything else left."""
    numeric = [all(isinstance(r[i], (int, float)) or str(r[i]).startswith("$") or r[i] == "-"
                   for r in rows) for i in range(len(head))]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]

    def fmt(r):
        return "  ".join(str(c).rjust(w) if num else str(c).ljust(w)
                         for c, w, num in zip(r, widths, numeric)).rstrip()

    return "\n".join([fmt(head), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])


def cmd_list(store: TraceStore, args) -> int:
    rows = store.recent(args.limit, args.model)
    if not rows:
        print("no calls recorded")
        return 0
    out = []
    for r in rows:
        when = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        out.append([r["id"], when, r["model"], r["latency_ms"], r["prompt_tokens"] or "-",
                    r["completion_tokens"] or "-", _fmt_cost(r["cost_usd"]), r["status"],
                    r["tag"] or ""])
    print(_table(["id", "time", "model", "ms", "in", "out", "cost", "status", "tag"], out))
    return 0


def cmd_summary(store: TraceStore, args) -> int:
    since = time.time() - args.hours * 3600 if args.hours else None
    rows = store.summary(since)
    if not rows:
        print("no calls recorded")
        return 0
    out, total = [], 0.0
    for r in rows:
        total += r["cost_usd"] or 0
        out.append([r["model"], r["calls"], r["errors"], int(r["avg_ms"] or 0),
                    r["prompt_tokens"],
                    r["completion_tokens"], _fmt_cost(r["cost_usd"])])
    print(_table(["model", "calls", "err", "avg_ms", "in_tok", "out_tok", "cost"], out))
    print(f"\ntotal estimated cost: ${total:.4f}  (approximate, see pricing.py)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trace", description="Inspect recorded LLM calls.")
    p.add_argument("--db", default=None, help=f"sqlite file (default: {default_path()})")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="most recent calls")
    ls.add_argument("-n", "--limit", type=int, default=20)
    ls.add_argument("--model")

    sm = sub.add_parser("summary", help="per-model totals")
    sm.add_argument("--hours", type=float, help="only the last N hours")

    args = p.parse_args(argv)
    store = TraceStore(args.db)
    try:
        return {"list": cmd_list, "summary": cmd_summary}[args.cmd](store, args)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
