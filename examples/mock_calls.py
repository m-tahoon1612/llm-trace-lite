"""Record a handful of fake calls so there's something to look at.

    python examples/mock_calls.py
    python -m llm_trace_lite summary --db demo.db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_trace_lite import TracedClient, TraceStore  # noqa: E402

db = sys.argv[1] if len(sys.argv) > 1 else "demo.db"
client = TracedClient(store=TraceStore(db), mock=True, seed=7)

tickets = [
    "Customer says the export button does nothing on Safari.",
    "Invoice shows VAT twice for a German address.",
    "Can I change the billing email without contacting support?",
]

for t in tickets:
    client.chat("gpt-4o-mini", [
        {"role": "system", "content": "Classify the support ticket: bug, billing, or question."},
        {"role": "user", "content": t},
    ], tag="ticket-triage", max_tokens=40)

client.chat("gpt-4o", [
    {"role": "user", "content": "Summarise these three tickets for the weekly report:\n" + "\n".join(tickets)},
], tag="weekly-summary", max_tokens=300)

client.chat("claude-3-5-sonnet", [
    {"role": "user", "content": "Draft a polite reply for the VAT issue."},
], tag="reply-draft", max_tokens=250)

client.chat("some-local-model", [
    {"role": "user", "content": "Unknown model, so cost should stay empty."},
], tag="local")

try:
    client.chat("gpt-4o-mini", [{"role": "user", "content": "FAIL on purpose"}], tag="errors")
except RuntimeError as e:
    print("expected error:", e)

print(f"recorded calls into {db}")
