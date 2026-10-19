# llm-trace-lite

Local, boring tracing for LLM calls. Every chat completion goes into a SQLite
file with model, latency, token counts and an estimated cost, and a small CLI
answers "what did we call, how slow was it, what did it cost".

I wanted this for prototypes and evals where setting up a hosted tracing
product is overkill, but `print()` isn't enough once you're comparing
models. It's one file per project, you can open it with any SQLite tool, and
there are no dependencies.

Works with any OpenAI-compatible `/chat/completions` endpoint (OpenAI,
vLLM, Ollama's `/v1`, a LiteLLM proxy, ...). There's a mock mode so the
demo and tests don't need an API key.

## Usage

```python
from llm_trace_lite import TracedClient, TraceStore

client = TracedClient(store=TraceStore("traces.db"))   # reads OPENAI_API_KEY
resp = client.chat("gpt-4o-mini",
                   [{"role": "user", "content": "Classify: refund request"}],
                   tag="triage", max_tokens=20)
```

Set `OPENAI_BASE_URL` (or pass `base_url=`) to point at another server.
Failed calls are recorded with `status = error` and then re-raised.

## CLI

```
python examples/mock_calls.py demo.db          # writes a few fake calls
python -m llm_trace_lite --db demo.db list -n 10
python -m llm_trace_lite --db demo.db summary --hours 24
python -m llm_trace_lite --db demo.db export --csv -o calls.csv
```

After `pip install -e .` the same commands are available as `trace ...`.
The default database is `./traces.db`, or `$LLM_TRACE_DB`.

```
$ python -m llm_trace_lite --db demo.db summary
model              calls  err  avg_ms  p95_ms  in_tok  out_tok       cost
-----------------  -----  ---  ------  ------  ------  -------  ---------
gpt-4o                 1    0    2099    2099      57      279  $0.002933
claude-3-5-sonnet      1    0    1497    1497      13      131  $0.002004
gpt-4o-mini            4    1     561     571     104       74  $0.000060
some-local-model       1    0    1388    1388      14       66          -

total estimated cost: $0.0050  (approximate, see pricing.py)
```

## Costs

Prices live in `llm_trace_lite/pricing.py` as USD per 1M input/output tokens.
They are approximate list prices typed in by hand, don't cover cached input or
batch discounts, and will go stale. Unknown models get no cost rather than a
wrong one. The Claude entries are placeholders for when you route through an
OpenAI-compatible proxy.

## Tests

```
python -m unittest discover -s tests -t .
```
