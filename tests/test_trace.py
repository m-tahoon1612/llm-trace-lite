import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from llm_trace_lite import TracedClient, TraceStore
from llm_trace_lite.cli import main
from llm_trace_lite.pricing import estimate_cost


class PricingTests(unittest.TestCase):
    def test_known_model(self):
        # 1M in + 1M out on gpt-4o-mini = 0.15 + 0.60
        self.assertAlmostEqual(estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000), 0.75)

    def test_dated_snapshot_uses_prefix(self):
        self.assertEqual(estimate_cost("gpt-4o-mini-2024-07-18", 1000, 0),
                         estimate_cost("gpt-4o-mini", 1000, 0))

    def test_unknown_model(self):
        self.assertIsNone(estimate_cost("llama-local", 10, 10))


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.store = TraceStore(self.db)
        self.client = TracedClient(store=self.store, mock=True, seed=1)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_records_ok_call(self):
        resp = self.client.chat("gpt-4o", [{"role": "user", "content": "hi"}], tag="t")
        self.assertIn("choices", resp)
        self.assertNotIn("_mock_latency_ms", resp)
        (row,) = self.store.recent()
        self.assertEqual(row["model"], "gpt-4o")
        self.assertEqual(row["status"], "ok")
        self.assertGreater(row["cost_usd"], 0)
        self.assertEqual(row["tag"], "t")

    def test_records_error_and_reraises(self):
        with self.assertRaises(RuntimeError):
            self.client.chat("gpt-4o-mini", [{"role": "user", "content": "FAIL"}])
        (row,) = self.store.recent()
        self.assertEqual(row["status"], "error")

    def test_requires_key_without_mock(self):
        env = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                TracedClient(store=self.store)
        finally:
            if env:
                os.environ["OPENAI_API_KEY"] = env

    def test_cli_summary_and_export(self):
        for _ in range(3):
            self.client.chat("gpt-4o-mini", [{"role": "user", "content": "x"}])
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--db", self.db, "summary"]), 0)
        self.assertIn("gpt-4o-mini", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--db", self.db, "export", "--csv"])
        lines = buf.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 4)  # header + 3
        self.assertTrue(lines[0].startswith("id,ts,model"))


if __name__ == "__main__":
    unittest.main()
