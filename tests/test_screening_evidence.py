"""Keep reduced screening separate from the original full-quality gate."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from budget_screen import read_subset
from compare_quality import read_run


class ScreeningEvidenceTests(unittest.TestCase):
    def fixture(self, folder):
        rows = [{'id': f'gsm8k-{i}', 'symbolic_correct': True,
                 'finish_reason': 'stop', 'num_generated_tokens': 10} for i in range(2)]
        (folder / 'output-rs0.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (folder / 'metrics.json').write_text(json.dumps({'name': 'gsm8k', 'num_examples': 2,
            'n_repeats': 1, 'generation': {'max_tokens': 16384}, 'aggregate': {'score': 1.}}))
        return rows

    def test_complete_subset_still_cannot_pass_full_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            self.assertEqual(len(read_subset(folder)[1]), 2)
            with self.assertRaises(ValueError):
                read_run(folder)

    def test_missing_duplicate_and_clamped_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = self.fixture(folder)
            for invalid in (rows[:1], [rows[0], rows[0]],
                            [dict(rows[0], finish_reason='length'), rows[1]]):
                (folder / 'output-rs0.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in invalid))
                with self.assertRaises(ValueError):
                    read_subset(folder)

    def test_cli_keeps_full_default_and_explicit_subset(self):
        for flags, expected in (([], 1319), (['--num-examples', '64'], 64)):
            result = subprocess.check_output([sys.executable, str(ROOT / 'bench/eval_quality.py'),
                '--tag', 'screening-unit-dry-run', '--dry-run', *flags], text=True)
            self.assertEqual(json.loads(result)['num_examples'], expected)


if __name__ == '__main__':
    unittest.main()
