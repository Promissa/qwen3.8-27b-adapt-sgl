"""Retain expensive probe evidence while finishing the same bounded search."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from envelope_sweep import bracket_search, load_progress


class EnvelopeResumeTests(unittest.TestCase):
    def test_passing_upper_endpoint_needs_no_intermediate_probes(self):
        called = []
        def check(value):
            called.append(value)
            return True
        best, samples = bracket_search(check, .88, .98, .005)
        self.assertEqual(best, .98)
        self.assertEqual(called, [.88, .98])
        self.assertEqual(len(samples), 2)

    def test_resume_does_not_repeat_existing_expensive_probes(self):
        called = []
        best, samples = bracket_search(lambda value: called.append(value) or True,
            .88, .98, .005, prior_samples={.88: True, .90: True, .92: True, .94: True, .96: True})
        self.assertEqual(called, [.98])
        self.assertEqual(best, .98)
        self.assertEqual(len(samples), 6)

    def test_failed_endpoints_still_discover_middle_pass_interval(self):
        best, samples = bracket_search(lambda value: .91 <= value <= .955, .88, .98, .001)
        self.assertFalse(samples[.88]); self.assertFalse(samples[.98])
        self.assertTrue(.954 < best <= .955)

    def test_resume_rejects_protocol_and_infrastructure_corruption(self):
        label = 'present-disabled-2048-c1-bfloat16'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grid.csv'
            row = {'desktop': 'present', 'strategy': 'disabled', 'prefill_chunk': 2048,
                'concurrency': 1, 'dtype': 'bfloat16', 'mem_fraction': .88,
                'status': 'passed', 'outcome': 'passed'}
            def save():
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=row); writer.writeheader(); writer.writerow(row)
            save()
            complete, samples = load_progress(path, 'present', {label}, .88, .98)
            self.assertEqual(complete, []); self.assertEqual(samples[label], {.88: True})
            with self.assertRaisesRegex(ValueError, 'protocol'):
                load_progress(path, 'absent', {label}, .88, .98)
            row.update(status='failed', outcome='infrastructure_or_unclassified_failure'); save()
            with self.assertRaisesRegex(ValueError, 'infrastructure'):
                load_progress(path, 'present', {label}, .88, .98)
            row.update(status='passed', outcome='passed'); save()
            path.with_suffix('.summary.json').write_text(json.dumps([{'combination': label,
                'samples': {'.88': True}, 'highest_tested_passing_fraction': .98}]))
            with self.assertRaisesRegex(ValueError, 'summary disagrees'):
                load_progress(path, 'present', {label}, .88, .98)


if __name__ == '__main__':
    unittest.main()
