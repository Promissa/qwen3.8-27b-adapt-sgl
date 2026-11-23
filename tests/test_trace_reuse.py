"""Completed trace reuse must survive JSON serialization, preserving provenance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import trace_pair


def sources(root):
    (root / 'bench').mkdir()
    names = ['trace_pair.py', 'forward_trace.py', 'capture_forward_trace.py', 'compare_forward_trace.py']
    for name in names:
        (root / 'bench' / name).write_text('recorded ' + name)
    return {name: hashlib.sha256((root / 'bench' / name).read_bytes()).hexdigest() for name in names}


class TraceReuseTests(unittest.TestCase):
    def test_corrected_runner_needs_original_archive_and_collectors_stay_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digests = sources(root)
            completed = {'source_sha256': digests}
            with patch.object(trace_pair, 'ROOT', root):
                self.assertIsNone(trace_pair.retained_sources(completed)['archived_measured_runner'])
                original = (root / 'bench/trace_pair.py').read_bytes()
                (root / 'bench/trace_pair.py').write_text('corrected verifier')
                with self.assertRaisesRegex(ValueError, 'exact archived'):
                    trace_pair.retained_sources(completed)
                archive = root / 'results/driver-sources/archive' / ('trace-pair-' + digests['trace_pair.py'] + '.py')
                archive.parent.mkdir(parents=True)
                archive.write_bytes(original)
                verified = trace_pair.retained_sources(completed)
                self.assertEqual(verified['archived_measured_runner'], str(archive))
                self.assertNotEqual(verified['measured_runner_sha256'], verified['verification_runner_sha256'])
                (root / 'bench/forward_trace.py').write_text('changed collector')
                with self.assertRaisesRegex(ValueError, 'Collector source changed'):
                    trace_pair.retained_sources(completed)
                (root / 'bench/forward_trace.py').write_text('recorded forward_trace.py')
                archive.write_text('wrong runner')
                with self.assertRaisesRegex(ValueError, 'exact archived'):
                    trace_pair.retained_sources(completed)

    def test_json_roundtrip_reuse_is_read_only_and_rejects_changed_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digests = sources(root)
            output = root / 'trace'
            output.mkdir()
            (output / 'pair.json').write_text(json.dumps({'status': 'completed_diagnostic',
                'queue': [], 'source_sha256': digests}))
            for role in ('control', 'ngram'):
                (output / role / 'probe').mkdir(parents=True)
                (output / role / 'tensors/worker-fixture').mkdir(parents=True)
                (output / role / 'probe/probe.json').write_text(json.dumps({'runtime_provenance': {
                    'engine_source': 'owned-engine', 'engine_commit': 'revision',
                    'engine_diff_sha256': hashlib.sha256(b'diff').hexdigest()}}))
            raw_comparison = {'frames': [{'opaque_layouts_not_compared': [('module', 'input', 'value')],
                                        'shared_dense_bit_identical': True}]}
            report = output / 'comparison.json'
            report.write_text(json.dumps(raw_comparison))
            before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}

            def git(command, text=False):
                return 'revision\n' if text else b'diff'

            argv = ['trace_pair.py', '--output', str(output)]
            with patch.object(trace_pair, 'ROOT', root), patch.object(sys, 'argv', argv), \
                 patch.object(trace_pair, 'plans', return_value=[]), \
                 patch.object(trace_pair, 'compare', return_value=raw_comparison), \
                 patch.object(trace_pair.subprocess, 'check_output', side_effect=git), \
                 patch.object(trace_pair.subprocess, 'run', side_effect=AssertionError('Must not start a model')), \
                 patch.object(trace_pair.socket, 'socket', side_effect=AssertionError('Must not bind a server')):
                self.assertEqual(trace_pair.main(), 0)
                self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
                report.write_text(json.dumps({'frames': []}))
                with self.assertRaises(SystemExit) as stopped:
                    trace_pair.main()
                self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
