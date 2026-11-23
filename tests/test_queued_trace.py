"""The forward diagnostic starts through the normal completed-probe boundary."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QueuedTraceTests(unittest.TestCase):
    def run_stage(self, parent, status, complete):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'results').mkdir()
            (root / 'results/resume-validation-20261112.json').write_text(
                json.dumps({'controller_pid': parent, 'status': status}))
            calls = []
            state = {}

            def run(command, env):
                calls.append(command)
                return SimpleNamespace(returncode=0)

            namespace = {'root': root, 'json': json, 'os': SimpleNamespace(getppid=lambda: 42),
                'state': state, 'save': lambda: None, 'py': '/project/python',
                'environment': lambda: {}, 'subprocess': SimpleNamespace(run=run),
                'completed_probe': lambda tag: {'status': 'passed'} if complete else None}
            source = ROOT / 'results/driver-sources/post-grid-validation-resume.py'
            tree = ast.parse(source.read_text())
            module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                      and node.name == 'trace_after_resumed_embedding'], type_ignores=[])
            exec(compile(module, str(source), 'exec'), namespace)
            namespace['trace_after_resumed_embedding']()
            return calls, state

    def test_active_quality_wrong_owner_and_missing_completion_keep_diagnostic_queued(self):
        for parent, status, complete in [(42, 'running', True), (99, 'running_post_grid_child', True),
                                          (42, 'running_post_grid_child', False)]:
            calls, state = self.run_stage(parent, status, complete)
            self.assertEqual(calls, [])
            self.assertEqual(state, {})

    def test_normal_child_entry_after_complete_probe_runs_existing_trace_command(self):
        calls, state = self.run_stage(42, 'running_post_grid_child', True)
        self.assertEqual(calls, [['/project/python', 'bench/trace_pair.py', '--output',
                                 'results/forward-traces/eager-ngram-r1']])
        self.assertEqual(state['forward_trace']['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
