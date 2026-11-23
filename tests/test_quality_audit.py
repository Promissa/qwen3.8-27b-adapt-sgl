"""A green rescore status cannot hide stale hashes or missing/altered rows."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from audit_quality_run import audit, validate_rescore


class QualityAuditTests(unittest.TestCase):
    def evidence(self):
        rows = {f'gsm8k-{i}': {'expected_answer': '7', 'predicted_answer': '7',
                              'symbolic_correct': True} for i in range(1319)}
        run = {'directory': '/evidence/run', 'correct': 1319,
               'evidence_sha256': {'output-rs0.jsonl': 'current'}, 'rows': rows}
        result = {'status': 'passed', 'checked': 1319, 'n': 1319, 'correct': 1319,
            'run_directory': run['directory'], 'original_evidence_sha256': run['evidence_sha256'],
            'score_mismatch_ids': [], 'extraction_mismatch_ids': [],
            'rows': [dict(row, id=identifier) for identifier, row in rows.items()]}
        return run, result

    def test_complete_status_does_not_waive_evidence_identity_or_count(self):
        run, report = self.evidence()
        validate_rescore(report, run)
        for key, value in [('checked', 1318), ('original_evidence_sha256', {}),
                           ('run_directory', '/other/run'), ('score_mismatch_ids', ['gsm8k-0'])]:
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'current evidence'):
                validate_rescore(changed, run)

    def test_repeated_ids_and_changed_extractions_reject_green_status(self):
        run, report = self.evidence()
        changed = copy.deepcopy(report)
        changed['rows'][-1] = copy.deepcopy(changed['rows'][0])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_rescore(changed, run)
        changed = copy.deepcopy(report)
        changed['rows'][0]['predicted_answer'] = '8'
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            validate_rescore(changed, run)

    def test_existing_audit_directory_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                audit('example', Path(tmp))
            with self.assertRaisesRegex(ValueError, 'plain evaluation tag'):
                audit('../other', Path(tmp) / 'new')
            self.assertFalse((Path(tmp) / 'new').exists())


if __name__ == '__main__':
    unittest.main()
