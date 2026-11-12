"""Offline checks: python -m unittest discover -s tests -v"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluator import parse_llm_response, evaluate_sample  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from ground_truth_checker import check_syntax  # noqa: E402


class CoreTests(unittest.TestCase):
    def test_valid_json(self):
        result = parse_llm_response('{"has_error": true, "error_line": 1, "error_description": "colon missing"}')
        self.assertEqual(result["error_line"], 1)

    def test_clean_code(self):
        self.assertFalse(check_syntax('def add(a,b):\n    return a+b\n')["has_error"])
        self.assertFalse(parse_llm_response('{"has_error": false, "error_line": null, "error_description": ""}')["has_error"])

    def test_bad_types_rejected(self):
        for raw in ['{"has_error":"false","error_line":null,"error_description":""}',
                    '{"has_error":true,"error_line":"1","error_description":"err"}',
                    '{"has_error":true,"error_line":1}',
                    '{"_error": "quota"}', '[]']:
            self.assertTrue(parse_llm_response(raw).get('_parse_error'), raw)

    def test_parse_failure_counts_as_miss(self):
        metrics = compute_metrics([{'ground_truth_has_error': True, 'predicted_has_error': None,
                                    'ground_truth_line': 1, 'predicted_line': None,
                                    'parse_success': False}])
        self.assertEqual(metrics['false_negatives'], 1)
        self.assertEqual(metrics['accuracy'], 0)
        self.assertIsNone(metrics['false_positive_rate'])

    def test_published_dataset_size(self):
        with open(ROOT / 'data' / 'generated_dataset.json', encoding='utf8') as handle:
            self.assertEqual(len(json.load(handle)), 66)

    def test_clean_dataset_labels(self):
        with open(ROOT / 'data' / 'dataset_with_clean.json', encoding='utf8') as handle:
            data = json.load(handle)
        self.assertEqual(len(data), 76)
        for entry in data:
            parser_says_error = check_syntax(entry['buggy_code'])['has_error']
            self.assertEqual(parser_says_error, entry['has_error'], entry['sample_id'])

    def test_clean_sample_scoring(self):
        clean = {'sample_id': 'C001', 'source_file': 'sample_01.py', 'error_type_id': 'none',
                 'error_line': None, 'has_error': False, 'buggy_code': 'x = 1\n'}
        ok = evaluate_sample(clean, 'zero_shot',
                             '{"has_error": false, "error_line": null, "error_description": ""}')
        alarm = evaluate_sample(clean, 'zero_shot',
                                '{"has_error": true, "error_line": 1, "error_description": "x"}')
        self.assertTrue(ok['correct_detection'])
        self.assertFalse(alarm['correct_detection'])
        self.assertFalse(alarm['correct_localization'])
        metrics = compute_metrics([
            {'ground_truth_has_error': False, 'predicted_has_error': False,
             'ground_truth_line': None, 'predicted_line': None, 'parse_success': True},
            {'ground_truth_has_error': False, 'predicted_has_error': True,
             'ground_truth_line': None, 'predicted_line': 1, 'parse_success': True},
            {'ground_truth_has_error': True, 'predicted_has_error': True,
             'ground_truth_line': 2, 'predicted_line': 2, 'parse_success': True},
        ])
        self.assertEqual((metrics['true_negatives'], metrics['false_positives']), (1, 1))
        self.assertEqual(metrics['false_positive_rate'], 0.5)
        self.assertEqual(metrics['specificity'], 0.5)
        self.assertEqual(metrics['precision'], 0.5)

    def test_old_entries_default_to_buggy(self):
        old = {'sample_id': 'S001', 'source_file': 'sample_01.py', 'error_type_id': 'missing_colon',
               'error_line': 1, 'buggy_code': 'def f()\n    pass\n'}
        rec = evaluate_sample(old, 'cot',
                              '{"has_error": true, "error_line": 1, "error_description": "colon"}')
        self.assertTrue(rec['ground_truth_has_error'])
        self.assertTrue(rec['correct_localization'])


if __name__ == '__main__':
    unittest.main()
