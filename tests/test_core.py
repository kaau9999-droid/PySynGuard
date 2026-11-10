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


if __name__ == '__main__':
    unittest.main()
