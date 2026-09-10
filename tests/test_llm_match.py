import unittest

from llm_match import (
    compute_llm_match_spl_score,
    compute_statistics,
    normalize_llm_match_score,
)


class LLMMatchTest(unittest.TestCase):
    def test_score_and_spl_statistics(self):
        self.assertEqual(normalize_llm_match_score(1), 0.0)
        self.assertEqual(normalize_llm_match_score(5), 100.0)
        self.assertEqual(
            compute_llm_match_spl_score("q", 5, {"q": 10}, {"q": 5}),
            50.0,
        )
        statistics = compute_statistics(
            {"q": (5, 50.0)},
            total_questions=2,
            total_predictions=1,
            blind_scores={"failed": 1},
        )
        self.assertEqual(statistics["original_llm_match_mean"], 50.0)
        self.assertEqual(statistics["grounding_llm_match_mean"], 50.0)
        self.assertEqual(statistics["original_llm_match_spl_mean"], 25.0)


if __name__ == "__main__":
    unittest.main()
