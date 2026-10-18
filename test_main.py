"""Các bài kiểm thử đơn vị cơ bản của thuật toán KNN."""

import unittest
from pathlib import Path
from main import FEATURES, knn_predict, load_data, scale


class TestKNN(unittest.TestCase):
    def test_nearest_neighbor(self):
        data = [((0.0, 0.0), "iPhone"), ((1.0, 1.0), "Android")]
        self.assertEqual(knn_predict(data, (0.1, 0.1), 1), "iPhone")

    def test_majority_vote(self):
        data = [((0.0,), "Android"), ((0.2,), "Android"),
                ((0.1,), "iPhone")]
        self.assertEqual(knn_predict(data, (0.11,), 3), "Android")

    def test_tie_break_by_nearest(self):
        data = [((0.0,), "iPhone"), ((1.0,), "Android")]
        self.assertEqual(knn_predict(data, (0.1,), 2), "iPhone")

    def test_min_max(self):
        self.assertEqual(scale((20, 4), [10, 1], [20, 4]),
                         (0.5, 0.75))

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            knn_predict([((0.0,), "iPhone")], (0.1,), 2)

    def test_only_two_features(self):
        self.assertEqual(FEATURES, ("budget", "camera"))

    def test_csv_labels_and_columns(self):
        rows = load_data(Path(__file__).with_name("data.csv"))
        self.assertGreaterEqual(len(rows), 10)
        self.assertEqual({label for _, label in rows}, {"iPhone", "Android"})
        self.assertTrue(all(len(point) == 2 for point, _ in rows))


if __name__ == "__main__":
    unittest.main()
