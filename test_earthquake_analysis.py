"""
Correctness tests: every Wavelet Tree answer is compared with a simple
Python calculation (sorting / counting the selected slice).

Run:  python test_earthquake_analysis.py        (or: python -m unittest -v)
"""
import os
import random
import tempfile
import unittest
from datetime import datetime

from earthquake_analysis import (EarthquakeAnalyzer, WaveletTree,
                                 compress_magnitudes, load_earthquakes,
                                 parse_magnitude, parse_user_date)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(HERE, "sample_earthquakes.csv")


# ---------- simple reference implementations (the "plain Python" side) ----
def simple_kth_smallest(values, k):
    return sorted(values)[k - 1]

def simple_kth_largest(values, k):
    return sorted(values, reverse=True)[k - 1]

def simple_count_at_least(values, threshold):
    return sum(1 for value in values if value >= threshold)

def simple_median(values):
    ordered, n = sorted(values), len(values)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def build_tree(magnitudes):
    distinct, ranks = compress_magnitudes(magnitudes)
    return WaveletTree(ranks, len(distinct)), distinct


class TestCompression(unittest.TestCase):
    def test_order_and_duplicates(self):
        distinct, ranks = compress_magnitudes([4.5, 7.1, 4.5, 3.0])
        self.assertEqual(distinct, [3.0, 4.5, 7.1])
        self.assertEqual(ranks, [1, 2, 1, 0])


class TestWaveletTreeAgainstPlainPython(unittest.TestCase):
    def test_random_arrays_all_queries(self):
        random.seed(12345)
        for _ in range(150):
            n = random.randint(1, 60)
            # few distinct values on purpose -> lots of duplicates
            magnitudes = [round(random.choice([0.5, 1.0, 2.5, 3.0, 3.0, 4.9, 5.0,
                                               5.1, 6.9, 7.0, 8.8]) , 1) for _ in range(n)]
            tree, distinct = build_tree(magnitudes)
            for _ in range(20):
                left = random.randint(0, n - 1)
                right = random.randint(left + 1, n)
                part = magnitudes[left:right]
                for k in range(1, len(part) + 1):
                    self.assertEqual(distinct[tree.kth_smallest_rank(left, right, k)],
                                     simple_kth_smallest(part, k))
                    self.assertEqual(distinct[tree.kth_largest_rank(left, right, k)],
                                     simple_kth_largest(part, k))
                from bisect import bisect_left
                for threshold in (-1, 0.5, 3.0, 4.95, 5.0, 7.0, 8.8, 9.0, 100):
                    rank = bisect_left(distinct, threshold)
                    smaller = tree.count_less_than_rank(left, right, rank)
                    self.assertEqual((right - left) - smaller,
                                     simple_count_at_least(part, threshold))

    def test_all_equal_values(self):
        tree, distinct = build_tree([5.0] * 10)
        self.assertEqual(distinct[tree.kth_smallest_rank(2, 9, 4)], 5.0)

    def test_single_element(self):
        tree, distinct = build_tree([6.1])
        self.assertEqual(distinct[tree.kth_largest_rank(0, 1, 1)], 6.1)


class TestAnalyzerOnSampleFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.report = load_earthquakes(SAMPLE_CSV)
        cls.analyzer = EarthquakeAnalyzer(cls.records)

    def day(self, text):
        return datetime.strptime(text, "%Y-%m-%d")

    def test_validation_report(self):
        self.assertEqual(self.report["kept"], 28)
        self.assertEqual(self.report["bad_magnitude"], 3)   # empty, 'abc', 99
        self.assertEqual(self.report["bad_time"], 1)
        self.assertEqual(self.report["duplicate_id"], 1)

    def test_sorted_chronologically(self):
        times = [quake.timestamp for quake in self.records]
        self.assertEqual(times, sorted(times))

    def test_date_range_boundaries_are_inclusive(self):
        # 2020-01-28 has exactly one quake; a one-day range must find it.
        left, right = self.analyzer.date_range_to_indices(self.day("2020-01-28"),
                                                          self.day("2020-01-28"))
        self.assertEqual(right - left, 1)
        self.assertEqual(self.analyzer.kth_largest(left, right, 1), 7.7)

    def test_example_queries_2020(self):
        a = self.analyzer
        left, right = a.date_range_to_indices(self.day("2020-01-01"), self.day("2020-12-31"))
        part = [q.magnitude for q in self.records[left:right]]
        self.assertEqual(len(part), 10)
        self.assertEqual(a.kth_largest(left, right, 3), 7.0)
        self.assertEqual(a.kth_smallest(left, right, 1), 2.9)
        self.assertEqual(a.count_at_least(left, right, 5.0), 6)
        self.assertAlmostEqual(a.median(left, right), simple_median(part))
        self.assertEqual(a.distribution(left, right),
                         [("Below 3.0", 1), ("3.0 - 4.9", 3),
                          ("5.0 - 6.9", 3), ("7.0 and above", 3)])

    def test_distribution_is_a_partition(self):
        a = self.analyzer
        n = len(self.records)
        buckets = a.distribution(0, n)
        self.assertEqual(sum(count for _, count in buckets), n)

    def test_boundary_values_belong_to_one_bucket_only(self):
        # sample has magnitudes exactly 3.0, 5.0 and 7.0
        a = self.analyzer
        n = len(self.records)
        counts = dict(a.distribution(0, n))
        plain = {"Below 3.0": 0, "3.0 - 4.9": 0, "5.0 - 6.9": 0, "7.0 and above": 0}
        for q in self.records:
            m = q.magnitude
            key = ("Below 3.0" if m < 3 else "3.0 - 4.9" if m < 5
                   else "5.0 - 6.9" if m < 7 else "7.0 and above")
            plain[key] += 1
        self.assertEqual(counts, plain)

    def test_median_even_and_odd(self):
        a = self.analyzer
        for left, right in [(0, 4), (0, 5), (3, 20), (0, len(self.records))]:
            part = [q.magnitude for q in self.records[left:right]]
            self.assertAlmostEqual(a.median(left, right), simple_median(part))

    def test_empty_range_and_bad_k(self):
        a = self.analyzer
        left, right = a.date_range_to_indices(self.day("1990-01-01"), self.day("1990-12-31"))
        self.assertEqual(right - left, 0)
        self.assertIsNone(a.median(left, right))
        self.assertIsNone(a.maximum(left, right))
        with self.assertRaises(ValueError):
            a.kth_largest(left, right, 1)
        with self.assertRaises(ValueError):
            a.kth_smallest(0, 5, 0)
        with self.assertRaises(ValueError):
            a.kth_smallest(0, 5, 6)

    def test_start_after_end_rejected(self):
        with self.assertRaises(ValueError):
            self.analyzer.date_range_to_indices(self.day("2021-01-01"), self.day("2020-01-01"))

    def test_user_date_parsing(self):
        self.assertEqual(parse_user_date("2020", False), datetime(2020, 1, 1))
        self.assertEqual(parse_user_date("2020", True), datetime(2020, 12, 31))
        for bad in ("2020-13-01", "31/12/2020", "hello", "2020-02-30"):
            with self.assertRaises(ValueError):
                parse_user_date(bad, False)

    def test_compare_two_periods(self):
        a = self.analyzer
        l1, r1 = a.date_range_to_indices(self.day("2019-01-01"), self.day("2020-12-31"))
        l2, r2 = a.date_range_to_indices(self.day("2021-01-01"), self.day("2022-12-31"))
        first = a.summarize_period(l1, r1, 7.0)
        second = a.summarize_period(l2, r2, 7.0)
        self.assertEqual((first["count"], first["maximum"], first["above_threshold"]), (15, 7.7, 4))
        self.assertEqual((second["count"], second["maximum"], second["above_threshold"]), (13, 8.2, 7))


class TestLoaderAndMagnitudes(unittest.TestCase):
    def test_magnitude_validation(self):
        for bad in ("", "abc", "nan", "inf", "99", "-50"):
            with self.assertRaises(ValueError):
                parse_magnitude(bad)
        self.assertEqual(parse_magnitude("-0.3"), -0.3)

    def test_missing_required_column(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write("time,latitude\n2020-01-01T00:00:00Z,1\n")
        try:
            with self.assertRaises(ValueError):
                load_earthquakes(handle.name)
        finally:
            os.remove(handle.name)


class TestLargeRandomDataset(unittest.TestCase):
    def test_20000_records(self):
        random.seed(7)
        magnitudes = [round(2.5 + random.expovariate(1.6), 1) for _ in range(20000)]
        tree, distinct = build_tree(magnitudes)
        for _ in range(300):
            left = random.randint(0, 19999)
            right = random.randint(left + 1, 20000)
            part = magnitudes[left:right]
            k = random.randint(1, len(part))
            self.assertEqual(distinct[tree.kth_smallest_rank(left, right, k)],
                             simple_kth_smallest(part, k))
            self.assertEqual(distinct[tree.kth_largest_rank(left, right, k)],
                             simple_kth_largest(part, k))


if __name__ == "__main__":
    unittest.main(verbosity=2)
