"""
Earthquake Magnitude Range Analysis Using a Wavelet Tree
=========================================================


Run the menu :   python earthquake_analysis.py [path/to/earthquakes.csv]
Run the demo :   python earthquake_analysis.py --demo [path/to/earthquakes.csv]

Program workflow (matches the problem statement)
------------------------------------------------
 1. Load the CSV file                      -> load_earthquakes()
 2. Validate records / skip bad rows       -> load_earthquakes()
 3. Sort records chronologically           -> load_earthquakes()
 4. Extract magnitudes into an array       -> EarthquakeAnalyzer.__init__()
 5. Coordinate compression                 -> compress_magnitudes()
 6. Build the Wavelet Tree                 -> WaveletTree
 7. Accept date-range queries              -> menu functions
 8. Dates -> index boundaries (bisect)     -> EarthquakeAnalyzer.date_range_to_indices()
 9. Run queries on the Wavelet Tree        -> EarthquakeAnalyzer.*
10. Display results                        -> menu functions

Index convention used everywhere: a range is HALF-OPEN [left, right), i.e.
it contains positions left, left+1, ..., right-1 and has (right - left) items.
"""

import csv
import math
import sys
import time
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta


# ======================================================================
# PART 1 - LOADING, VALIDATION AND SORTING
# ======================================================================

@dataclass
class Earthquake:
    """One earthquake record (only the fields we need)."""
    event_id: str
    timestamp: datetime      # UTC time of the earthquake
    magnitude: float
    latitude: float          # may be None if missing in the CSV
    longitude: float         # may be None if missing in the CSV
    depth_km: float          # may be None if missing in the CSV
    place: str


def parse_timestamp(text):
    """Convert a USGS time string such as '2020-01-28T19:55:30.000Z'
    into a datetime. Raises ValueError if the text is not a valid date."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError("unrecognised date/time: " + repr(text))


def parse_magnitude(text):
    """Convert text to a magnitude. Raises ValueError when the value is
    empty, not a number, NaN/infinity, or physically impossible."""
    value = float(text)                      # ValueError if empty / not numeric
    if not math.isfinite(value):
        raise ValueError("magnitude is NaN or infinite")
    if value < -3.0 or value > 10.0:         # tiny quakes can be negative
        raise ValueError("magnitude outside the plausible range -3..10")
    return value


def parse_optional_float(text):
    """Latitude / longitude / depth are not needed for the queries, so a
    bad value there gives None instead of rejecting the whole row."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_earthquakes(csv_path):
    """Read the CSV, validate every row and return (records, report).

    * Rows with a missing/invalid magnitude or time are skipped.
    * Rows whose event id was already seen are skipped (duplicate events).
      NOTE: duplicate MAGNITUDE values are perfectly fine and are kept.
    * The returned list is sorted chronologically (ties broken by id so the
      order is always the same).
    """
    report = {"rows_read": 0, "kept": 0, "bad_magnitude": 0,
              "bad_time": 0, "duplicate_id": 0}
    records = []
    seen_ids = set()

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        for required in ("time", "mag"):
            if required not in columns:
                raise ValueError("CSV is missing the required column '%s'" % required)

        for row in reader:
            report["rows_read"] += 1

            try:
                magnitude = parse_magnitude(row.get("mag") or "")
            except ValueError:
                report["bad_magnitude"] += 1
                continue

            try:
                timestamp = parse_timestamp(row.get("time") or "")
            except ValueError:
                report["bad_time"] += 1
                continue

            event_id = (row.get("id") or "").strip()
            if event_id:
                if event_id in seen_ids:
                    report["duplicate_id"] += 1
                    continue
                seen_ids.add(event_id)

            records.append(Earthquake(
                event_id=event_id,
                timestamp=timestamp,
                magnitude=magnitude,
                latitude=parse_optional_float(row.get("latitude")),
                longitude=parse_optional_float(row.get("longitude")),
                depth_km=parse_optional_float(row.get("depth")),
                place=(row.get("place") or "").strip(),
            ))

    # Chronological order is REQUIRED: the Wavelet Tree is built on the
    # magnitudes in this order, so "a time period" = "a contiguous slice".
    records.sort(key=lambda quake: (quake.timestamp, quake.event_id))
    report["kept"] = len(records)
    return records, report


# ======================================================================
# PART 2 - COORDINATE COMPRESSION
# ======================================================================

def compress_magnitudes(magnitudes):
    """Map floating-point magnitudes to small integer ranks 0..sigma-1.

    Example: [4.5, 7.1, 4.5, 3.0]
        distinct sorted values  -> [3.0, 4.5, 7.1]
        ranks                   -> [1,   2,   1,   0]

    Equal magnitudes get the SAME rank (so duplicates are handled), and the
    ORDER of ranks equals the order of magnitudes (rank 0 = smallest).
    Returns (sorted_distinct_values, ranks).
    Cost: O(n log n) because of the sort.
    """
    sorted_distinct_values = sorted(set(magnitudes))
    rank_of_value = {value: rank for rank, value in enumerate(sorted_distinct_values)}
    ranks = [rank_of_value[value] for value in magnitudes]
    return sorted_distinct_values, ranks


# ======================================================================
# PART 3 - THE WAVELET TREE (written from scratch)
# ======================================================================

class WaveletNode:
    """One node of the tree.

    A node is responsible for the value range [low, high] (these are RANKS).
    Instead of storing its sequence of values, it stores only:

        left_prefix[i] = how many of the first i items of this node's
                         sequence went to the LEFT child.

    That single array is enough to answer every query (see the methods of
    WaveletTree). A leaf (low == high) needs no array because all of its
    items are equal to `low`.
    """
    __slots__ = ("low", "high", "left_prefix", "left_child", "right_child")

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.left_prefix = None
        self.left_child = None
        self.right_child = None


class WaveletTree:
    """Wavelet Tree over a sequence of integer ranks in [0, num_distinct).

    CONSTRUCTION - step by step
    ---------------------------
    Take the whole sequence and the full rank range [0, sigma-1].
      1. mid = (low + high) // 2.
      2. Walk through the sequence. A value <= mid belongs to the LEFT child,
         otherwise to the RIGHT child. Items keep their original relative
         order in the child sequences (this is what makes ranges map nicely).
      3. While walking, record left_prefix. Starting from [0], append the
         running number of "went left" items after every element.
      4. Recurse on the left sequence with [low, mid] and on the right
         sequence with [mid+1, high]. Stop when low == high (a leaf).
    The tree has about log2(sigma) levels and every level processes at most
    n items, so construction costs O(n log sigma) time and space.

    HOW A RANGE QUERY MOVES DOWN
    ----------------------------
    A range [left, right) at a node becomes
        LEFT  child: [ left_prefix[left],         left_prefix[right]         )
        RIGHT child: [ left - left_prefix[left],  right - left_prefix[right] )
    because left_prefix[left] items before `left` went left, so the other
    (left - left_prefix[left]) items went right. Each step is O(1), and the
    depth is log sigma, so each query costs O(log sigma).
    """

    def __init__(self, ranks, num_distinct):
        self.size = len(ranks)
        self.num_distinct = num_distinct
        self.root = None
        if ranks:
            self.root = self._build(ranks, 0, num_distinct - 1)

    def _build(self, sequence, low, high):
        node = WaveletNode(low, high)
        if low == high:                       # leaf: every item equals `low`
            return node

        mid = (low + high) // 2
        left_prefix = [0]                     # left_prefix[0] = 0 items so far
        left_sequence = []
        right_sequence = []
        for rank in sequence:
            if rank <= mid:
                left_sequence.append(rank)
                left_prefix.append(left_prefix[-1] + 1)
            else:
                right_sequence.append(rank)
                left_prefix.append(left_prefix[-1])
        node.left_prefix = left_prefix

        if left_sequence:
            node.left_child = self._build(left_sequence, low, mid)
        if right_sequence:
            node.right_child = self._build(right_sequence, mid + 1, high)
        return node

    # ------------------------------------------------------------------
    def kth_smallest_rank(self, left, right, k):
        """Rank of the k-th smallest item (k = 1 is the minimum) inside the
        half-open index range [left, right).  O(log sigma).

        Logic: at each node, count how many items of our range go left.
        If k <= that count the answer lives in the left child; otherwise the
        answer is the (k - count)-th smallest of the right child.
        """
        if not (0 <= left <= right <= self.size):
            raise ValueError("range is outside the array")
        if not (1 <= k <= right - left):
            raise ValueError("k must be between 1 and %d" % (right - left))

        node = self.root
        while node.low != node.high:
            prefix = node.left_prefix
            left_of_range = prefix[left]
            left_of_right = prefix[right]
            items_going_left = left_of_right - left_of_range
            if k <= items_going_left:
                left, right = left_of_range, left_of_right
                node = node.left_child
            else:
                k -= items_going_left
                left, right = left - left_of_range, right - left_of_right
                node = node.right_child
        return node.low

    def kth_largest_rank(self, left, right, k):
        """k-th largest = (length - k + 1)-th smallest."""
        if not (1 <= k <= right - left):
            raise ValueError("k must be between 1 and %d" % max(right - left, 0))
        return self.kth_smallest_rank(left, right, (right - left) - k + 1)

    def count_less_than_rank(self, left, right, rank_limit):
        """How many items in [left, right) have rank STRICTLY LESS than
        rank_limit?  O(log sigma).

        Logic: we walk down following rank_limit.
          * rank_limit <= mid : everything >= rank_limit is on the right or
            equal, so only the left child can contain smaller items -> go left.
          * rank_limit  > mid : ALL items that went left are smaller, so add
            that count to the answer and continue in the right child.
        """
        if not (0 <= left <= right <= self.size):
            raise ValueError("range is outside the array")
        if left == right or rank_limit <= 0:
            return 0
        if rank_limit >= self.num_distinct:
            return right - left

        total = 0
        node = self.root
        while node is not None and left < right:
            if rank_limit <= node.low:
                break                          # nothing here is smaller
            if rank_limit > node.high:
                total += right - left          # everything here is smaller
                break
            # now node.low < rank_limit <= node.high, so node is not a leaf
            mid = (node.low + node.high) // 2
            prefix = node.left_prefix
            left_of_range = prefix[left]
            left_of_right = prefix[right]
            if rank_limit <= mid:
                left, right = left_of_range, left_of_right
                node = node.left_child
            else:
                total += left_of_right - left_of_range
                left, right = left - left_of_range, right - left_of_right
                node = node.right_child
        return total


# ======================================================================
# PART 4 - THE ANALYSER: dates + magnitudes + Wavelet Tree together
# ======================================================================

class EarthquakeAnalyzer:
    """Glues everything together. Every magnitude query below is answered
    by the Wavelet Tree; bisect is only used to turn dates/thresholds into
    positions."""

    def __init__(self, records):
        if not records:
            raise ValueError("no valid earthquake records to analyse")
        self.records = records                                   # sorted by time
        self.timestamps = [quake.timestamp for quake in records]  # for bisect
        magnitudes = [quake.magnitude for quake in records]       # step 4
        self.distinct_magnitudes, ranks = compress_magnitudes(magnitudes)  # step 5
        self.tree = WaveletTree(ranks, len(self.distinct_magnitudes))      # step 6

    # ---- date handling ------------------------------------------------
    def date_range_to_indices(self, start_date, end_date):
        """Convert two dates (inclusive) into the index range [left, right).

        bisect_left(timestamps, start 00:00)       -> first quake on/after start
        bisect_left(timestamps, end + 1 day 00:00) -> first quake AFTER end date
        Cost: O(log n).
        """
        if start_date > end_date:
            raise ValueError("start date is after end date")
        left = bisect_left(self.timestamps, start_date)
        right = bisect_left(self.timestamps, end_date + timedelta(days=1))
        return left, right

    # ---- helpers ------------------------------------------------------
    def _value(self, rank):
        return self.distinct_magnitudes[rank]

    def _rank_at_or_above(self, magnitude):
        """Smallest rank whose magnitude is >= `magnitude` (O(log sigma))."""
        return bisect_left(self.distinct_magnitudes, magnitude)

    # ---- the six queries ---------------------------------------------
    def count_in_range(self, left, right):
        return right - left

    def kth_largest(self, left, right, k):
        """Feature 1."""
        return self._value(self.tree.kth_largest_rank(left, right, k))

    def kth_smallest(self, left, right, k):
        """Feature 2."""
        return self._value(self.tree.kth_smallest_rank(left, right, k))

    def count_at_least(self, left, right, threshold):
        """Feature 3: number of quakes with magnitude >= threshold."""
        limit_rank = self._rank_at_or_above(threshold)
        smaller = self.tree.count_less_than_rank(left, right, limit_rank)
        return (right - left) - smaller

    def median(self, left, right):
        """Feature 4. Odd count -> middle value; even count -> average of
        the two middle values. Returns None for an empty range."""
        count = right - left
        if count == 0:
            return None
        if count % 2 == 1:
            return self.kth_smallest(left, right, (count + 1) // 2)
        lower_middle = self.kth_smallest(left, right, count // 2)
        upper_middle = self.kth_smallest(left, right, count // 2 + 1)
        return (lower_middle + upper_middle) / 2

    def maximum(self, left, right):
        return self.kth_largest(left, right, 1) if right > left else None

    def count_between(self, left, right, low_inclusive, high_exclusive=None):
        """Number of quakes with low_inclusive <= magnitude < high_exclusive.
        high_exclusive=None means 'no upper limit'."""
        at_least_low = self.count_at_least(left, right, low_inclusive)
        if high_exclusive is None:
            return at_least_low
        return at_least_low - self.count_at_least(left, right, high_exclusive)

    def distribution(self, left, right):
        """Feature 5. Four NON-OVERLAPPING buckets using half-open
        boundaries, so every record falls in exactly one bucket."""
        buckets = [
            ("Below 3.0",  None, 3.0),
            ("3.0 - 4.9",  3.0, 5.0),
            ("5.0 - 6.9",  5.0, 7.0),
            ("7.0 and above", 7.0, None),
        ]
        result = []
        for label, low, high in buckets:
            if low is None:                                   # "below 3.0"
                count = (right - left) - self.count_at_least(left, right, high)
            else:
                count = self.count_between(left, right, low, high)
            result.append((label, count))
        return result

    def summarize_period(self, left, right, threshold):
        """Feature 6 helper: the four statistics of one period."""
        return {
            "count": right - left,
            "maximum": self.maximum(left, right),
            "median": self.median(left, right),
            "above_threshold": (self.count_at_least(left, right, threshold)
                                if right > left else 0),
        }


# ======================================================================
# PART 5 - COMMAND-LINE INTERFACE
# ======================================================================

def parse_user_date(text, is_end_of_range):
    """Accept 'YYYY-MM-DD' or just 'YYYY'.
    A bare year means Jan 1 (start) or Dec 31 (end) of that year."""
    text = text.strip()
    if len(text) == 4 and text.isdigit():
        year = int(text)
        if is_end_of_range:
            return datetime(year, 12, 31)
        return datetime(year, 1, 1)
    return datetime.strptime(text, "%Y-%m-%d")     # ValueError if invalid


def ask_date_range(analyzer, label=""):
    """Ask for start/end dates until valid. Returns (left, right, start, end)
    or None if the user presses Enter to cancel."""
    while True:
        start_text = input("Start date%s (YYYY-MM-DD or YYYY, Enter to cancel): " % label)
        if not start_text.strip():
            return None
        end_text = input("End date%s   (YYYY-MM-DD or YYYY): " % label)
        try:
            start = parse_user_date(start_text, False)
            end = parse_user_date(end_text, True)
            left, right = analyzer.date_range_to_indices(start, end)
            return left, right, start, end
        except ValueError as error:
            print("  Invalid input: %s. Please try again.\n" % error)


def ask_number(prompt, number_type):
    while True:
        text = input(prompt)
        try:
            return number_type(text)
        except ValueError:
            print("  Please enter a valid %s." % number_type.__name__)


def say_range(start, end, count):
    print("Period %s to %s -> %d earthquake(s)"
          % (start.date(), end.date(), count))


def menu_kth(analyzer, largest):
    picked = ask_date_range(analyzer)
    if picked is None:
        return
    left, right, start, end = picked
    say_range(start, end, right - left)
    if right == left:
        print("No earthquakes in this period - nothing to rank.")
        return
    k = ask_number("k (1 to %d): " % (right - left), int)
    if not 1 <= k <= right - left:
        print("Invalid k: it must be between 1 and %d." % (right - left))
        return
    began = time.perf_counter()
    if largest:
        answer = analyzer.kth_largest(left, right, k)
    else:
        answer = analyzer.kth_smallest(left, right, k)
    elapsed = (time.perf_counter() - began) * 1e6
    print("k = %d -> %s magnitude: %.2f   (query time %.1f microseconds)"
          % (k, "k-th largest" if largest else "k-th smallest", answer, elapsed))


def menu_frequency(analyzer):
    picked = ask_date_range(analyzer)
    if picked is None:
        return
    left, right, start, end = picked
    say_range(start, end, right - left)
    threshold = ask_number("Magnitude threshold: ", float)
    count = analyzer.count_at_least(left, right, threshold) if right > left else 0
    print("Earthquakes with magnitude >= %.1f: %d" % (threshold, count))


def menu_median(analyzer):
    picked = ask_date_range(analyzer)
    if picked is None:
        return
    left, right, start, end = picked
    say_range(start, end, right - left)
    median = analyzer.median(left, right)
    if median is None:
        print("No earthquakes in this period - median is undefined.")
    else:
        print("Median magnitude: %.3f" % median)


def menu_distribution(analyzer):
    picked = ask_date_range(analyzer)
    if picked is None:
        return
    left, right, start, end = picked
    say_range(start, end, right - left)
    for label, count in analyzer.distribution(left, right):
        print("  %-15s : %d" % (label, count))


def menu_compare(analyzer):
    print("--- Period A ---")
    first = ask_date_range(analyzer, " A")
    if first is None:
        return
    print("--- Period B ---")
    second = ask_date_range(analyzer, " B")
    if second is None:
        return
    threshold = ask_number("Magnitude threshold for 'above' count: ", float)

    def show(value):
        if value is None:
            return "n/a"
        return "%.2f" % value if isinstance(value, float) else str(value)

    stats_a = analyzer.summarize_period(first[0], first[1], threshold)
    stats_b = analyzer.summarize_period(second[0], second[1], threshold)
    print()
    print("%-28s %-22s %-22s" % (
        "",
        "A: %s..%s" % (first[2].date(), first[3].date()),
        "B: %s..%s" % (second[2].date(), second[3].date())))
    rows = [("Number of earthquakes", "count"),
            ("Maximum magnitude", "maximum"),
            ("Median magnitude", "median"),
            ("Count with mag >= %.1f" % threshold, "above_threshold")]
    for label, key in rows:
        print("%-28s %-22s %-22s" % (label, show(stats_a[key]), show(stats_b[key])))


def show_dataset_summary(analyzer, report):
    first, last = analyzer.records[0], analyzer.records[-1]
    print("Rows read: %d | kept: %d | bad magnitude: %d | bad time: %d | duplicate id: %d"
          % (report["rows_read"], report["kept"], report["bad_magnitude"],
             report["bad_time"], report["duplicate_id"]))
    print("Time span : %s  ->  %s" % (first.timestamp.date(), last.timestamp.date()))
    print("Magnitudes: min %.2f, max %.2f, %d distinct values (sigma)"
          % (analyzer.distinct_magnitudes[0], analyzer.distinct_magnitudes[-1],
             len(analyzer.distinct_magnitudes)))


def run_menu(analyzer, report):
    options = {
        "1": ("Kth-largest earthquake",         lambda: menu_kth(analyzer, True)),
        "2": ("Kth-smallest earthquake",        lambda: menu_kth(analyzer, False)),
        "3": ("Count with magnitude >= X",      lambda: menu_frequency(analyzer)),
        "4": ("Median magnitude",               lambda: menu_median(analyzer)),
        "5": ("Magnitude distribution",         lambda: menu_distribution(analyzer)),
        "6": ("Compare two time periods",       lambda: menu_compare(analyzer)),
        "7": ("Dataset summary",                lambda: show_dataset_summary(analyzer, report)),
    }
    while True:
        print("\n===== Earthquake Analysis (Wavelet Tree) =====")
        for key, (title, _) in options.items():
            print(" %s. %s" % (key, title))
        print(" 0. Exit")
        choice = input("Choose: ").strip()
        if choice == "0":
            print("Goodbye!")
            return
        if choice in options:
            print()
            options[choice][1]()
        else:
            print("Please choose a number from the menu.")


# ======================================================================
# PART 6 - NON-INTERACTIVE DEMO (also checks results against plain Python)
# ======================================================================

def run_demo(analyzer):
    """Runs sample queries and double-checks each one with a simple
    sort/scan. The brute-force lines are ONLY for verification."""
    def magnitudes_between(left, right):
        return [quake.magnitude for quake in analyzer.records[left:right]]

    def check(name, wavelet_answer, simple_answer):
        status = "OK" if wavelet_answer == simple_answer else "MISMATCH"
        print("   %-34s wavelet=%s  plain-python=%s  [%s]"
              % (name, wavelet_answer, simple_answer, status))

    def day(text):
        return datetime.strptime(text, "%Y-%m-%d")

    left, right = analyzer.date_range_to_indices(day("2020-01-01"), day("2020-12-31"))
    values = magnitudes_between(left, right)
    print("Period 2020-01-01 .. 2020-12-31  (%d quakes, index range [%d, %d))"
          % (len(values), left, right))
    check("3rd largest", analyzer.kth_largest(left, right, 3), sorted(values, reverse=True)[2])
    check("2nd smallest", analyzer.kth_smallest(left, right, 2), sorted(values)[1])
    check("count >= 5.0", analyzer.count_at_least(left, right, 5.0),
          sum(1 for value in values if value >= 5.0))
    ordered = sorted(values)
    middle = len(ordered) // 2
    simple_median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    check("median", analyzer.median(left, right), simple_median)
    print("   distribution:", analyzer.distribution(left, right))

    left, right = analyzer.date_range_to_indices(day("2020-01-01"), day("2022-12-31"))
    values = magnitudes_between(left, right)
    print("Period 2020 .. 2022 (%d quakes)" % len(values))
    check("count >= 5.0", analyzer.count_at_least(left, right, 5.0),
          sum(1 for value in values if value >= 5.0))

    a_left, a_right = analyzer.date_range_to_indices(day("2019-01-01"), day("2020-12-31"))
    b_left, b_right = analyzer.date_range_to_indices(day("2021-01-01"), day("2022-12-31"))
    print("Compare 2019-2020 (A) vs 2021-2022 (B), threshold 7.0")
    print("   A:", analyzer.summarize_period(a_left, a_right, 7.0))
    print("   B:", analyzer.summarize_period(b_left, b_right, 7.0))


def main():
    arguments = sys.argv[1:]
    demo_mode = "--demo" in arguments
    paths = [argument for argument in arguments if argument != "--demo"]
    csv_path = paths[0] if paths else "sample_earthquakes.csv"

    try:
        began = time.perf_counter()
        records, report = load_earthquakes(csv_path)
        analyzer = EarthquakeAnalyzer(records)
        built = time.perf_counter() - began
    except FileNotFoundError:
        print("Cannot find the file '%s'. See README.md for how to get the data." % csv_path)
        return
    except ValueError as error:
        print("Problem with the dataset: %s" % error)
        return

    print("Loaded '%s' and built the Wavelet Tree in %.3f s" % (csv_path, built))
    show_dataset_summary(analyzer, report)
    if demo_mode:
        print()
        run_demo(analyzer)
    else:
        try:
            run_menu(analyzer, report)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")


if __name__ == "__main__":
    main()
