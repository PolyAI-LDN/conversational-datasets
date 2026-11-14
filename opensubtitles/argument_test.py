"""Tests for command line context limits."""

import unittest

from opensubtitles import create_data


class ContextCountTest(unittest.TestCase):
    def _parse(self, value):
        return create_data._parse_args([
            "--sentence_files=input", "--output_dir=output",
            "--num_extra_contexts=" + value])[0]

    def test_explicit_count_is_integer(self):
        self.assertEqual(self._parse("2").num_extra_contexts, 2)

    def test_zero_extra_contexts_is_allowed(self):
        self.assertEqual(self._parse("0").num_extra_contexts, 0)

    def test_negative_count_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("-1")

    def test_noninteger_count_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("1.5")


if __name__ == "__main__":
    unittest.main()
