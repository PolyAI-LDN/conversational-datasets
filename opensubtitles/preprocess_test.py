"""Tests for subtitle line ending normalization."""

import unittest

from opensubtitles import create_data


class PreprocessLineTest(unittest.TestCase):
    def test_crlf_matches_lf(self):
        self.assertEqual(create_data._preprocess_line(b"hello\r\n"),
                         create_data._preprocess_line(b"hello\n"))

    def test_blank_crlf_line_is_empty(self):
        self.assertEqual(create_data._preprocess_line(b"\r\n"), "")

    def test_crlf_is_removed_after_speaker_and_event_processing(self):
        self.assertEqual(create_data._preprocess_line(
            b"MATT: hello (laughs)\r\n"), "hello")


if __name__ == "__main__":
    unittest.main()
