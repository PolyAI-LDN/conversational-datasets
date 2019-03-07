"""Tests for create_data.py."""

import json
import unittest

from reddit import create_data


class CreateDataTest(unittest.TestCase):

    def test_trim(self):
        self.assertEqual(
            "Matthew",
            create_data.trim("Matthew Henderson", 7)
        )

    def test_trim_do_not_split_word(self):
        self.assertEqual(
            "Matthew ",
            create_data.trim("Matthew Henderson", 9)
        )

    def test_trim_string_short(self):
        self.assertEqual(
            "Matthew",
            create_data.trim("Matthew", 9)
        )

    def test_normalise_comment(self):
        comment = create_data.normalise_comment(
            {
                'body': "ABC EFG HIJ KLM NOP",
                'score_hidden': None,
                'archived': None,
                'name': None,
                'author_flair_text': None,
                'downs': None,
                'created_utc': "1520704245",
                'subreddit_id': "t5_AAAAA",
                'link_id': "t3_BBBBB",
                'parent_id': "t1_CCCCC",
                'score': "1",
                'retrieved_on': "1525020075",
                'controversiality': "0",
                'gilded': "0",
                'id': "DDDDD",
                'subreddit': "EEEEE",
                'author': "FFFFF",
                'ups': None,
                'distinguished': None,
                'author_flair_css_class': None,
            },
            max_length=16)
        self.assertEqual(
            comment,
            create_data.Comment(
                body="ABC EFG HIJ KLM ",
                thread_id="BBBBB",
                parent_id="CCCCC",
                id="DDDDD",
                body_is_trimmed=True,
                subreddit="EEEEE",
                author="FFFFF",
            )
        )

    def test_linear_paths(self):
        with open("reddit/testdata/thread.json") as f:
            comments = json.loads(f.read())
        comments = [
            create_data.normalise_comment(comment, max_length=127)
            for comment in comments]
        id_to_comment = {comment.id: comment for comment in comments}
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertItemsEqual(
            [["dvedzte", "dvfdfd4"], ["dvedzte", "dveh7r5"],
             ["dve3v95", "dvhjrkc"], ["dve3v95", "dvhjrkc", "dvhktmd"],
             ["dve3v95", "dvhjrkc", "dvhktmd", "dvhn7hh"],
             ["dve3v95", "dvhjrkc", "dvhktmd", "dvhn7hh", "dvhvg4m"]],
            paths
        )

    @staticmethod
    def _create_test_comment(id, parent_id):
        return create_data.Comment(
            body="body",
            thread_id="thread_id",
            parent_id=parent_id,
            id=id,
            body_is_trimmed=True,
            subreddit="subreddit",
            author="author",
        )

    def test_linear_paths_with_self_loop(self):
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="1"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertEqual([], paths)

    def test_linear_paths_with_loop(self):
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="3"),
            "2": self._create_test_comment(id="2", parent_id="1"),
            "3": self._create_test_comment(id="3", parent_id="2"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertEqual([], paths)

    def test_linear_paths_with_stranded_threads(self):
        """Check that it picks up threads whose parents are missing."""
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="unseen"),
            "2": self._create_test_comment(id="2", parent_id="1"),


            "3": self._create_test_comment(id="3", parent_id="unseen 2"),
            "4": self._create_test_comment(id="4", parent_id="3"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertItemsEqual([
            ["1", "2"],
            ["3", "4"],
        ], paths)

    def test_long_thread(self):
        """Check there is no issue with long threads (e.g. recursion limits)"""
        id_to_comment = {
            i: self._create_test_comment(id=i, parent_id=i - 1)
            for i in range(2000)
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=10))
        self.assertItemsEqual([
            range(max(i - 11, 0), i)
            for i in range(2, 2001)
        ], paths)

    def test_create_examples(self):
        pass


if __name__ == "__main__":
    unittest.main()
