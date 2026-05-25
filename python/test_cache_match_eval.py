#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import cache_match_eval


class CacheMatchEvalArgsTest(unittest.TestCase):
    def test_descriptor_topk_default_matches_cpp_inference_default(self) -> None:
        argv = [
            "cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--checkpoint",
            "model.pt",
        ]

        with patch.object(sys, "argv", argv):
            args = cache_match_eval.parse_args()

        self.assertEqual(args.descriptor_topk, 4)

    def test_descriptor_topk_override_is_preserved(self) -> None:
        argv = [
            "cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--checkpoint",
            "model.pt",
            "--descriptor-topk",
            "64",
        ]

        with patch.object(sys, "argv", argv):
            args = cache_match_eval.parse_args()

        self.assertEqual(args.descriptor_topk, 64)


if __name__ == "__main__":
    unittest.main()
