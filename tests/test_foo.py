# -*- coding: utf-8 -*-
""" Test for usage of Travis CI.
"""
from __future__ import division, print_function, absolute_import

import unittest

class TestUnitTest(unittest.TestCase):

    def test_foo(self):
        self.assertTrue(True)

    def test_foo2(self):
        self.assertFalse(False)


if __name__ == '__main__':
    unittest.main()
