import unittest


class MyTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 2, 3)

    def test_sub(self):
        self.assertEqual(3 - 2, 1)
