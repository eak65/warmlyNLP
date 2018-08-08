from __future__ import division
from newspaper import Article
import nltk, re, pprint
import unittest
from app import generateSnippet
from newspaper import fulltext


class Test_test1(unittest.TestCase):
    def test_A(self):
        text = "Today was a good day. \"We want our student to be successful\" said Ranall Dieke."
        generatedSnippet = generateSnippet("Randell Dieke", text)

if __name__ == '__main__':
    unittest.main()
