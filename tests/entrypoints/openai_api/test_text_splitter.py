"""Unit tests for SentenceSplitter."""

import pytest

from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter


class TestSentenceSplitter:
    def test_basic_english_splitting(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hello world. How are you? ")
        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"

    def test_no_complete_sentence(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hello world")
        assert sentences == []

    def test_incremental_accumulation(self):
        splitter = SentenceSplitter()
        # Feed partial text
        sentences = splitter.feed("Hello ")
        assert sentences == []
        sentences = splitter.feed("world. ")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    def test_flush(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world")
        remaining = splitter.flush()
        assert remaining == "Hello world"

    def test_flush_empty_buffer(self):
        splitter = SentenceSplitter()
        remaining = splitter.flush()
        assert remaining is None

    def test_cjk_splitting(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("你好世界。今天天气很好！")
        assert len(sentences) >= 1
        # At least the first sentence should be split
        assert "你好世界" in sentences[0]

    def test_mixed_english_cjk(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hello世界。How are you? ")
        assert len(sentences) >= 1

    def test_exclamation_mark(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Wow! That is great. ")
        assert len(sentences) == 2
        assert sentences[0] == "Wow!"
        assert sentences[1] == "That is great."

    def test_question_mark(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("What? Why? ")
        assert len(sentences) == 2

    def test_min_sentence_length(self):
        splitter = SentenceSplitter(min_sentence_length=5)
        sentences = splitter.feed("Hi. Hello world. ")
        # "Hi" should be filtered out (length < 5)
        filtered = [s for s in sentences if len(s) >= 5]
        assert any("Hello world" in s for s in filtered)

    def test_reset(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world")
        splitter.reset()
        remaining = splitter.flush()
        assert remaining is None

    def test_multiple_feed_and_flush(self):
        splitter = SentenceSplitter()
        s1 = splitter.feed("First sentence. ")
        s2 = splitter.feed("Second sentence. Third")
        remaining = splitter.flush()

        assert len(s1) == 1
        assert s1[0] == "First sentence."
        assert len(s2) == 1
        assert s2[0] == "Second sentence."
        assert remaining == "Third"

    def test_empty_feed(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("")
        assert sentences == []

    def test_whitespace_only_feed(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("   ")
        assert sentences == []
