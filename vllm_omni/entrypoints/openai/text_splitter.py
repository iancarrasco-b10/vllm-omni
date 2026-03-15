"""Multi-language sentence boundary detector for streaming TTS input.

Buffers incoming text and splits at sentence boundaries (English and CJK),
yielding complete sentences for audio generation.
"""

import re
from re import Pattern

# Maximum buffer size (in characters) to prevent unbounded memory growth.
_MAX_BUFFER_SIZE = 100_000  # ~100 KB of text

# Sentence-level: .!? + CJK sentence-ending 。！？
# NOTE: English requires trailing whitespace to confirm a boundary —
# end-of-string is NOT treated as a boundary (that is what flush() is for).
SPLIT_SENTENCE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？])"
)

# Clause-level: adds CJK commas ， and semicolons ；
SPLIT_CLAUSE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？，；])"
)

# Default alias
_SENTENCE_BOUNDARY_RE = SPLIT_SENTENCE


class SentenceSplitter:
    """Incremental sentence splitter for streaming text input.

    Buffers text and yields complete sentences when boundaries are detected.
    Designed for TTS pipelines where text arrives incrementally (e.g., from STT).

    Args:
        min_sentence_length: Minimum character length for a sentence.
            Sentences shorter than this are kept in the buffer to avoid
            splitting on abbreviations like "Dr." or "U.S.".
        boundary_re: Custom compiled regex for sentence boundaries.
            Use ``SPLIT_SENTENCE`` (default) for sentence-level splitting,
            ``SPLIT_CLAUSE`` for finer-grained clause-level splitting,
            or pass your own ``re.Pattern``.
        max_buffered_words: When set, flush the buffer once it contains at
            least this many words even if no sentence boundary has been found.
            Useful for long-running input without punctuation.  Set to 0 or
            ``None`` to disable (default).
    """

    def __init__(
        self,
        min_sentence_length: int = 2,
        boundary_re: Pattern[str] | None = None,
        max_buffered_words: int | None = None,
    ) -> None:
        self._buffer: str = ""
        self._min_sentence_length = min_sentence_length
        self._boundary_re = boundary_re or _SENTENCE_BOUNDARY_RE
        self._max_buffered_words = max_buffered_words or 0

    @property
    def buffer(self) -> str:
        """Current buffered text."""
        return self._buffer

    def add_text(self, text: str) -> list[str]:
        """Add text to the buffer and return any complete sentences.

        Args:
            text: Incoming text chunk.

        Returns:
            List of complete sentences extracted from the buffer.
            May be empty if no sentence boundary was found.

        Raises:
            ValueError: If the buffer exceeds the maximum size.
        """
        if not text:
            return []

        self._buffer += text
        if len(self._buffer) > _MAX_BUFFER_SIZE:
            raise ValueError(
                f"Text buffer exceeded maximum size ({_MAX_BUFFER_SIZE} chars). "
                "Consider adding sentence-ending punctuation to your input."
            )
        return self._extract_sentences()

    def flush(self) -> str | None:
        """Flush remaining buffered text as a final sentence.

        Returns:
            The remaining buffered text (stripped), or None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _extract_sentences(self) -> list[str]:
        """Split buffer at sentence boundaries, keeping incomplete text buffered.

        If ``max_buffered_words`` is set and no sentence boundary is found,
        the buffer is flushed at the last word boundary once the word count
        threshold is reached.
        """
        parts = self._boundary_re.split(self._buffer)

        if len(parts) <= 1:
            # No boundary found — try word-count flush
            return self._maybe_flush_by_word_count()

        sentences: list[str] = []
        carry = ""
        # All parts except the last are complete sentences
        for i in range(len(parts) - 1):
            text = carry + parts[i]
            carry = ""
            stripped = text.strip()
            if len(stripped) >= self._min_sentence_length:
                sentences.append(stripped)
            elif stripped:
                # Too short (e.g. "Dr.") — carry forward to next part
                carry = text
            # else: empty, skip

        # Last part stays in buffer (may be incomplete)
        self._buffer = carry + parts[-1]

        # After extracting boundary-based sentences, the remainder may
        # still exceed the word threshold.
        sentences.extend(self._maybe_flush_by_word_count())

        return sentences

    def _maybe_flush_by_word_count(self) -> list[str]:
        """Flush buffer at word boundary if max_buffered_words is exceeded."""
        if self._max_buffered_words <= 0:
            return []

        words = self._buffer.split()
        if len(words) < self._max_buffered_words:
            return []

        # Flush exactly max_buffered_words, keep the rest
        flush_words = words[: self._max_buffered_words]
        remaining_words = words[self._max_buffered_words :]
        self._buffer = " ".join(remaining_words) if remaining_words else ""

        sentence = " ".join(flush_words).strip()
        if not sentence:
            return []

        # Recurse in case remaining still exceeds threshold
        result = [sentence]
        result.extend(self._maybe_flush_by_word_count())
        return result
