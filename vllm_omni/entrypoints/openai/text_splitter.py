"""Sentence splitter for streaming TTS text input.

Provides incremental sentence boundary detection for English and CJK text,
enabling real-time text-to-speech generation as text arrives.
"""

import re


class SentenceSplitter:
    """Regex-based sentence boundary detector for streaming text input.

    Supports:
    - English sentence boundaries: . ! ? followed by whitespace
    - CJK sentence boundaries: \u3002 \uff01 \uff1f \uff0c \uff1b

    Usage:
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hello world. ")
        # sentences == ["Hello world."]
        sentences = splitter.feed("How are you? I am fine.")
        # sentences == ["How are you?"]
        remaining = splitter.flush()
        # remaining == "I am fine."
    """

    # English sentence endings: .!? followed by whitespace or end
    # CJK sentence endings: \u3002\uff01\uff1f\uff0c\uff1b
    _SPLIT_PATTERN = re.compile(
        r'(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f\uff0c\uff1b])'
    )

    def __init__(self, min_sentence_length: int = 2):
        self._buffer = ""
        self._min_length = min_sentence_length

    def feed(self, text: str) -> list[str]:
        """Feed text and return any complete sentences.

        Args:
            text: Incoming text chunk

        Returns:
            List of complete sentences (may be empty)
        """
        self._buffer += text
        return self._extract_sentences()

    def flush(self) -> str | None:
        """Flush remaining buffer content.

        Returns:
            Remaining text or None if buffer is empty
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def reset(self):
        """Clear the buffer."""
        self._buffer = ""

    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from buffer."""
        parts = self._SPLIT_PATTERN.split(self._buffer)
        if len(parts) <= 1:
            return []

        sentences = []
        # All parts except the last are complete sentences
        for part in parts[:-1]:
            part = part.strip()
            if len(part) >= self._min_length:
                sentences.append(part)

        # Keep the last part in the buffer (incomplete sentence)
        self._buffer = parts[-1]
        return sentences
