"""Split long document text into overlapping chunks of about a fixed number of tokens."""

import re

import tiktoken

# The tokenizer of OpenAI's text-embedding-3 models.
_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s)")


def split_text_into_chunks(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack whole lines into chunks of up to about max_tokens tokens.

    Each chunk after the first starts with up to overlap_tokens tokens of whole
    lines from the end of the previous chunk. Text within max_tokens is one chunk.
    """
    lines = _split_into_lines_within_limit(text, max_tokens)
    if sum(tokens for _, tokens in lines) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while True:
        end, size = start, 0
        while end < len(lines) and size + lines[end][1] <= max_tokens:
            size += lines[end][1]
            end += 1
        chunks.append("".join(line for line, _ in lines[start:end]))
        if end == len(lines):
            return chunks
        # Never step back to this chunk's first line, so every chunk moves forward.
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap + lines[next_start - 1][1] <= overlap_tokens:
            next_start -= 1
            overlap += lines[next_start][1]
        start = next_start


def _split_into_lines_within_limit(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Split text into lines, each with its token count, none over max_tokens.

    A longer line is split into sentences, and a sentence still too long into
    slices of max_tokens tokens. Joined, the returned lines are the text.
    """
    lines = []
    for line in text.splitlines(keepends=True):
        tokens = _count_tokens(line)
        if tokens <= max_tokens:
            lines.append((line, tokens))
            continue
        for sentence in _SENTENCE_END.split(line):
            encoded = _ENCODING.encode_ordinary(sentence)
            # Cut at token start offsets, so a character spanning two tokens is never split.
            _, offsets = _ENCODING.decode_with_offsets(encoded)
            starts = list(range(0, len(encoded), max_tokens))
            ends = [offsets[s] for s in starts[1:]] + [len(sentence)]
            for start, char_start, char_end in zip(starts, [offsets[s] for s in starts], ends):
                lines.append((sentence[char_start:char_end], min(max_tokens, len(encoded) - start)))
    return lines


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode_ordinary(text))
