def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits the input text into chunks of roughly `chunk_size` characters,
    with an overlap of `overlap` characters between consecutive chunks.

    This helps preserve context across chunk boundaries.

    Args:
        text (str): The plain text to chunk.
        chunk_size (int): The target maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []

    # Ensure overlap is strictly less than chunk_size to avoid infinite loops
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # If we are not at the very end of the text, try to find a nice breaking point
        # (like a newline or space) so we don't cut words in half.
        if end < text_length:
            # Try to break at a double newline (paragraph)
            break_point = text.rfind('\n\n', start, end)

            # Fallback to single newline if no double newline found in the latter half of the chunk
            if break_point == -1 or break_point < start + (chunk_size // 2):
                break_point = text.rfind('\n', start, end)

            # Fallback to space if no newline found
            if break_point == -1 or break_point < start + (chunk_size // 2):
                break_point = text.rfind(' ', start, end)

            # If we found a suitable breaking point, use it. Otherwise, hard break at chunk_size.
            if break_point != -1 and break_point > start:
                end = break_point + 1 # Include the breaking character

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks
