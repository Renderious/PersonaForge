import re

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits the input text into semantic chunks of roughly `chunk_size` characters,
    with an overlap of `overlap` characters between consecutive chunks.

    It tries to split by paragraph first, then by sentence, to maintain semantic meaning.

    Args:
        text (str): The plain text to chunk.
        chunk_size (int): The target maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size.")

    chunks = []

    # Pre-process: split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # If adding the paragraph exceeds chunk_size and we already have content
        if len(current_chunk) + len(p) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap from the end of the previous chunk
            # We approximate overlap by taking the last 'overlap' characters from the previous chunk,
            # ensuring we break at a word boundary.
            if len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                # Try to find a space to break on
                space_idx = overlap_text.find(' ')
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx+1:]
                current_chunk = overlap_text + "\n\n" + p
            else:
                current_chunk = p
        else:
            if current_chunk:
                current_chunk += "\n\n" + p
            else:
                current_chunk = p

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
