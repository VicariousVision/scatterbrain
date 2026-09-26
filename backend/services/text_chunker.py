"""Text chunking service with simple recursive splitting.

Provides semantic chunking with configurable size and overlap.
"""

from __future__ import annotations

import re
from typing import List

from config import settings


def chunk_text(text: str) -> List[str]:
    """Split text into semantically meaningful chunks.
    
    Uses recursive character-based splitting that tries to split on natural
    boundaries (double newlines, single newlines, sentences, words) before
    falling back to character-based splitting.
    
    Args:
        text: Input text to chunk.
        
    Returns:
        List of text chunks with configured size and overlap.
    """
    if not text or not text.strip():
        return []
    
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    
    # Separators in order of preference
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    return _recursive_split(text, separators, chunk_size, chunk_overlap)


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursively split text on separators."""
    final_chunks = []
    
    # Get the current separator
    separator = separators[0] if separators else ""
    
    # Split on separator
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)
    
    # Merge splits into chunks
    current_chunk = []
    current_length = 0
    
    for split in splits:
        split_len = len(split)
        
        # If single split is too long and we have more separators, recurse
        if split_len > chunk_size and len(separators) > 1:
            # Save current chunk if it exists
            if current_chunk:
                final_chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Recurse on the long split
            sub_chunks = _recursive_split(
                split,
                separators[1:],
                chunk_size,
                chunk_overlap,
            )
            final_chunks.extend(sub_chunks)
        else:
            # Add to current chunk if it fits
            if current_length + split_len + len(separator) <= chunk_size:
                current_chunk.append(split)
                current_length += split_len + len(separator)
            else:
                # Save current chunk and start new one
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                
                # Handle overlap
                if chunk_overlap > 0 and current_chunk:
                    # Keep last few items for overlap
                    overlap_text = separator.join(current_chunk)
                    if len(overlap_text) > chunk_overlap:
                        overlap_start = len(overlap_text) - chunk_overlap
                        overlap_text = overlap_text[overlap_start:]
                        current_chunk = [overlap_text, split]
                        current_length = len(overlap_text) + len(separator) + split_len
                    else:
                        current_chunk = [split]
                        current_length = split_len
                else:
                    current_chunk = [split]
                    current_length = split_len
    
    # Add final chunk
    if current_chunk:
        final_chunks.append(separator.join(current_chunk))
    
    return [chunk for chunk in final_chunks if chunk.strip()]
