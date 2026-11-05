import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class MarkdownChunk:
    title: str
    level: int
    content: str
    file_path: str
    breadcrumbs: List[str]
    start_line: int
    end_line: int


class MarkdownHeaderChunker:

    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def chunk_file(self) -> List[MarkdownChunk]:
        """Splits a markdown file into chunks grouped by heading context."""
        chunks = []
        current_headers: Dict[int, str] = {}
        current_content: List[str] = []

        current_title = "Introduction"
        current_level = 0
        start_line = 1

        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for idx, line in enumerate(self.lines):
            line_num = idx + 1
            match = header_pattern.match(line)

            if match:
                # Save previous chunk if it has content
                stripped_content = "".join(current_content).strip()
                if stripped_content:
                    # Fix: Collect breadcrumbs relevant to the *current_level* of the chunk being saved
                    breadcrumbs = [
                        current_headers[lvl]
                        for lvl in sorted(current_headers.keys())
                        if lvl < current_level
                    ]
                    chunks.append(
                        MarkdownChunk(
                            title=current_title,
                            level=current_level,
                            content=stripped_content,
                            file_path=self.file_path,
                            breadcrumbs=breadcrumbs,
                            start_line=start_line,
                            end_line=line_num - 1,
                        )
                    )

                # Parse new header properties
                hashes, title = match.groups()
                new_level = len(hashes)
                new_title = title.strip()

                # Fix: Clear out deeper or equal level paths BEFORE building new context
                current_headers = {
                    k: v for k, v in current_headers.items() if k < new_level
                }
                current_headers[new_level] = new_title

                # Switch active tracker targets
                current_level = new_level
                current_title = new_title
                start_line = line_num
                current_content = [line]
            else:
                current_content.append(line)

        # Flush out remaining text from trailing blocks
        stripped_content = "".join(current_content).strip()
        if stripped_content:
            breadcrumbs = [
                current_headers[lvl]
                for lvl in sorted(current_headers.keys())
                if lvl < current_level
            ]
            chunks.append(
                MarkdownChunk(
                    title=current_title,
                    level=current_level,
                    content=stripped_content,
                    file_path=self.file_path,
                    breadcrumbs=breadcrumbs,
                    start_line=start_line,
                    end_line=len(self.lines),
                )
            )

        return chunks


# ==========================================
# Example Usage Validation
# ==========================================
if __name__ == "__main__":
    dummy_markdown = """
# API Documentation
Welcome to the internal core infrastructure documentation manual.

## Authentication
To query endpoints safely, you must generate access identifiers.

### JWT Strategy
We rely heavily on structured cryptographically signed tokens.
* Format: Bearer tokens
* Expiry: 15 Minutes

### OAuth Strategy
For 3rd party vendor access setups.

## Rate Limits
Max requests are restricted to 60 calls per minute standard.
"""

    target_filename = "api_docs.md"
    with open(target_filename, "w", encoding="utf-8") as f:
        f.write(dummy_markdown.strip())

    chunker = MarkdownHeaderChunker(file_path=target_filename)
    processed_chunks = chunker.chunk_file()

    print(json.dumps([asdict(c) for c in processed_chunks], indent=2))
