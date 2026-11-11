## 10. Medium: Markdown heading detection is not aware of fenced code blocks

The heading regex detects any line beginning with `#`:

```python
HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+.*$")
```

It may treat this as a Markdown heading even when it appears inside a fenced code block:

````markdown
```python
# This is a Python comment
```
````

That can split code examples incorrectly.

### Required fix

Ignore heading-like lines inside fenced code blocks, or use a Markdown parser if the assignment requires accurate Markdown structure.

---

## 11. Medium: Markdown chunking splits without semantic overlap

The Markdown chunker creates independent chunks with no overlap.

This can separate related context at boundaries. For example, a paragraph may be separated from the heading or explanation that gives it meaning.

### Required decision

Decide whether the subject requires:

- no overlap,
- fixed token/character overlap, or
- heading/context propagation into every section.

For RAG, preserving heading context is often more useful than plain character splitting.

---

## 12. Medium: Character offsets and line offsets are inconsistent

Markdown stores:

```python
first_character_index
last_character_index
```

Python stores:

```python
start_line
end_line
```

This may be acceptable, but it makes downstream citation and retrieval handling inconsistent.

### Required decision

Standardize the location model. Possible options:

- always store line ranges,
- always store character ranges,
- store both where available.

Also clarify whether end positions are inclusive or exclusive. Markdown currently uses an inclusive `last_character_index`, while Python line ranges are effectively inclusive.

---

## 15. Medium: Error handling is not suitable for a pipeline

Python syntax errors are handled with:

```python
print(f"Skipping file {self.file_path} due to SyntaxError: {e}")
return []
```

This silently skips files from the corpus. The caller still receives a successful chunk count.

Unreadable files and decoding errors are not handled explicitly and will abort the entire operation.

### Required decision

Decide whether the desired behavior is:

- fail fast,
- skip invalid files but report them,
- skip invalid files and return an error summary.

For a corpus-building pipeline, skipped files should at least be counted and reported.

---

## 16. Testing is incomplete

The only current tests are Markdown tests in `test_markdown.py`.

There are no tests covering Python chunking.

At minimum, tests are needed for:

- regular functions
- async functions
- methods
- empty classes
- classes with methods
- nested functions
- decorators
- multiline functions
- source newline preservation
- syntax errors
- invalid chunk sizes
- long functions exceeding `max_chunk_size`
- module-level code
- relative output paths
- JSON schema consistency
- Markdown headings inside fenced code blocks

The current Markdown suite also has one failing test because of the absolute/relative path issue.

---

# What is currently working

The following parts appear to work at a basic level:

- Markdown files are discovered recursively.
- Python files are discovered recursively.
- Markdown sections are split at headings.
- Markdown chunks try to split at paragraph, newline, or space boundaries.
- Empty Markdown files return no chunks.
- Python syntax errors do not crash the entire Python chunking loop.
- The Pydantic circular-reference problem has been resolved through `models.py`.
- `uv run python maix.py` currently runs and produces chunks.

---

# Recommended order before moving to indexing

## Must fix first

1. Preserve Python source newlines.
2. Implement Python `max_chunk_size`.
3. Standardize Python and Markdown output into one chunk schema.
4. Define and fix path handling.
5. Add Python chunking tests.
6. Decide how module-level code and class context are represented.

## Fix next

7. Handle decorators.
8. Handle nested definitions.
9. Add Markdown chunk IDs.
10. Improve syntax/read-error reporting.
11. Decide whether Markdown fenced code blocks need special handling.
12. Remove or implement the empty `Chunking` class.
13. Rename the misleading `with_rust` functions if Rust is not actually used.

## Important architectural decision

Before starting indexing, finalize this contract:

```text
Every chunk must have:
- stable id
- content
- file path
- chunk type
- location information
- metadata
```

Without a stable and consistent chunk schema, BM25/TF-IDF indexing and later citation will require special handling for Python and Markdown.