

## 4. High: Nested functions and nested classes are omitted

Only direct children of the module and direct children of a class are inspected.

This means the following are not fully handled:

```python
def outer():
    def inner():
        pass
```

and:

```python
class Outer:
    class Inner:
        def method(self):
            pass
```

The inner function/class is not independently chunked.

### Required fix

Use recursive AST traversal if nested definitions are expected to be searchable. The traversal should preserve the full parent context.

---

## 5. High: Class chunks are inconsistent

For a class containing methods, the class itself is not emitted as a chunk. Only its methods are emitted.

For an empty class, the class is emitted as a chunk.

Therefore:

```python
class User:
    def save(self):
        pass
```

produces only a method chunk, while:

```python
class User:
    pass
```

produces a class chunk.

This means class-level information such as:

- base classes
- class docstring
- class attributes
- decorators
- class-level configuration

is lost whenever the class has methods.

### Required decision

Choose one consistent strategy:

- emit a class chunk plus method chunks, or
- emit one complete class chunk, or
- explicitly define that only methods are indexed.

For code search and RAG, emitting both class context and method chunks is usually more useful.

---

## 6. High: Function decorators are likely excluded

AST positions for a function generally point to the `def` line, while decorators are stored separately in `decorator_list`.

For example:

```python
@app.route("/")
def index():
    return "ok"
```

The chunk may begin at:

```python
def index():
```

and omit:

```python
@app.route("/")
```

This loses important semantic information, particularly for frameworks such as Flask, FastAPI, and pytest.

### Required fix

Include decorators in the extracted function chunk and signature when present.

---

## 7. High: The two chunkers use different output schemas

Python emits the shared `Chunk` model:

```json
{
  "id": "...",
  "content": "...",
  "file_path": "...",
  "chunk_type": "code",
  "start_line": 1,
  "end_line": 5,
  "Metadata": {...}
}
```

Markdown emits `MarkdownChunk` directly:

```json
{
  "text": "...",
  "file_path": "...",
  "first_character_index": 0,
  "last_character_index": 20
}
```

This contradicts the purpose of `Chunk` in `chuncking.py`, which describes a standardized representation for both Python and Markdown.

### Required fix

Make both chunkers produce the same outer schema.

For example:

```json
{
  "id": "...",
  "content": "...",
  "file_path": "...",
  "chunk_type": "markdown",
  "start_line": 1,
  "end_line": 5,
  "metadata": {...}
}
```

The exact schema should be decided before implementing indexing and retrieval.

---

## 8. High: Markdown chunks do not have IDs

Python chunks receive UUIDs, but Markdown chunks do not.

A retrieval system generally needs a stable identifier for every chunk. Without an ID, later indexing, deduplication, citation, or result tracking becomes more difficult.

### Required fix

Add an ID to Markdown chunks, preferably through the shared `Chunk` model.

Also decide whether IDs should be:

- random UUIDs, or
- deterministic IDs based on file path and offset.

Deterministic IDs are usually better for repeatable indexing.

---

## 9. Medium: Markdown file paths can become absolute unexpectedly

In `markdown.py`, the stored path is created with:

```python
str(corpus_root / cur_dir / filename)
```

If `corpus_root` is an absolute path, the JSON output contains an absolute path.

The existing test expects:

```text
corpus/nested/guide.md
```

but receives an absolute path such as:

```text
/tmp/.../corpus/nested/guide.md
```

This causes the current Markdown test failure.

The Python writer has the same potential behavior.

### Required fix

Choose one path policy and use it consistently:

- path relative to project root, or
- path relative to corpus root, or
- absolute path.

For portability, indexing, and reproducible output, project-relative paths are usually preferable.

---

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

## 13. Medium: `Chunking` is empty and unused

`Chunking` is declared but contains only:

```python
pass
```

This is not necessarily a bug, but it indicates that the chunking abstraction is unfinished or unnecessary.

### Required decision

Either:

- implement the intended orchestration API, or
- remove the unused class and keep the module as model definitions only.

---

## 14. Medium: The name says Rust, but the implementation is Python

Both functions are named:

```python
chunk_python_corpus_with_rust
chunk_markdown_corpus_with_rust
```

but the implementation uses Python `ast`, regexes, and `scandir_rs`. No Rust chunking implementation is actually present in these functions.

This is misleading for maintainers and documentation.

### Required fix

Either implement the Rust-backed logic or rename the functions to accurately describe their behavior.

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