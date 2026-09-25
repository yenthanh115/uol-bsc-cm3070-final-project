#!/usr/bin/env python3
"""Per-section word counter for Pandoc Markdown reports.

Counts *prose only* by walking Pandoc's JSON AST, so markup is classified
precisely rather than guessed at with regexes.

Excluded from every count (per report conventions / standard academic rules):
  - YAML metadata and raw blocks (e.g. the ```{=typst} styling block)
  - Headings (their text is shown as the section label, never counted)
  - Tables, including the whole table and its caption
  - Figures / images and their captions
  - Code blocks and inline code
  - Math
  - Everything from the References heading onward (references, appendices)

Counted:
  - Paragraph text, block-quote text, and list-item text (plain running prose)

Requires: pandoc on PATH (no Python third-party packages).

Usage:
  python scripts/wordcount.py reports/05-final/final-report-v1.0.0.md
  python scripts/wordcount.py <file.md> --split-level 2 --budget 800
  python scripts/wordcount.py <file.md> --json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# Headings that mark the start of back-matter; this section and everything
# after it is excluded from the word count.
STOP_HEADINGS = {"references", "bibliography", "works cited"}

# Inline AST node types whose "c" payload is a literal string we should count.
_WORD_STR_TYPES = {"Str"}


def get_pandoc() -> str:
    exe = shutil.which("pandoc")
    if not exe:
        sys.exit("error: pandoc not found on PATH. Install pandoc and retry.")
    return exe


def to_ast(path: str) -> dict:
    exe = get_pandoc()
    try:
        raw = subprocess.run(
            [exe, path, "-t", "json"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: pandoc failed:\n{e.stderr.decode('utf-8', 'replace')}")
    return json.loads(raw.stdout.decode("utf-8"))


def count_words_in_inlines(inlines: list) -> int:
    """Count words in a list of inline AST nodes.

    Skips Code and Math. Recurses into emphasis/links/etc. A "word" is a
    maximal run of non-space characters in the concatenated Str content.
    Footnote (Note) prose is counted and folded into the returned total.
    """
    parts: list[str] = []
    extra = 0

    def walk(nodes: list) -> None:
        nonlocal extra
        for node in nodes:
            if not isinstance(node, dict):
                continue
            t = node.get("t")
            c = node.get("c")
            if t in _WORD_STR_TYPES:
                parts.append(c)
            elif t in ("Space", "SoftBreak", "LineBreak"):
                parts.append(" ")
            elif t in ("Code", "Math"):
                # excluded: inline code and math contribute no prose words
                continue
            elif t in ("Emph", "Strong", "Strikeout", "Superscript",
                       "Subscript", "SmallCaps", "Underline"):
                walk(c)
            elif t == "Quoted":
                walk(c[1])
            elif t in ("Link", "Span"):
                walk(c[1])
            elif t == "Cite":
                # c[0] is citation list, c[1] is the fallback inlines; count
                # neither — citations are references, not prose.
                continue
            elif t == "Note":
                # Footnote content is prose belonging to the current section.
                for blk in c:
                    if blk.get("t") in ("Para", "Plain"):
                        extra += count_words_in_inlines(blk.get("c", []))
            # Unknown inline containers fall through and are ignored.

    walk(inlines)
    text = "".join(parts)
    return len(re.findall(r"\S+", text)) + extra


@dataclass
class Section:
    title: str
    level: int
    words: int = 0


@dataclass
class Counter:
    split_level: int
    sections: list[Section] = field(default_factory=list)
    stopped: bool = False

    def current(self) -> Section:
        if not self.sections:
            self.sections.append(Section("(preamble)", 0))
        return self.sections[-1]

    def add(self, n: int) -> None:
        if not self.stopped and n:
            self.current().words += n


def heading_text(inlines: list) -> str:
    parts: list[str] = []

    def walk(nodes: list) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            t, c = node.get("t"), node.get("c")
            if t == "Str":
                parts.append(c)
            elif t in ("Space", "SoftBreak", "LineBreak"):
                parts.append(" ")
            elif isinstance(c, list):
                walk(c)

    walk(inlines)
    return "".join(parts).strip()


def process_blocks(blocks: list, counter: Counter) -> None:
    for block in blocks:
        if counter.stopped:
            return
        t = block.get("t")
        c = block.get("c")

        if t == "Header":
            level = c[0]
            title = heading_text(c[2])
            if title.strip().lower() in STOP_HEADINGS:
                counter.stopped = True
                return
            if level <= counter.split_level:
                counter.sections.append(Section(title, level))
            # headings themselves are never counted
            continue

        if t in ("Para", "Plain"):
            counter.add(count_words_in_inlines(c))
        elif t == "BlockQuote":
            process_blocks(c, counter)
        elif t in ("BulletList", "OrderedList"):
            items = c[1] if t == "OrderedList" else c
            for item in items:
                process_blocks(item, counter)
        elif t == "DefinitionList":
            for term, defs in c:
                for d in defs:
                    process_blocks(d, counter)
        elif t == "Div":
            process_blocks(c[1], counter)
        # Excluded block types (no counting, no recursion):
        #   Table, Figure, CodeBlock, RawBlock, HorizontalRule
        # Figures/images captions live inside Table/Figure and are skipped.


def format_table(sections: list[Section], budget: int | None) -> str:
    counted = [s for s in sections if s.words > 0 or s.title != "(preamble)"]
    name_w = max([len("Section")] + [len(s.title) for s in counted]) + 2
    lines = []
    header = f"{'Section'.ljust(name_w)}{'Words':>8}"
    if budget:
        header += f"{'Budget':>9}{'Status':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    total = 0
    for s in counted:
        total += s.words
        row = f"{s.title.ljust(name_w)}{s.words:>8}"
        if budget:
            over = s.words - budget
            status = "OK" if over <= 0 else f"+{over}"
            row += f"{budget:>9}{status:>10}"
        lines.append(row)
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL'.ljust(name_w)}{total:>8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="Path to the Markdown report")
    ap.add_argument("--split-level", type=int, default=1,
                    help="Heading level to split sections on (1 or 2). Default 1.")
    ap.add_argument("--budget", type=int, default=None,
                    help="Per-section word budget; flags sections over it.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of a table.")
    args = ap.parse_args()

    ast = to_ast(args.file)
    counter = Counter(split_level=args.split_level)
    process_blocks(ast.get("blocks", []), counter)

    sections = [s for s in counter.sections
                if not (s.title == "(preamble)" and s.words == 0)]

    if args.json:
        payload = {
            "file": args.file,
            "split_level": args.split_level,
            "budget": args.budget,
            "total": sum(s.words for s in sections),
            "sections": [{"title": s.title, "level": s.level, "words": s.words}
                         for s in sections],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_table(sections, args.budget))


if __name__ == "__main__":
    main()
