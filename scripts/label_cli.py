"""Label a labelling sheet in the terminal, one pair at a time (L-B).

    python scripts/label_cli.py ~/projects/PhoneSense-share/sheets/labels_share_3of3.csv

Shows the search and the phone, you type a grade 0-3 and press Enter. The sheet is saved after
every answer, so you can quit any time (q) and run the same command later: it continues with the
first unlabelled pair. Keys: 0 1 2 3 = grade, s = skip for now, u = undo the last answer,
q = save and quit. The sheet comes from scripts/label_sheet.py export and never shows the LLM
grade; when you are done, send the CSV back, or merge it yourself with
``python scripts/label_sheet.py import --sheet <file>``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable
from pathlib import Path

GRADES = {"0", "1", "2", "3"}
RUBRIC = """\
Grade how well the phone answers the search:
  3 = perfect: clearly what they want, every stated requirement is confirmed in the listing
  2 = relevant: a good fit; at most minor requirements are unconfirmed or roughly met
  1 = marginal: related, but a key requirement is unconfirmed or only partly met
  0 = not relevant: not a phone (case, charger...), the wrong model, or the listing clearly
      breaks a requirement
A spec the listing does not state is "unconfirmed", not wrong: never 0 for that alone, but not 3."""


def load(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in ("qid", "query", "doc_id", "phone", "grade") if c not in fields]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return fields, rows


def save(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write the whole sheet atomically (temp file, then replace), so a crash cannot corrupt it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _query_text(cell: str) -> str:
    """The sheet's query cell is the search plus a bracketed rubric line; show the search only."""
    return cell.split("\n[", 1)[0].strip()


def run(
    path: Path,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Interactive loop; returns the number of pairs labelled in this session."""
    fields, rows = load(path)
    todo = [i for i, r in enumerate(rows) if not (r["grade"] or "").strip()]
    total = len(rows)
    show(RUBRIC + "\n")
    show(f"{total - len(todo)}/{total} already labelled, {len(todo)} to go. "
         "Keys: 0-3 grade, s skip, u undo, q quit.\n")
    history: list[int] = []  # indices answered this session, for undo
    position = 0
    skipped: list[int] = []
    while position < len(todo):
        i = todo[position]
        row = rows[i]
        done = total - len(todo) + len(history)
        show(f"--- pair {done + 1}/{total}  (query {row['qid']}) " + "-" * 24)
        show(f"SEARCH: {_query_text(row['query'])}\n")
        show(row["phone"] + "\n")
        answer = ask("grade [0-3 / s / u / q]: ").strip().lower()
        if answer in GRADES:
            row["grade"] = answer
            history.append(i)
            save(path, fields, rows)
            position += 1
        elif answer == "s":
            skipped.append(i)
            position += 1
        elif answer == "u":
            if not history:
                show("nothing to undo\n")
                continue
            last = history.pop()
            rows[last]["grade"] = ""
            save(path, fields, rows)
            todo.insert(position, last)  # show it again
        elif answer == "q":
            break
        else:
            show("please type 0, 1, 2, 3, s, u or q\n")
    if skipped and position >= len(todo):
        show(f"\n{len(skipped)} skipped pair(s) are still blank: run the command again to do them.")
    left = sum(1 for r in rows if not (r["grade"] or "").strip())
    show(f"\nsaved to {path}. labelled this session: {len(history)}. still blank: {left}.")
    if left == 0:
        show("All done! Send the CSV back (private chat only: it contains Amazon text).")
    return len(history)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sheet", type=Path, help="a CSV from scripts/label_sheet.py export")
    args = ap.parse_args(argv)
    if not args.sheet.is_file():
        print(f"no such file: {args.sheet}", file=sys.stderr)
        return 1
    try:
        run(args.sheet)
    except ValueError as exc:  # not a labelling sheet
        print(str(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):  # Ctrl-C / Ctrl-D: every answer was already saved
        print("\nstopped; your answers so far are saved. Run the same command to continue.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
