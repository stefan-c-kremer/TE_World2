#!/usr/bin/env python3
"""Render the original-paper outcome grid from a study manifest and ledger."""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


TERMINAL_STATUSES = {"host_extinction", "te_extinction", "maximum_generations"}
COLORS = {
    "host_extinction": "#440154",
    "maximum_generations": "#21918c",
    "te_extinction": "#fde725",
    "incomplete": "#bdbdbd",
}
STATUS_ORDER = {
    "incomplete": 0,
    "host_extinction": 1,
    "maximum_generations": 2,
    "te_extinction": 3,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def final_outcomes(manifest: Path, ledger: Path) -> dict[str, list[str]]:
    manifest_rows = read_csv(manifest)
    terminal_by_run: dict[tuple[str, str], str] = {}
    for row in read_csv(ledger):
        condition = row.get("condition_code", "")
        status = row.get("scientific_status", "")
        if condition.endswith("-Z") and status in TERMINAL_STATUSES:
            terminal_by_run[(condition, row["run"])] = status

    outcomes: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        condition = row["condition_code"]
        if not condition.endswith("-Z"):
            continue
        outcomes[condition].append(
            terminal_by_run.get((condition, row["run"]), "incomplete")
        )
    return dict(outcomes)


def grid_position(condition: str) -> tuple[int, int]:
    bits, suffix = condition.split("-")
    if suffix != "Z" or len(bits) != 7 or set(bits) - {"H", "L"}:
        raise ValueError(f"Unsupported condition code: {condition}")

    high = [bit == "H" for bit in bits]
    # Original-paper column order: progeny, excision, death, insertion.
    # The corrected experiments fix excision at Low, selecting columns 4-7/12-15.
    column = (0 if high[0] else 8) + 4 + (0 if high[1] else 2) + (0 if high[2] else 1)
    # Original-paper row order, from coarsest to finest grouping.
    row = (0 if high[3] else 8) + (0 if high[4] else 4) + (0 if high[5] else 2) + (0 if high[6] else 1)
    return column, row


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_svg(outcomes: dict[str, list[str]]) -> str:
    width, height = 1240, 1120
    grid_x, grid_y = 220, 205
    cell_w, cell_h = 49.75, 49.75
    grid_w = cell_w * 16
    grid_h = cell_h * 16
    right_x = grid_x + grid_w + 18
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">TE accumulation outcomes for corrected -Z experiments</title>',
        '<desc id="desc">Original-paper parameter grid with excision rate fixed Low. High-excision columns and unfinished replicate slots are grey.</desc>',
        '<rect width="1240" height="1120" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.subtitle{font-size:13px}.factor{font-size:14px}.bit{font-size:14px;font-weight:700}.legend{font-size:13px}.note{font-size:12px}.frame{fill:none;stroke:#333;stroke-width:1.5}.header{fill:#fafafa;stroke:#8b8b8b;stroke-width:1.2}.slot{stroke:#fff;stroke-width:.7;shape-rendering:crispEdges}</style>',
        '<text class="title" x="620" y="30" text-anchor="middle">TE accumulation outcomes for corrected -Z experiments</text>',
        '<text class="subtitle" x="620" y="53" text-anchor="middle">Excision rate fixed Low; six replicate slots per original-paper parameter combination</text>',
        '<rect class="frame" x="210" y="70" width="985" height="943" rx="8"/>',
    ]

    horizontal = [
        ("TE_progeny", 80, 28, 8),
        ("TE_excision_rate", 111, 28, 4),
        ("TE_death_rate", 142, 28, 2),
        ("Insertion_bias", 173, 28, 1),
    ]
    for label, y, box_h, span in horizontal:
        lines.append(f'<text class="factor" x="201" y="{y + 19}" text-anchor="end">{esc(label)}</text>')
        for start in range(0, 16, span):
            x = grid_x + start * cell_w
            box_w = span * cell_w - 3
            value = "H" if (start // span) % 2 == 0 else "L"
            lines.append(f'<rect class="header" x="{x:.2f}" y="{y}" width="{box_w:.2f}" height="{box_h}"/>')
            lines.append(f'<text class="bit" x="{x + box_w / 2:.2f}" y="{y + 19}" text-anchor="middle">{value}</text>')

    vertical = [
        ("Carrying_capacity", 1),
        ("Mutation_effect", 2),
        ("NC_BP", 4),
        ("Corrected_mutation_rate", 8),
    ]
    for factor_index, (label, span) in enumerate(vertical):
        x = right_x + factor_index * 39
        lines.append(
            f'<text class="factor" transform="translate({x + 19:.2f} 192) rotate(-90)" text-anchor="start">{esc(label)}</text>'
        )
        for start in range(0, 16, span):
            y = grid_y + start * cell_h
            box_h = span * cell_h - 3
            value = "H" if (start // span) % 2 == 0 else "L"
            lines.append(f'<rect class="header" x="{x:.2f}" y="{y:.2f}" width="34" height="{box_h:.2f}"/>')
            lines.append(f'<text class="bit" x="{x + 17:.2f}" y="{y + box_h / 2 + 5:.2f}" text-anchor="middle">{value}</text>')

    positions = {grid_position(condition): statuses for condition, statuses in outcomes.items()}
    totals = Counter()
    for row in range(16):
        for column in range(16):
            statuses = positions.get((column, row))
            slots = ["incomplete"] * 6
            if statuses is not None:
                slots = ["incomplete"] * (6 - len(statuses)) + list(statuses)
                totals.update(statuses)
            slots.sort(key=STATUS_ORDER.__getitem__)
            x = grid_x + column * cell_w
            y = grid_y + row * cell_h
            inner_x, inner_y = x + 1.5, y + 1.5
            inner_w, inner_h = cell_w - 4, cell_h - 4
            slot_h = inner_h / 6
            lines.append(f'<g data-column="{column}" data-row="{row}">')
            for slot, status in enumerate(slots):
                slot_y = inner_y + slot * slot_h
                lines.append(
                    f'<rect class="slot" x="{inner_x:.2f}" y="{slot_y:.2f}" width="{inner_w:.2f}" height="{slot_h:.2f}" fill="{COLORS[status]}"/>'
                )
            lines.append(f'<rect x="{inner_x:.2f}" y="{inner_y:.2f}" width="{inner_w:.2f}" height="{inner_h:.2f}" fill="none" stroke="#f5f5f5" stroke-width="1"/>')
            lines.append('</g>')

    legend_y = 1065
    legend_items = [
        ("host_extinction", "Host extinction", 220),
        ("maximum_generations", "TE accumulation", 440),
        ("te_extinction", "TE extinction", 660),
        ("incomplete", "Not conducted or incomplete", 880),
    ]
    for status, label, x in legend_items:
        lines.append(f'<rect x="{x}" y="{legend_y - 10}" width="24" height="16" fill="{COLORS[status]}" stroke="#777" stroke-width=".6"/>')
        lines.append(f'<text class="legend" x="{x + 32}" y="{legend_y + 3}">{esc(label)}</text>')

    incomplete = totals["incomplete"]
    lines.append(
        f'<text class="note" x="620" y="1098" text-anchor="middle">128 -Z configurations; 3 runs each; 384 total runs; {incomplete} incomplete</text>'
    )
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args=None) -> int:
    options = parse_arguments(args)
    outcomes = final_outcomes(options.manifest, options.ledger)
    if len(outcomes) != 128 or any(len(statuses) != 3 for statuses in outcomes.values()):
        raise SystemExit("Expected 128 -Z conditions with three replicates each")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(render_svg(outcomes), encoding="utf-8")
    totals = Counter(status for statuses in outcomes.values() for status in statuses)
    print(options.output)
    print(dict(sorted(totals.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
