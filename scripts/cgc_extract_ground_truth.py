#!/usr/bin/env python3
"""Extract vulnerability ground truth from CGC cb-multios challenges.

Parses README.md files for CWE classifications and source files for
#ifdef PATCHED / #ifndef PATCHED blocks to build a ground truth JSON.

Usage:
    python scripts/cgc_extract_ground_truth.py [--cgc-dir /path/to/cb-multios] [-o output.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

# CWE -> issue_kind mapping (only memory safety CWEs we can verify)
CWE_TO_ISSUE_KIND = {
    119: "buffer_overflow",   # Improper Restriction of Operations within the Bounds of a Memory Buffer
    120: "buffer_overflow",   # Buffer Copy without Checking Size of Input
    121: "buffer_overflow",   # Stack-based Buffer Overflow
    122: "buffer_overflow",   # Heap-based Buffer Overflow
    124: "buffer_overflow",   # Buffer Underwrite
    125: "buffer_overflow",   # Out-of-bounds Read
    126: "buffer_overflow",   # Buffer Over-read
    127: "buffer_overflow",   # Buffer Under-read
    787: "buffer_overflow",   # Out-of-bounds Write
    788: "buffer_overflow",   # Access of Memory Location After End of Buffer
    415: "double_free",
    416: "use_after_free",
    476: "null_deref",
    457: "uninitialized_use",
    824: "use_after_free",    # Access of Uninitialized Pointer
    # CWEs that don't map to our issue_kinds get None:
    # 134 (format string), 190 (integer overflow), 191 (integer underflow),
    # 193 (off-by-one), 194 (unexpected sign extension),
    # 195 (signed to unsigned conversion), 200/201 (info exposure),
    # 285/287 (improper auth), 468 (incorrect pointer scaling),
    # 680 (integer overflow to buffer overflow), etc.
}


def find_challenges(cgc_dir: Path) -> list[Path]:
    """Find single-binary challenges (skip multi-binary with cb_1/ subdirs)."""
    challenges_dir = cgc_dir / "challenges"
    if not challenges_dir.exists():
        print(f"Error: {challenges_dir} not found", file=sys.stderr)
        sys.exit(1)

    results = []
    for d in sorted(challenges_dir.iterdir()):
        if not d.is_dir():
            continue
        # Skip multi-binary challenges
        if (d / "cb_1").exists():
            continue
        results.append(d)
    return results


def parse_cwes(readme_path: Path) -> list[int]:
    """Extract CWE numbers from README.md."""
    if not readme_path.exists():
        return []
    text = readme_path.read_text(errors="replace")
    return [int(m) for m in re.findall(r"CWE-(\d+)", text)]


def parse_vuln_description(readme_path: Path) -> str:
    """Extract vulnerability description between ## Vulnerability and next ##."""
    if not readme_path.exists():
        return ""
    text = readme_path.read_text(errors="replace")
    m = re.search(
        r"##\s*Vulnerability\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return ""


def find_enclosing_function(lines: list[str], target_line: int) -> str | None:
    """Scan backwards from target_line to find the enclosing function name."""
    brace_depth = 0
    for i in range(target_line - 1, -1, -1):
        line = lines[i]
        # Count braces (reversed: } increases depth, { decreases)
        brace_depth += line.count("}") - line.count("{")
        if brace_depth <= 0:
            # We're at or above the function's opening brace level.
            # Scan this line and nearby lines for a function signature.
            for j in range(i, max(i - 5, -1), -1):
                candidate_line = lines[j].strip()
                if candidate_line.startswith("#") or candidate_line.startswith("//"):
                    continue
                # Match: word followed by ( — typical function definition
                m = re.search(r"\b([a-zA-Z_]\w*)\s*\(", candidate_line)
                if m:
                    name = m.group(1)
                    # Skip C keywords
                    if name not in (
                        "if", "for", "while", "switch", "return",
                        "sizeof", "typeof", "defined", "else",
                    ):
                        return name
            # If we found brace_depth <= 0 but no function name,
            # keep scanning upward
    return None


def parse_patched_blocks(
    file_path: Path, rel_path: str
) -> list[dict]:
    """Parse #ifdef PATCHED / #ifndef PATCHED blocks from a source file."""
    try:
        lines = file_path.read_text(errors="replace").splitlines()
    except Exception:
        return []

    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(
            r"#\s*(ifdef|ifndef)\s+(PATCHED(?:_(\d+))?)\s*$", stripped
        )
        if not m:
            i += 1
            continue

        directive = m.group(1)   # "ifdef" or "ifndef"
        patch_id = m.group(3)    # None or "1", "2", etc.

        # Find matching #else and #endif, handling nesting
        ifdef_line = i
        else_line = None
        endif_line = None
        depth = 1
        j = i + 1
        while j < len(lines) and depth > 0:
            s = lines[j].strip()
            if re.match(r"#\s*(ifdef|ifndef|if)\b", s):
                depth += 1
            elif re.match(r"#\s*else\b", s) and depth == 1:
                else_line = j
            elif re.match(r"#\s*endif\b", s):
                depth -= 1
                if depth == 0:
                    endif_line = j
            j += 1

        if endif_line is None:
            i += 1
            continue

        if directive == "ifdef":
            # #ifdef PATCHED: patched code in body, vulnerable in else
            patched_code = "\n".join(lines[ifdef_line + 1 : else_line or endif_line])
            if else_line is not None:
                vuln_code = "\n".join(lines[else_line + 1 : endif_line])
            else:
                vuln_code = ""  # No else → vulnerability is absence of patch
        else:
            # #ifndef PATCHED: vulnerable code in body, patched in else
            vuln_code = "\n".join(lines[ifdef_line + 1 : else_line or endif_line])
            if else_line is not None:
                patched_code = "\n".join(lines[else_line + 1 : endif_line])
            else:
                patched_code = ""  # No else → patch is to remove the vulnerable code

        func_name = find_enclosing_function(lines, ifdef_line)

        blocks.append({
            "patch_id": int(patch_id) if patch_id else None,
            "file": rel_path,
            "line": ifdef_line + 1,  # 1-indexed
            "function": func_name,
            "vulnerable_code": vuln_code.strip(),
            "patched_code": patched_code.strip(),
        })

        i = endif_line + 1

    return blocks


def extract_patch_defines(
    all_entries: list[dict], challenge_name: str
) -> list[str]:
    """Extract sorted unique PATCHED defines for a challenge from compile_commands.

    Returns e.g. ["PATCHED", "PATCHED_1", "PATCHED_2"].
    """
    defines: set[str] = set()
    challenge_pattern = f"/challenges/{challenge_name}/"
    for entry in all_entries:
        if challenge_pattern not in entry.get("file", ""):
            continue
        for m in re.findall(r"-D(PATCHED(?:_\d+)?)\b", entry.get("command", "")):
            defines.add(m)
    return sorted(defines)


def extract_challenge(challenge_dir: Path) -> dict | None:
    """Extract ground truth for a single challenge."""
    name = challenge_dir.name
    readme = challenge_dir / "README.md"

    cwes = parse_cwes(readme)
    if not cwes:
        return None

    vuln_desc = parse_vuln_description(readme)

    # Find all source files under src/ and lib/
    source_dirs = [challenge_dir / "src", challenge_dir / "lib"]
    source_files = []
    for sd in source_dirs:
        if sd.exists():
            source_files.extend(
                sorted(sd.rglob("*.c")) + sorted(sd.rglob("*.cc"))
                + sorted(sd.rglob("*.cpp"))
            )

    # Parse PATCHED blocks
    all_blocks = []
    for sf in source_files:
        rel = str(sf.relative_to(challenge_dir))
        all_blocks.extend(parse_patched_blocks(sf, rel))

    if not all_blocks:
        return None

    # Assign CWEs to vulnerabilities
    # When N PATCHED_N blocks and N CWEs, assign positionally (CGC convention)
    # Single PATCHED (patch_id=None) gets all CWEs
    numbered_blocks = [b for b in all_blocks if b["patch_id"] is not None]
    unnumbered_blocks = [b for b in all_blocks if b["patch_id"] is None]

    vulnerabilities = []

    if numbered_blocks:
        # Sort by patch_id
        numbered_blocks.sort(key=lambda b: b["patch_id"])
        for block in numbered_blocks:
            pid = block["patch_id"]
            # Assign CWE positionally: PATCHED_1 → first CWE, etc.
            if pid is not None and 1 <= pid <= len(cwes):
                block_cwes = [cwes[pid - 1]]
            else:
                block_cwes = cwes  # fallback: all CWEs
            issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in block_cwes]
            # Use first mappable issue_kind
            issue_kind = next((ik for ik in issue_kinds if ik), None)
            vulnerabilities.append({
                "patch_id": block["patch_id"],
                "file": block["file"],
                "line": block["line"],
                "function": block["function"],
                "issue_kind": issue_kind,
                "cwes": block_cwes,
                "vulnerable_code": block["vulnerable_code"],
                "patched_code": block["patched_code"],
            })

    for block in unnumbered_blocks:
        issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in cwes]
        issue_kind = next((ik for ik in issue_kinds if ik), None)
        vulnerabilities.append({
            "patch_id": block["patch_id"],
            "file": block["file"],
            "line": block["line"],
            "function": block["function"],
            "issue_kind": issue_kind,
            "cwes": cwes,
            "vulnerable_code": block["vulnerable_code"],
            "patched_code": block["patched_code"],
        })

    return {
        "cwes": cwes,
        "vuln_description": vuln_desc,
        "vulnerabilities": vulnerabilities,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract CGC vulnerability ground truth"
    )
    parser.add_argument(
        "--cgc-dir", type=Path,
        default=Path("/data/csong/cgc/cb-multios"),
        help="Path to cb-multios directory",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="cgc_ground_truth.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only extract challenges matching this substring",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    challenges = find_challenges(args.cgc_dir)
    print(f"Found {len(challenges)} single-binary challenges")

    if args.filter:
        challenges = [
            c for c in challenges
            if args.filter.lower() in c.name.lower()
        ]
        print(f"Filter '{args.filter}': {len(challenges)} challenges")

    # Load compile_commands.json for patch define extraction
    cc_path = args.cgc_dir / "build" / "compile_commands.json"
    all_cc_entries = []
    if cc_path.exists():
        with open(cc_path) as f:
            all_cc_entries = json.load(f)

    result = {"challenges": {}}
    stats = {"total": 0, "extracted": 0, "no_cwes": 0, "no_patches": 0,
             "mappable": 0, "unmappable": 0}

    for cdir in challenges:
        stats["total"] += 1
        entry = extract_challenge(cdir)
        if entry is None:
            if args.verbose:
                cwes = parse_cwes(cdir / "README.md")
                if not cwes:
                    print(f"  {cdir.name}: no CWEs found")
                    stats["no_cwes"] += 1
                else:
                    print(f"  {cdir.name}: no PATCHED blocks found")
                    stats["no_patches"] += 1
            continue

        # Add patch defines from compile_commands
        if all_cc_entries:
            entry["patch_defines"] = extract_patch_defines(
                all_cc_entries, cdir.name
            )

        stats["extracted"] += 1
        n_mappable = sum(
            1 for v in entry["vulnerabilities"] if v["issue_kind"]
        )
        n_unmappable = len(entry["vulnerabilities"]) - n_mappable
        stats["mappable"] += n_mappable
        stats["unmappable"] += n_unmappable

        if args.verbose:
            cwes_str = ",".join(f"CWE-{c}" for c in entry["cwes"])
            print(
                f"  {cdir.name}: {cwes_str}, "
                f"{len(entry['vulnerabilities'])} vuln(s), "
                f"{n_mappable} mappable"
            )

        result["challenges"][cdir.name] = entry

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary:")
    print(f"  Challenges scanned: {stats['total']}")
    print(f"  Challenges extracted: {stats['extracted']}")
    print(f"  Vulnerabilities with mappable issue_kind: {stats['mappable']}")
    print(f"  Vulnerabilities with unmappable CWEs: {stats['unmappable']}")
    print(f"\nGround truth written to: {args.output}")


if __name__ == "__main__":
    main()
