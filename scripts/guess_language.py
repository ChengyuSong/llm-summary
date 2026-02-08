#!/usr/bin/env python3
"""Guess the primary language of each project using an LLM and update gpr_projects.json."""

import argparse
import json
import os
import sys
from pathlib import Path
from gpr_utils import find_project_dir

try:
    from google import genai
except ImportError:
    genai = None


PROMPT_TEMPLATE = """Given the following project information, what is the primary programming language of this project?

Project name: {name}
URL: {url}

Files in project root directory:
{file_list}

Reply with ONLY the language name, e.g. "C", "C++", "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "PHP", "Ruby", "Mixed" (for multi-language projects where no single language dominates), etc.
If the project is primarily C and C++, reply "C/C++".
If you cannot determine the language, reply "Unknown".
"""


def list_root_files(project_path: Path) -> str:
    """List files and directories in the project root (1 level deep)."""
    try:
        entries = sorted(os.listdir(project_path))
        return "\n".join(entries)
    except OSError as e:
        return f"(error listing files: {e})"


def ask_gemini(client, prompt: str, model_name: str) -> str:
    """Ask Gemini via google.genai client."""
    response = client.models.generate_content(model=model_name, contents=[prompt])
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(description="Guess project languages using an LLM")
    parser.add_argument(
        "--projects-json",
        type=Path,
        default=Path("scripts/gpr_projects.json"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/data/csong/opensource"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling LLM",
    )
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        if genai is None:
            print("Error: google-genai not installed. pip install google-genai")
            sys.exit(1)
        client = genai.Client(vertexai=True, location="global")

    with open(args.projects_json) as f:
        projects = json.load(f)

    for i, project in enumerate(projects):
        name = project["name"]
        url = project.get("url", "")

        # Skip if already has language
        if project.get("language"):
            print(f"[{i+1}/{len(projects)}] {name}: already has language={project['language']}")
            continue

        project_path = find_project_dir(project, args.source_dir)
        if not project_path:
            print(f"[{i+1}/{len(projects)}] {name}: directory not found, skipping")
            continue

        file_list = list_root_files(project_path)
        prompt = PROMPT_TEMPLATE.format(name=name, url=url, file_list=file_list)

        if args.dry_run:
            print(f"[{i+1}/{len(projects)}] {name}:")
            print(f"  dir: {project_path}")
            print(f"  would send prompt ({len(prompt)} chars)")
            continue

        try:
            language = ask_gemini(client, prompt, args.model)
            project["language"] = language
            # Demote non-C/C++ projects to tier 3
            if language not in ("C", "C++", "C/C++") and project.get("tier", 3) < 3:
                print(f"[{i+1}/{len(projects)}] {name}: {language} (tier {project['tier']} -> 3)")
                project["tier"] = 3
            else:
                print(f"[{i+1}/{len(projects)}] {name}: {language}")
        except Exception as e:
            print(f"[{i+1}/{len(projects)}] {name}: error - {e}")

    if not args.dry_run:
        with open(args.projects_json, "w") as f:
            json.dump(projects, f, indent=2)
            f.write("\n")
        print(f"\nUpdated {args.projects_json}")


if __name__ == "__main__":
    main()
