"""Error analysis and retry logic for build failures."""

import json
from pathlib import Path

from ..llm.base import LLMBackend
from .prompts import BUILD_FAILURE_PROMPT, ERROR_ANALYSIS_PROMPT


class BuildError(Exception):
    """Exception raised when a build fails."""

    pass


class ErrorAnalyzer:
    """Analyzes build errors and suggests fixes using LLM."""

    def __init__(self, llm: LLMBackend, verbose: bool = False, log_file: str | None = None):
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file

    def analyze_cmake_error(
        self,
        error_output: str,
        current_flags: list[str],
        project_path: Path,
        cmakelists_content: str | None = None,
    ) -> dict:
        """
        Analyze a CMake configuration error and suggest fixes.

        Returns a dict with:
        - diagnosis: str
        - suggested_flags: list[str]
        - install_commands: list[str]
        - confidence: str
        """
        # Extract relevant excerpt from CMakeLists.txt if available
        cmakelists_excerpt = ""
        if cmakelists_content:
            # Take first 100 lines or full content if shorter
            lines = cmakelists_content.split("\n")
            cmakelists_excerpt = "\n".join(lines[:100])

        prompt = ERROR_ANALYSIS_PROMPT.format(
            current_flags="\n".join(current_flags),
            error_output=error_output,
            cmakelists_excerpt=cmakelists_excerpt,
            project_path=str(project_path),
        )

        if self.verbose:
            print(f"\n[LLM] Analyzing CMake error...")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"CMAKE ERROR ANALYSIS\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"PROMPT:\n{prompt}\n\n")

        response = self.llm.complete(prompt)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"RESPONSE:\n{response}\n\n")

        if self.verbose:
            print(f"[LLM] Response: {response[:500]}...")

        try:
            # Try to parse JSON response, stripping markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            if json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```
            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove trailing ```
            json_str = json_str.strip()

            result = json.loads(json_str)
            return {
                "diagnosis": result.get("diagnosis", "Unknown error"),
                "suggested_flags": result.get("suggested_flags", []),
                "install_commands": result.get("install_commands", []),
                "confidence": result.get("confidence", "low"),
            }
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
                print(f"[ERROR] Response: {response[:500]}...")

            # Return a best-effort result
            return {
                "diagnosis": "Failed to parse LLM response",
                "suggested_flags": [],
                "install_commands": [],
                "confidence": "low",
            }

    def analyze_build_error(
        self,
        error_output: str,
        current_flags: list[str],
    ) -> dict:
        """
        Analyze a compilation/build error and suggest fixes.

        Returns a dict with:
        - diagnosis: str
        - suggested_flags: list[str]
        - compiler_flag_changes: dict[str, str]
        - confidence: str
        - notes: str
        """
        prompt = BUILD_FAILURE_PROMPT.format(
            current_flags="\n".join(current_flags),
            error_output=error_output,
        )

        if self.verbose:
            print(f"\n[LLM] Analyzing build error...")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"BUILD ERROR ANALYSIS\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"PROMPT:\n{prompt}\n\n")

        response = self.llm.complete(prompt)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"RESPONSE:\n{response}\n\n")

        if self.verbose:
            print(f"[LLM] Response: {response[:500]}...")

        try:
            # Try to parse JSON response, stripping markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            if json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```
            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove trailing ```
            json_str = json_str.strip()

            result = json.loads(json_str)
            return {
                "diagnosis": result.get("diagnosis", "Unknown build error"),
                "suggested_flags": result.get("suggested_flags", []),
                "compiler_flag_changes": result.get("compiler_flag_changes", {}),
                "confidence": result.get("confidence", "low"),
                "notes": result.get("notes", ""),
            }
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
                print(f"[ERROR] Response: {response[:500]}...")

            return {
                "diagnosis": "Failed to parse LLM response",
                "suggested_flags": [],
                "compiler_flag_changes": {},
                "confidence": "low",
                "notes": "",
            }
