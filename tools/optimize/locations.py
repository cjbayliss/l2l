from __future__ import annotations

import os

TOOLS_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TOOLS_DIRECTORY)
JUDGE_TEMPLATE_PATH = os.path.join(TOOLS_DIRECTORY, "judge.txt")
REWRITE_TEMPLATE_PATH = os.path.join(TOOLS_DIRECTORY, "rewrite.txt")


def join_directory(working_directory: str, name: str) -> str:
    return os.path.join(working_directory, name)


def text_stem(path: str) -> str:
    return os.path.basename(path).removesuffix(".txt")


def output_path(working_directory: str, version: str, chapter: str) -> str:
    return os.path.join(
        join_directory(join_directory(working_directory, "out"), version),
        text_stem(chapter) + ".en.txt",
    )
