# -*- coding: utf-8 -*-
"""Assemble the paper's companion Git release under release/eaqi-release/.

Structure (what reviewers need, nothing machine-private):
  eaqi-release/
    README.md                 top-level guide (paper -> repo map)
    LICENSE                   MIT
    code/                     reference implementation (without datasets)
    study_platform/           human-study platform + anonymized ratings + analysis
    results/                  every JSON the paper cites (script outputs)
    manuscript/latex/         main.tex, refs.bib, figures, table fragments
    docs/
      review_guide.md         for reviewers: how to run/verify each claim
      online_iqa_demo.md      IQM comparison web demo instructions
    tools/                    aggregation/figure scripts (verbatim outputs)

Datasets (BSR/, clipstim/, koniq10k/, koniq10k.tgz, BSR_bsds500.tgz),
checkpoints (*.pth), caches (__pycache__), and LaTeX build artifacts are
excluded; the study.db is replaced by the anonymized CSV export + schema.
"""
import io
import os
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REL = os.path.join(ROOT, "release", "eaqi-release")

CODE_EXCLUDE_DIRS = {"__pycache__", "data", "tests_cache"}
TOP_COPY = ["code", "results", "tools"]
PLATFORM_FILES = [
    "analysis.py", "import_manual.py", "make_manual_template.py",
    "make_stimuli.py", "server.py",
    "manual_data_participants.csv", "manual_data_ratings.csv",
    "MANUAL_DATA_README.md", "README.md",
]
LATEX_FILES = [
    "main.tex", "refs.bib", "make_figures.py",
    "fig_location_forest.pdf", "fig_dose_response.pdf", "fig_koniq_srcc.pdf",
    "fig_human_study.pdf",
    "tab_human_h1.tex", "tab_human_conditions.tex",
]
PLATFORM_STATIC = os.path.join(ROOT, "study_platform", "static")
PLATFORM_STIMULI_INFO = "stimuli/manifest.json"


def ignore_pycache(_dir, names):
    return [n for n in names if n in CODE_EXCLUDE_DIRS or n == "__pycache__"]


TOOLS_KEEP = {
    "iqa_demo_server.py", "pair_stability.py", "merge_seeds.py",
    "prepare_release.py", "extend_figures.py", "summarize_studies12.py",
    "test_demo.py", "review_guide.md", "release_readme.md",
}


def ignore_tools(_dir, names):
    if _dir.replace(os.sep, "/").endswith("tools"):
        return sorted(n for n in names if n not in TOOLS_KEEP)
    drop = {"__pycache__"}
    for n in names:
        if n.startswith("_") or n.endswith((".log", ".tmp")):
            drop.add(n)
    return sorted(drop)


def _force_remove(func, path, _exc):
    """chmod + retry (git objects are written read-only)."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree_keep_git(path: str) -> None:
    """Remove the release tree but preserve an existing .git directory."""
    import stat

    def chmod_walker(d):
        for root, dirs, files in os.walk(d):
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), stat.S_IWRITE)
                except OSError:
                    pass

    git_dir = os.path.join(path, ".git")
    keep = None
    if os.path.isdir(git_dir):
        keep = git_dir + ".keep"
        if os.path.exists(keep):
            chmod_walker(keep)
            shutil.rmtree(keep, onerror=_force_remove)
        os.rename(git_dir, keep)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_force_remove)
    os.makedirs(path)
    if keep is not None:
        os.rename(keep, git_dir)


def main() -> None:
    _rmtree_keep_git(REL)
    os.makedirs(os.path.join(REL, "docs"))

    # 1) code, results, tools
    for name in TOP_COPY:
        shutil.copytree(os.path.join(ROOT, name), os.path.join(REL, name),
                        ignore=ignore_tools if name == "tools"
                        else ignore_pycache)

    # 2) study platform (code + static assets + manifest, anonymized data)
    plat_dst = os.path.join(REL, "study_platform")
    os.makedirs(plat_dst)
    for f in PLATFORM_FILES:
        shutil.copy2(os.path.join(ROOT, "study_platform", f),
                     os.path.join(plat_dst, f))
    shutil.copytree(PLATFORM_STATIC, os.path.join(plat_dst, "static"))
    os.makedirs(os.path.join(plat_dst, "stimuli"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "study_platform", "stimuli", "manifest.json"),
                 os.path.join(plat_dst, "stimuli", "manifest.json"))

    # anonymized ratings export straight from study.db (official export format)
    db = os.path.join(ROOT, "study_platform", "study.db")
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT participant, base, cond, is_repeat, rating, severity_delta,"
        " rt_ms, shown_left, trial_index, ts FROM ratings").fetchall()
    con.close()
    out_csv = os.path.join(plat_dst, "exported_ratings.csv")
    with io.open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("participant,base,cond,is_repeat,rating,severity_delta,"
                "rt_ms,shown_left,trial_index,ts\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")

    # 3) manuscript (source only)
    lat_dst = os.path.join(REL, "manuscript", "latex")
    os.makedirs(lat_dst)
    for f in LATEX_FILES:
        shutil.copy2(os.path.join(ROOT, "manuscript", "latex", f),
                     os.path.join(lat_dst, f))

    # 4) README + docs + demo server script (tools/ already copied)
    shutil.copy2(os.path.join(HERE, "review_guide.md"),
                 os.path.join(REL, "docs", "review_guide.md"))
    shutil.copy2(os.path.join(HERE, "release_readme.md"),
                 os.path.join(REL, "README.md"))

    # 5) LICENSE
    lic = os.path.join(REL, "LICENSE")
    with io.open(lic, "w", encoding="utf-8") as f:
        f.write(
            "MIT License\n\n"
            "Copyright (c) 2026 Ali Hosseini, Saeed Sedighi\n\n"
            "Permission is hereby granted, free of charge, to any person "
            "obtaining a copy of this software and associated documentation "
            "files (the \"Software\"), to deal in the Software without "
            "restriction, including without limitation the rights to use, "
            "copy, modify, merge, publish, distribute, sublicense, and/or "
            "sell copies of the Software, and to permit persons to whom the "
            "Software is furnished to do so, subject to the following "
            "conditions:\n\n"
            "The above copyright notice and this permission notice shall be "
            "included in all copies or substantial portions of the "
            "Software.\n\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY "
            "KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE "
            "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE "
            "AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT "
            "HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, "
            "WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING "
            "FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR "
            "OTHER DEALINGS IN THE SOFTWARE.\n")

    n_rating_rows = len(rows)
    print("release assembled at", REL)
    print("anonymized rating rows exported:", n_rating_rows)


if __name__ == "__main__":
    sys.exit(main())
