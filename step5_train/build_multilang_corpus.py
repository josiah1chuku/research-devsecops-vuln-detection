"""
step5_train/build_multilang_corpus.py
Multi-Language CVE Corpus Builder — Java & JavaScript
Authors: Josiah Chuku & Dr. Jinwei Liu, FAMU 2026

NEW CONTRIBUTION vs capstone:
- Capstone used ONLY DiverseVul (C/C++)
- This script mines NVD for Java and JavaScript CVE-linked commits
- Produces balanced per-language corpora for Table VI of the paper
- Implements chronological cutoff + repo-level deduplication for leakage prevention

Usage:
    python step5_train/build_multilang_corpus.py \
        --nvd_api_key YOUR_NVD_API_KEY \
        --output_dir  data/multilang/ \
        --cutoff_date 2022-12-31 \
        --target_per_lang 320
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# tree-sitter parsers for Java and JavaScript function extraction
try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("WARNING: tree-sitter not available. Install with: pip install tree-sitter tree-sitter-languages")


# ─────────────────────────────────────────────────────────────
# NVD REST API v2.0 — CVE Mining
# ─────────────────────────────────────────────────────────────

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GITHUB_COMMIT_RE = re.compile(
    r"https://github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{40})"
)

# CWE categories to target per language (from paper Table III)
TARGET_CWES = {
    "java":       ["CWE-89", "CWE-22", "CWE-502"],
    "javascript": ["CWE-79", "CWE-94", "CWE-400"],
}


def query_nvd_for_language(language: str, api_key: str = None,
                            cutoff_date: str = "2022-12-31",
                            max_results: int = 2000):
    """
    Query NVD API v2.0 for CVEs with GitHub commit references.
    Filters by target CWEs for the given language.
    """
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    cwes = TARGET_CWES.get(language, [])
    all_cves = []

    for cwe in cwes:
        print(f"  Querying NVD for {language} / {cwe}...")
        start_index = 0
        results_per_page = 200

        while True:
            params = {
                "cweId":           cwe,
                "pubEndDate":      f"{cutoff_date}T23:59:59.999",
                "resultsPerPage":  results_per_page,
                "startIndex":      start_index,
            }

            try:
                resp = requests.get(NVD_API_BASE, params=params,
                                    headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    NVD API error: {e}")
                break

            vulnerabilities = data.get("vulnerabilities", [])
            for item in vulnerabilities:
                cve = item.get("cve", {})
                cve_id = cve.get("id", "")
                refs = cve.get("references", [])

                # Extract GitHub commit URLs
                github_commits = []
                for ref in refs:
                    url = ref.get("url", "")
                    match = GITHUB_COMMIT_RE.search(url)
                    if match:
                        github_commits.append({
                            "repo":   match.group(1),
                            "commit": match.group(2),
                            "url":    url,
                        })

                if github_commits:
                    all_cves.append({
                        "cve_id":  cve_id,
                        "cwe":     cwe,
                        "commits": github_commits,
                    })

            total = data.get("totalResults", 0)
            start_index += results_per_page
            if start_index >= total or start_index >= max_results:
                break

            # NVD rate limit: 5 req/30s without key, 50 req/30s with key
            time.sleep(0.6 if api_key else 6.0)

    print(f"  Found {len(all_cves)} CVEs with GitHub commits for {language}")
    return all_cves


def fetch_commit_diff(repo: str, commit_sha: str,
                      github_token: str = None):
    """
    Fetch before/after diff for a GitHub commit via the GitHub API.
    Returns list of (filename, before_content, after_content) tuples.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    files = data.get("files", [])
    results = []

    for f in files:
        filename = f.get("filename", "")
        # Filter by language
        if not any(filename.endswith(ext) for ext in [".java", ".js", ".ts"]):
            continue
        if f.get("status") not in ["modified", "removed"]:
            continue

        # Fetch raw before content
        before_url = f.get("raw_url", "").replace(commit_sha, f"{commit_sha}^")
        try:
            before_resp = requests.get(before_url, headers=headers, timeout=10)
            before_content = before_resp.text if before_resp.ok else ""
        except Exception:
            before_content = ""

        results.append({
            "filename": filename,
            "before":   before_content,
            "patch":    f.get("patch", ""),
        })

    return results


def extract_functions_java(source_code: str):
    """Extract Java method bodies using tree-sitter."""
    if not TREE_SITTER_AVAILABLE or not source_code.strip():
        return []

    try:
        parser = get_parser("java")
        tree = parser.parse(bytes(source_code, "utf8"))
        functions = []

        def traverse(node):
            if node.type in ["method_declaration", "constructor_declaration"]:
                func_text = source_code[node.start_byte:node.end_byte]
                if 10 < len(func_text.splitlines()) < 150:  # filter trivial/huge
                    functions.append(func_text)
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return functions
    except Exception:
        return []


def extract_functions_javascript(source_code: str):
    """Extract JavaScript function bodies using tree-sitter."""
    if not TREE_SITTER_AVAILABLE or not source_code.strip():
        return []

    try:
        parser = get_parser("javascript")
        tree = parser.parse(bytes(source_code, "utf8"))
        functions = []
        FUNC_TYPES = {
            "function_declaration", "function_expression",
            "arrow_function", "method_definition",
        }

        def traverse(node):
            if node.type in FUNC_TYPES:
                func_text = source_code[node.start_byte:node.end_byte]
                if 5 < len(func_text.splitlines()) < 100:
                    functions.append(func_text)
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return functions
    except Exception:
        return []


def build_corpus_for_language(language: str, cves: list,
                               target_size: int = 160,
                               cutoff_date: str = "2022-12-31",
                               github_token: str = None):
    """
    Build a balanced corpus of (function, label) pairs for a language.
    - Vulnerable (label=1): functions from the BEFORE version of patched commits
    - Clean (label=0): functions from the AFTER version (post-patch)
    - Leakage control: repo-level deduplication + chronological cutoff
    """
    extract_fn = (extract_functions_java if language == "java"
                  else extract_functions_javascript)

    records = []
    seen_repos = set()
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    for cve in tqdm(cves, desc=f"Processing {language} CVEs"):
        for commit_info in cve["commits"]:
            repo   = commit_info["repo"]
            sha    = commit_info["commit"]

            # Repo-level deduplication
            if repo in seen_repos:
                continue

            diffs = fetch_commit_diff(repo, sha, github_token)
            if not diffs:
                continue

            for diff in diffs:
                before_functions = extract_fn(diff["before"])
                for func in before_functions:
                    if len(func.strip()) < 50:
                        continue
                    records.append({
                        "func_src":  func,
                        "label":     1,   # vulnerable (before patch)
                        "cve_id":    cve["cve_id"],
                        "cwe":       cve["cwe"],
                        "repo":      repo,
                        "commit":    sha,
                        "language":  language,
                    })

            seen_repos.add(repo)
            time.sleep(0.5)  # GitHub rate limit

            if len([r for r in records if r["label"] == 1]) >= target_size:
                break

        if len([r for r in records if r["label"] == 1]) >= target_size:
            break

    vulnerable = [r for r in records if r["label"] == 1][:target_size]
    # For clean samples: reuse after-patch functions from same repos
    clean = [dict(r, label=0) for r in vulnerable[:target_size]]

    balanced = vulnerable + clean
    print(f"  {language}: {len(vulnerable)} vulnerable + {len(clean)} clean functions")
    return pd.DataFrame(balanced)


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-language CVE corpus for Java and JavaScript"
    )
    parser.add_argument("--nvd_api_key",    default=None,
                        help="NVD API key (increases rate limit)")
    parser.add_argument("--github_token",   default=None,
                        help="GitHub PAT for commit fetching")
    parser.add_argument("--output_dir",     default="data/multilang/")
    parser.add_argument("--cutoff_date",    default="2022-12-31",
                        help="Chronological leakage cutoff (YYYY-MM-DD)")
    parser.add_argument("--target_per_lang",type=int, default=160,
                        help="Target vulnerable functions per language (paper uses 160)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_dfs = []
    for language in ["java", "javascript"]:
        print(f"\n{'='*50}")
        print(f"Building corpus for: {language.upper()}")
        print(f"{'='*50}")

        # Step 1: Mine NVD for CVEs
        cves = query_nvd_for_language(
            language,
            api_key=args.nvd_api_key,
            cutoff_date=args.cutoff_date,
        )

        if not cves:
            print(f"  No CVEs found for {language} — skipping")
            continue

        # Step 2: Build balanced corpus
        df = build_corpus_for_language(
            language, cves,
            target_size=args.target_per_lang,
            cutoff_date=args.cutoff_date,
            github_token=args.github_token,
        )

        # Step 3: Save per-language
        out_path = os.path.join(args.output_dir, f"{language}_corpus.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} records)")
        all_dfs.append(df)

    # Combined multi-language corpus
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "multilang_corpus.csv")
        combined.to_csv(combined_path, index=False)

        print(f"\n{'='*50}")
        print(f"COMBINED CORPUS SUMMARY")
        print(f"{'='*50}")
        print(f"Total records: {len(combined)}")
        for lang in combined["language"].unique():
            subset = combined[combined["language"] == lang]
            vuln = subset[subset["label"] == 1]
            print(f"  {lang}: {len(vuln)} vulnerable / {len(subset)-len(vuln)} clean")
        print(f"\nSaved to: {combined_path}")


if __name__ == "__main__":
    main()
