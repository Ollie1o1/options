"""Configure exact-time, DST-aware cloud schedules on cron-job.org -> GitHub Actions.

Why this exists:
  GitHub Actions `on: schedule` runs on a best-effort queue (UTC-only, delayed 30-90m
  or silently dropped under load). `workflow_dispatch` fires immediately (<2s).
  This script registers 3 free, exact-time weekday jobs on cron-job.org in
  `America/New_York` (automatic EST/EDT handling + email alert on any failure) and
  sets the GitHub repo variable `EXTERNAL_CRON_ACTIVE=true` so the fallback
  `on: schedule` triggers never create duplicate Discord pings.

Usage:
  1) Preview copy-paste settings (no keys needed):
       python scripts/setup_cloud_cron.py --print-only

  2) Auto-create all 3 jobs via cron-job.org REST API:
       set CRONJOB_API_KEY=<key from cron-job.org Console -> Settings>
       set GH_DISPATCH_PAT=<fine-grained GitHub PAT with Actions: Read & write on Ollie1o1/options>
       python scripts/setup_cloud_cron.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REPO = "Ollie1o1/options"
DEFAULT_WORKFLOW = "discord_dispatch.yml"
DEFAULT_REF = "main"
DEFAULT_TIMEZONE = "America/New_York"
CRONJOB_API_BASE = "https://api.cron-job.org"


def build_dispatch_specs(
    *,
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    ref: str = DEFAULT_REF,
    timezone: str = DEFAULT_TIMEZONE,
) -> List[Dict[str, Any]]:
    """Return the 3 canonical weekday Discord dispatch slot definitions."""
    dispatch_url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    return [
        {
            "title": "Options Desk · Morning Brief (8:30 AM ET)",
            "hour": 8,
            "minute": 30,
            "timezone": timezone,
            "wdays": [1, 2, 3, 4, 5],
            "url": dispatch_url,
            "body": {
                "ref": ref,
                "inputs": {
                    "mode": "morning",
                },
            },
        },
        {
            "title": "Options Desk · Mid-Morning Sweep (11:00 AM ET)",
            "hour": 11,
            "minute": 0,
            "timezone": timezone,
            "wdays": [1, 2, 3, 4, 5],
            "url": dispatch_url,
            "body": {
                "ref": ref,
                "inputs": {
                    "mode": "options",
                    "strategy": "all",
                    "session": "midday",
                    "only_if_good": "true",
                },
            },
        },
        {
            "title": "Options Desk · Afternoon Sweep + Exits (2:30 PM ET)",
            "hour": 14,
            "minute": 30,
            "timezone": timezone,
            "wdays": [1, 2, 3, 4, 5],
            "url": dispatch_url,
            "body": {
                "ref": ref,
                "inputs": {
                    "mode": "afternoon",
                    "strategy": "all",
                    "session": "afternoon",
                    "only_if_good": "true",
                },
            },
        },
    ]


def build_cronjob_org_payload(spec: Dict[str, Any], github_pat: str) -> Dict[str, Any]:
    """Build a cron-job.org REST API `PUT /jobs` or `PATCH /jobs/{id}` payload."""
    return {
        "job": {
            "title": spec["title"],
            "url": spec["url"],
            "enabled": True,
            "saveResponses": True,
            "requestMethod": 1,  # POST
            "requestTimeout": 30,
            "schedule": {
                "timezone": spec["timezone"],
                "expiresAt": 0,
                "hours": [int(spec["hour"])],
                "minutes": [int(spec["minute"])],
                "mdays": [-1],
                "months": [-1],
                "wdays": list(spec["wdays"]),
            },
            "notification": {
                "onFailure": True,
                "onFailureCount": 1,
                "onSuccess": False,
                "onDisable": True,
            },
            "extendedData": {
                "headers": {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {github_pat}",
                    "Content-Type": "application/json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                "body": json.dumps(spec["body"], separators=(",", ":")),
            },
        }
    }


def _api_request(
    method: str,
    path: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{CRONJOB_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "options-desk-cron-setup/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"cron-job.org {method} {path} failed ({exc.code}): {body}") from exc


def print_manual_guide(specs: List[Dict[str, Any]]) -> None:
    print("=== cron-job.org Exact-Time Cloud Dispatch Setup ===")
    print("1. Create a fine-grained GitHub PAT at:")
    print("   https://github.com/settings/personal-access-tokens/new")
    print("   - Repository access: Only select repositories -> Ollie1o1/options")
    print("   - Permissions -> Repository permissions -> Actions: Read and write")
    print("")
    print("2. In https://console.cron-job.org/jobs create these 3 jobs (Method: POST):")
    print("   Headers for all 3 jobs:")
    print("     Accept: application/vnd.github+json")
    print("     Authorization: Bearer <YOUR_GITHUB_PAT>")
    print("     Content-Type: application/json")
    print("     X-GitHub-Api-Version: 2022-11-28")
    print("")
    for idx, s in enumerate(specs, start=1):
        print(f"   Job {idx}: {s['title']}")
        print(f"     URL      : {s['url']}")
        print(f"     Schedule : Mon-Fri at {s['hour']:02d}:{s['minute']:02d} ({s['timezone']})")
        print(f"     Body     : {json.dumps(s['body'])}")
        print("")
    print("3. Once created, silence GitHub's fallback `on: schedule` to prevent duplicate pings:")
    print("   gh variable set EXTERNAL_CRON_ACTIVE --body true")


def sync_cronjob_org(
    specs: List[Dict[str, Any]],
    *,
    api_key: str,
    github_pat: str,
    dry_run: bool = False,
    set_gh_var: bool = True,
) -> List[str]:
    """Create or update all 3 cron-job.org jobs idempotently by title."""
    if dry_run:
        actions: List[str] = []
        for s in specs:
            payload = build_cronjob_org_payload(s, github_pat="<REDACTED>")
            print(json.dumps(payload, indent=2))
            actions.append(f"dry-run: {s['title']}")
        return actions

    _, existing_resp = _api_request("GET", "/jobs", api_key)
    existing_by_title: Dict[str, int] = {}
    for j in existing_resp.get("jobs") or []:
        if isinstance(j, dict) and j.get("title") and j.get("jobId") is not None:
            existing_by_title[str(j["title"])] = int(j["jobId"])

    results: List[str] = []
    for s in specs:
        payload = build_cronjob_org_payload(s, github_pat=github_pat)
        title = str(s["title"])
        if title in existing_by_title:
            job_id = existing_by_title[title]
            _api_request("PATCH", f"/jobs/{job_id}", api_key, payload)
            msg = f"Updated job #{job_id}: {title}"
        else:
            _, created = _api_request("PUT", "/jobs", api_key, payload)
            job_id = int(created.get("jobId") or 0)
            msg = f"Created job #{job_id}: {title}"
        print(f"✓ {msg}")
        results.append(msg)

    if set_gh_var:
        res = subprocess.run(
            ["gh", "variable", "set", "EXTERNAL_CRON_ACTIVE", "--body", "true"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("✓ Set GitHub repo variable EXTERNAL_CRON_ACTIVE=true (silenced fallback schedule).")
        else:
            print(
                "Note: Run `gh variable set EXTERNAL_CRON_ACTIVE --body true` to disable fallback GitHub schedule.",
                file=sys.stderr,
            )
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Set up exact-time cron-job.org triggers for Discord Dispatch.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help=f"Workflow file (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--ref", default=DEFAULT_REF, help=f"Git ref/branch (default: {DEFAULT_REF})")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help=f"IANA timezone (default: {DEFAULT_TIMEZONE})")
    parser.add_argument("--cronjob-api-key", default=os.environ.get("CRONJOB_API_KEY", ""), help="cron-job.org API key")
    parser.add_argument("--github-pat", default=os.environ.get("GH_DISPATCH_PAT", ""), help="Fine-grained GitHub PAT")
    parser.add_argument("--print-only", action="store_true", help="Print manual web-UI setup instructions and exit")
    parser.add_argument("--dry-run", action="store_true", help="Preview redacted API payloads without calling API")
    parser.add_argument("--no-gh-var", action="store_true", help="Do not set EXTERNAL_CRON_ACTIVE repo variable")
    args = parser.parse_args(argv)

    specs = build_dispatch_specs(
        repo=args.repo,
        workflow=args.workflow,
        ref=args.ref,
        timezone=args.timezone,
    )
    if args.print_only:
        print_manual_guide(specs)
        return 0

    if not args.dry_run:
        if not args.cronjob_api_key.strip():
            print("ERROR: Set CRONJOB_API_KEY (or pass --cronjob-api-key). Use --print-only for manual UI setup.", file=sys.stderr)
            return 1
        if not args.github_pat.strip():
            print("ERROR: Set GH_DISPATCH_PAT (or pass --github-pat). Use --print-only for manual UI setup.", file=sys.stderr)
            return 1

    sync_cronjob_org(
        specs,
        api_key=args.cronjob_api_key.strip(),
        github_pat=args.github_pat.strip(),
        dry_run=args.dry_run,
        set_gh_var=not args.no_gh_var,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
