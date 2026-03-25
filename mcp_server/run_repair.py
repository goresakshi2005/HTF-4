from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

from mcp_server.config import get_settings
from mcp_server.main import orchestrate_autofix
from mcp_server.tools.github_client import GitHubClient


async def _main() -> int:
    load_dotenv()
    repository = os.getenv("REPOSITORY", "").strip()
    run_id_raw = os.getenv("RUN_ID", "").strip()
    workflow_name = os.getenv("WORKFLOW_NAME", "").strip()
    base_branch = os.getenv("BASE_BRANCH", "main").strip() or "main"

    if not repository:
        print("REPOSITORY is required.")
        return 1

    if run_id_raw:
        try:
            run_id = int(run_id_raw)
        except ValueError:
            print("RUN_ID must be an integer.")
            return 1
    else:
        settings = get_settings()
        github = GitHubClient(settings.GITHUB_TOKEN)
        run_id = await github.get_latest_failed_run_id(repository=repository, workflow_name=workflow_name)
        if not run_id:
            if workflow_name:
                print(f"No failed runs found for workflow '{workflow_name}' in {repository}.")
            else:
                print(f"No failed runs found in {repository}.")
            return 1
        print(f"Using latest failed run id: {run_id}")

    result = await orchestrate_autofix(repository=repository, run_id=run_id, base_branch=base_branch)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

