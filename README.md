# Agentic CI/CD MCP Orchestrator

Python MCP server for GitHub Actions failure diagnosis, LLM-driven unified-diff auto-repair PR creation, and governed release orchestration.

## What this provides

- MCP tooling to inspect failed workflow runs using commit, logs, and test signals.
- LLM diagnosis flow powered by OpenAI `gpt-4o-mini`.
- Governance layer to auto-fix low-risk issues and require human review for risky changes.
- Generic model-driven repair loop (up to 3 attempts) using unified diff patches.
- GitHub Actions workflows for CI, repair orchestration, and release policy gating.

## Project layout

- `mcp_server/main.py` - MCP tools and orchestration entrypoints
- `mcp_server/run_repair.py` - workflow-safe command runner
- `mcp_server/config.py` - typed environment config
- `mcp_server/tools/*` - GitHub, diagnosis, risk, and PR automation modules
- `.github/workflows/*` - CI/CD automation workflows

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy environment defaults:
   - `cp .env.example .env` (or create `.env` manually on Windows)
4. Fill in required values (`OPENAI_API_KEY`, `GITHUB_TOKEN`).

## Run MCP server locally

- `python -m mcp_server.main`

## Use in Cursor as MCP

- MCP config is included at `.cursor/mcp.json`.
- Restart Cursor so it loads the MCP server definition.
- Ensure your `.env` has `OPENAI_API_KEY` and `GITHUB_TOKEN`.
- In Cursor chat, call tools from `agentic-cicd-orchestrator` with:
  - `repository`: `owner/repo`
  - `run_id`: workflow run id (integer)
- Main tools:
  - `inspect_pipeline_failure`
  - `orchestrate_autofix`

## Run repair orchestration manually

- Set `REPOSITORY` (for example `org/repo`).
- Optional:
  - `RUN_ID` (if omitted, latest failed run is auto-selected)
  - `WORKFLOW_NAME` (filter latest failed run by workflow name, e.g. `ci`)
  - `BASE_BRANCH` (default `main`)
- Execute: `python -m mcp_server.run_repair`

## LLM auto-repair controls

- `MAX_REPAIR_ATTEMPTS` - number of patch generation/application retries (default `3`).
- `PATCH_STRATEGY` - patch format expected from model (must be `unified_diff`).
- `LLM_PATCH_MAX_CHARS` - upper bound on patch payload size.

## Governance model

- `risk_score < RISK_AUTO_FIX_THRESHOLD` -> autonomous auto-fix PR path.
- `RISK_AUTO_FIX_THRESHOLD <= risk_score < RISK_HUMAN_REVIEW_THRESHOLD` -> human approval required.
- `risk_score >= RISK_HUMAN_REVIEW_THRESHOLD` or high-risk file categories -> blocked/review-only path.
- `FORCE_AUTOFIX_ALL=true` -> bypass thresholds and force auto-fix path (dangerous; use only in controlled testing).

## Security notes

- Use least-privilege GitHub credentials.
- Keep production deployment credentials separate from auto-repair identity.
- Review generated PRs and audit artifacts before enabling automerge in production.

