from __future__ import annotations

import base64
import io
import zipfile
from typing import Any
import os

import httpx
import asyncio


class GitHubClient:
    def __init__(self, token: str, base_url: str | None = None) -> None:
        self._token = token
        resolved = base_url or os.getenv("GITHUB_API_URL") or "https://api.github.com"
        self._base_url = resolved.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get(self, path: str) -> Any:
        url = f"{self._base_url}{path}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, headers=self._headers())
                    response.raise_for_status()
                    return response.json()
            except httpx.RequestError as exc:
                # If last attempt, raise a clear runtime error
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Network error while contacting {url}: {exc}") from exc
                # exponential backoff before retrying
                backoff = 2 ** attempt
                await asyncio.sleep(backoff)

    async def get_workflow_run(self, repository: str, run_id: int) -> dict[str, Any]:
        return await self._get(f"/repos/{repository}/actions/runs/{run_id}")

    async def list_run_jobs(self, repository: str, run_id: int) -> list[dict[str, Any]]:
        payload = await self._get(f"/repos/{repository}/actions/runs/{run_id}/jobs")
        return payload.get("jobs", [])

    async def get_run_logs_download_url(self, repository: str, run_id: int) -> str:
        run = await self.get_workflow_run(repository, run_id)
        return run.get("logs_url", "")

    async def get_run_logs_text(self, repository: str, run_id: int, max_chars: int = 20000) -> str:
        logs_url = await self.get_run_logs_download_url(repository, run_id)
        if not logs_url:
            return ""
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(logs_url, headers=self._headers())
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/zip" in content_type or response.content.startswith(b"PK"):
                return _read_zip_logs(response.content, max_chars=max_chars)
            return response.text[-max_chars:]

    async def get_commit_files(self, repository: str, sha: str) -> list[str]:
        if not sha:
            return []
        payload = await self._get(f"/repos/{repository}/commits/{sha}")
        return [f.get("filename", "") for f in payload.get("files", []) if f.get("filename")]

    async def get_latest_failed_run_id(self, repository: str, workflow_name: str = "") -> int:
        payload = await self._get(
            f"/repos/{repository}/actions/runs?status=completed&conclusion=failure&per_page=30"
        )
        runs = payload.get("workflow_runs", [])
        target_name = workflow_name.strip().lower()
        for run in runs:
            if target_name:
                if str(run.get("name", "")).strip().lower() != target_name:
                    continue
            run_id = run.get("id")
            if isinstance(run_id, int):
                return run_id
        return 0

    async def create_branch(self, repository: str, branch: str, from_sha: str) -> dict[str, Any]:
        body = {"ref": f"refs/heads/{branch}", "sha": from_sha}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/repos/{repository}/git/refs",
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
            return response.json()

    async def create_or_update_file(
        self,
        repository: str,
        path: str,
        branch: str,
        content: str,
        message: str,
        sha: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"{self._base_url}/repos/{repository}/contents/{path}",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code < 400:
                return response.json()

            # Existing file updates require "sha"; fetch and retry automatically.
            if response.status_code == 422 and "sha" not in payload:
                existing_sha = await self.get_file_sha(repository, path, ref=branch)
                if existing_sha:
                    payload["sha"] = existing_sha
                    retry_response = await client.put(
                        f"{self._base_url}/repos/{repository}/contents/{path}",
                        headers=self._headers(),
                        json=payload,
                    )
                    retry_response.raise_for_status()
                    return retry_response.json()

            response.raise_for_status()
            return response.json()

    async def create_pull_request(
        self,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> dict[str, Any]:
        payload = {"title": title, "body": body, "head": head, "base": base}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/repos/{repository}/pulls",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def get_file_sha(self, repository: str, path: str, ref: str = "main") -> str:
        encoded_path = "/".join(segment for segment in path.split("/") if segment)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._base_url}/repos/{repository}/contents/{encoded_path}",
                headers=self._headers(),
                params={"ref": ref},
            )
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            payload = response.json()
            return payload.get("sha", "")

    async def get_file_content(self, repository: str, path: str, ref: str = "main") -> str:
        encoded_path = "/".join(segment for segment in path.split("/") if segment)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._base_url}/repos/{repository}/contents/{encoded_path}",
                headers=self._headers(),
                params={"ref": ref},
            )
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            payload = response.json()
            content = payload.get("content", "")
            encoding = payload.get("encoding", "base64")
            if encoding == "base64" and content:
                import base64 as _b64

                return _b64.b64decode(content).decode("utf-8", errors="replace")
            if isinstance(content, str):
                return content
            return ""


def _read_zip_logs(data: bytes, max_chars: int = 20000) -> str:
    combined: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        for name in zip_file.namelist():
            if not name.endswith(".txt"):
                continue
            try:
                text = zip_file.read(name).decode("utf-8", errors="replace")
                combined.append(f"## {name}\n{text}\n")
            except KeyError:
                continue
    blob = "\n".join(combined)
    return blob[-max_chars:]

