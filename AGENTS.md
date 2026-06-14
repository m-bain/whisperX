<INSTRUCTIONS>
<core-principles>
- Prefer the simplest, most elegant solution with the fewest LOC and the least complexity.
- Rely on the current stack when possible.
- Run E2E tests before finishing.
</core-principles>

<python>
- Use `uv` for all Python dependency, environment, and test commands.
- Do not install project dependencies into system Python and do not run `pip`, `python -m pytest`, or `pytest` directly for repo work.
- Set up the project with `uv sync --all-extras --dev`.
- Run tests and Python entry points with `uv run`, for example `uv run pytest`.
- If `uv` selects an incompatible system interpreter, use the project-pinned Python from `.python-version` or run `uv python install 3.12` before syncing.
</python>

<execution>
- Call out sandbox or scoping blockers early, explain how to resolve them, and resolve them yourself when possible.
- Do not ask "should I do this?" unless the change is destructive, irreversible, or genuinely blocked.
- Complete all plan steps without pausing for permission between reversible steps.
</execution>

<git-and-worktrees>
- When writing code, create a PR against `master` unless explicitly instructed not to.
- Never ask whether to create a PR. Create it.
- Work from a worktree created from `main` or `master` unless explicitly instructed otherwise.
- Create worktrees under `/tmp` using `git worktree add /tmp/[repo-name]-<task> ...`.
- Do not create worktrees outside `/tmp`.
</git-and-worktrees>

<sandbox-and-cleanup>
- Prefer sandbox-safe commands first. Only escalate if a required command fails in sandbox and no safe alternative exists.
- Avoid commands that commonly trigger escalation, including global installs, GUI or open commands, and writes outside the workspace or `/tmp`.
- Scoped deletion inside `/tmp` and `/private/tmp` is acceptable for disposable worktrees, caches, build artifacts, and other scratch paths.
- For disposable cleanup under `/tmp` or `/private/tmp`, use `tmp-rmrf <path>...` instead of raw `rm -rf`.
- `tmp-rmrf` lives in `~/.codex/bin` and refuses non-temp paths, path traversal outside temp, and deleting `/tmp` or `/private/tmp` themselves.
- For deleting `node_modules`, use `rm-node-modules [path ...]` instead of raw `rm -rf node_modules`.
- `rm-node-modules` lives in `~/.codex/bin`, defaults to `./node_modules`, and refuses any resolved path whose basename is not exactly `node_modules`.
- For a clean reinstall, use `npm-clean-install <dir> [npm install args ...]` instead of chaining `cd ... && rm -rf node_modules && npm install ...`.
- Allowlisted cleanup helpers only match when invoked as standalone commands, not when embedded inside compound shell commands joined with `&&` or `;`.
- Avoid broad or ambiguous destructive commands such as `rm -rf /tmp/*` or wide globs in shared temp directories.
- Avoid destructive shell patterns like `rm -rf`, `rm -f ... && cp -R ...`, or whole-directory replacement when a safer alternative exists.
- Prefer non-destructive workflows that create a fresh temp target and swap only when necessary.
- If a destructive command is still the simplest correct option, explain why before running it.
</sandbox-and-cleanup>
</INSTRUCTIONS>
