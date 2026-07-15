# CLAUDE.md

## Agent skills

### Issue tracker

Issues live in the repo's GitHub Issues (`YaoyiW27/cuda-kernel-lab`), managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Commit policy

Only commit project files: kernel code (`kernels/`), tests, benchmarks (`benchmarks/`),
profiling docs (`profiling/`), `README.md`, `Makefile`, `requirements.txt`, notebooks.

**Never commit the personal learning workspace.** These are gitignored and stay local only:
`lessons/`, `learning-records/`, `reference/`, `assets/`, `MISSION.md`, `NOTES.md`,
`GLOSSARY.md`, `RESOURCES.md`. Do not `git add` them, and never `git add -f` to override
the ignore.
