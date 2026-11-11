# Contributing

Development is ongoing. Discuss significant changes before implementation.
Use Python 3.12, install the documented dependencies, and run `python -m pytest tests -q`.

## Workflow

1. Inspect `git status`; preserve existing work. Fetch updates before starting.
2. Start one task branch from an up-to-date `main` (fast-forward only).
3. Use `feat/<description>`, `fix/<description>`, `refactor/<description>`,
   `docs/<description>`, or `chore/<description>`. Several logical commits may
   belong to one task branch. Keep `main` stable.
4. Make focused changes, run relevant tests, inspect the diff and staged files.
5. Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
6. The owner has authorized ongoing publication of reviewed source and
   documentation updates. After validation and staged-content review, commit
   and push updates to the task branch without requesting approval again.
   Obtain owner approval before merging a pull request. Never merge automatically.
7. After a push, open or update a pull request describing behavior, tests,
   limitations, and model/data dependencies. Prefer reviewed changes to main.

Never commit real plates, footage, datasets, annotations, vehicle metadata,
camera addresses, passwords, keys, `.env`, model checkpoints or local logs.
Use synthetic examples. Inspect staged filenames AND content; `.gitignore`
does not protect already tracked files or old commits. If sensitive content
was committed, stop publication, retain a local backup, assess exposure,
rotate exposed credentials, and agree on history remediation with the owner.
Do not force push, rewrite history, or delete data without explicit agreement.

The owner's legacy workspace has private artifacts in its existing history.
Only the reviewed clean publication directory may become the public repository;
do not add the legacy repository as a remote or merge its history.

No project LICENSE has been selected. Do not add a license without the owner's
decision. Check third-party code and model licensing before redistribution.
