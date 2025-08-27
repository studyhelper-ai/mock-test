Monorepo root

Structure

- docs/       — notes, prompts, API contract, DoD
- backend/    — API server, services, migrations, tests
- frontend/   — web app (React/Vue/next) and UI tests
- ai/         — doc -> MCQ pipeline, prompts, evaluation rules

Goals

- Clear separation of concerns
- Reusable AI pipeline under `ai/`
- Docs-first feature design and small, testable components

Quick start

1. Copy `.env.example` to `.env` and fill secrets
2. Initialize git: `git init`
3. Use the `docs/` folder for API contracts and prompts

Definition of Done: see `docs/definition_of_done.md`
