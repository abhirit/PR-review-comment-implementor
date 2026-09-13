# PR Review Comment Implementor

An agent that reads the review comments on a GitHub pull request, works out
which ones ask for a code change, and implements them in your checkout —
grounded by retrieval over the repository, verified by your own test suite, and
rolled back when it cannot get a change to pass.

Built with **LangGraph** (the control flow), **LangChain** (model, tools and
text splitting), **Claude** (`claude-opus-5` by default) and a **hybrid RAG**
index over the repo.

```
pr-agent serve                                    # web UI on localhost:8765
pr-agent run "owner/repo#123" --auto-validate     # or drive it from the terminal
```

---

## What it actually does

For each review thread on the PR:

1. **Triage** — decides whether the comment asks for a code change, asks a
   question, or needs nothing. Praise and resolved threads are left alone.
2. **Retrieve** — pulls the relevant code out of a hybrid index (below), plus
   the lines around the comment's own anchor in the file it is attached to.
3. **Plan** — commits to a short, concrete plan before touching anything.
4. **Implement** — edits files through sandboxed tools (`read_file`,
   `edit_file`, `write_file`, `search_repository`, `semantic_search`,
   `list_directory`).
5. **Validate** — runs your checks. On failure it reads the output and retries,
   up to `--max-fix-attempts`.
6. **Record** — keeps the change if it passed, **rolls it back if it did not**,
   so a broken change never rides along with the good ones.

Then once, at the end: commit, optionally push, optionally reply to each thread
and resolve the ones it implemented.

### The graph

```
        load_pr ──▶ index_repo ──▶ next_thread
                                       │
                   (queue empty) ──────┼──▶ finalize ──▶ END
                                       │
                                    triage ──(not actionable)──▶ record
                                       │
                                   retrieve ──▶ plan ──▶ implement ──▶ validate
                                                                          │
                                                  (failed, attempts left) │
                                                          ┌── fix ◀───────┤
                                                          └───────▶ validate
                                                                          │
                                                            record ◀──────┘
                                                               │
                                                               └──▶ next_thread
```

Threads are processed **one at a time**. Two comments often touch the same
file, and a queue removes any chance of concurrent edits clobbering each other.

---

## The web UI

```bash
pip install -e ".[web]"
pr-agent serve --repo ./checkout      # http://127.0.0.1:8765
```

The UI drives the same graph as the CLI, and shows the run as it happens:

- **Review threads** — load a PR's comments and pick which ones to act on.
- **Mode** — *dry run* (plan, diff and replies, changes nothing), *implement
  locally* (edits and commits), or *implement and publish* (also pushes and
  replies). Anything that leaves your machine asks for confirmation first.
- **Live run** — the pipeline lights up stage by stage, and each comment gets a
  card showing its triage decision, the plan, the files touched, the validation
  output, a coloured **diff** of what changed, and the reply the agent would
  post. Failed checks and their retries appear as they happen.
- **Retrieval** — query the index directly to check whether a comment's subject
  is findable before you run.
- **History** — replay any earlier run from this server session.

Progress streams over server-sent events, and every event is retained, so a
browser that connects late — or reconnects — replays the whole run rather than
showing a blank page.

It is a local tool: it binds to `127.0.0.1`, ships no authentication, and edits
files and runs your validation commands. Do not expose it to a network you do
not control. Two runs against one checkout are refused, since they would edit
the same files.

---

## Retrieval (RAG)

Reviewers write in two registers, so the index serves both:

- **BM25 over code-aware tokens.** `parseHTTPResponse` is indexed as
  `parsehttpresponse`, `parse`, `http`, `response` — so "rename `parse_lines`"
  finds the exact identifier even if the reviewer wrote it in another case.
- **Dense vectors** for paraphrases like *"this should validate the token
  before using it"*, where no shared keyword exists.

The two result lists are fused with **reciprocal rank fusion**, so a chunk
found by both ranks above one found by either.

Other details that matter in practice:

- **Language-aware chunking.** Files split on function and class boundaries
  using the splitter for their language, not at arbitrary character offsets.
- **Line numbers on every chunk**, so a retrieved excerpt is directly
  actionable.
- **Incremental embedding.** Chunking is redone each run (it is cheap);
  embedding is not, so only files whose content hash changed are re-embedded.
  Changing the embedding model or chunk size invalidates the store and forces a
  clean rebuild.
- **The excerpt is never trusted for editing.** The model is told to
  `read_file` before it edits, because a retrieved chunk can be stale.

Embeddings are optional. With `--embeddings none` the agent runs on BM25 alone
— no model download, no extra API key, and still useful, since review comments
usually name the identifier they are about.

---

## Install

```bash
git clone https://github.com/abhirit/PR-review-comment-implementor
cd PR-review-comment-implementor

python -m venv .venv && source .venv/bin/activate
pip install -e .

# optional: persistent vector store + local embeddings
pip install -e ".[vector,local-embeddings]"

# optional: Voyage AI embeddings instead (strong on code)
pip install -e ".[voyage]"

# optional: the web UI
pip install -e ".[web]"
```

Then copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` and
`GITHUB_TOKEN`.

> Anthropic does not serve an embeddings endpoint, which is why the embedding
> backend is configured separately from the chat model.

---

## Use

```bash
# See what the agent would consider, and act on nothing.
pr-agent comments "owner/repo#123"

# Dry run: edits nothing permanently, prints the plan and the replies it would post.
pr-agent run "owner/repo#123" --repo ./checkout --dry-run

# The normal run: implement, verify with your checks, commit locally.
pr-agent run "owner/repo#123" --repo ./checkout --auto-validate

# Full loop: implement, verify, commit, push to the PR branch, reply, resolve.
pr-agent run "owner/repo#123" --repo ./checkout \
    --validate "ruff check ." --validate "pytest -q" \
    --push --reply --resolve

# One specific comment only.
pr-agent run "owner/repo#123" --comment-id 1234567890

# Inspect retrieval quality on its own.
pr-agent index --repo ./checkout
pr-agent search "where are review threads grouped" --repo ./checkout
```

Check out the PR branch before running, so the agent edits the code the
reviewer was looking at:

```bash
git fetch origin pull/123/head:pr-123 && git checkout pr-123
```

### Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--repo`, `-r` | `.` | Local checkout to edit. |
| `--dry-run` | off | Do everything except commit, push and post. |
| `--no-commit` | commits | Leave changes in the working tree. |
| `--push` | off | Push the commit to the PR's head branch. |
| `--reply` | off | Post a reply on each thread that was handled. |
| `--resolve` | off | Also resolve implemented threads (implies `--reply`). |
| `--validate` | none | A check to run after each change. Repeatable. |
| `--auto-validate` | off | Detect checks from the repo (ruff, pytest, npm, go). |
| `--max-fix-attempts` | 3 | Retries after a failing check before rolling back. |
| `--embeddings` | `local` | `local`, `voyage` or `none`. |
| `--comment-id` | all | Only handle these comment ids. Repeatable. |
| `--ignore-author` | none | Skip a login's comments. Repeatable. |
| `--self-login` | none | The agent's own login, so it skips threads it already answered. |

Everything is also settable by environment variable — see `.env.example`.

---

## Safety

The design assumes the agent will sometimes be wrong.

- **Every file path is sandboxed.** Writes resolve against the repository root
  and anything escaping it is refused, so `../../etc/passwd` fails as a tool
  error rather than a write.
- **Per-comment transactions.** Each thread's edits are snapshotted. If the
  change fails validation after its retries, the files are restored to exactly
  their prior contents — so one bad change cannot contaminate the good ones in
  the same run.
- **Nothing leaves the machine unless you ask.** Committing is local; `--push`,
  `--reply` and `--resolve` are each explicit opt-ins.
- **Validation is the gate.** With no `--validate` command the agent warns you
  that nothing is being verified. Configure your checks; they are what makes
  the loop trustworthy.
- **The model is told not to weaken tests** to get a green run, and not to
  revert the reviewer's request to make a check pass.
- **Scope discipline.** Prompts push for the smallest change that satisfies the
  comment, with no drive-by refactors.

Review the diff before merging. This is an assistant, not an approver.

---

## Configuration

All settings come from the environment or a `.env` file (see `.env.example`).
Notable ones:

| Variable | Default | Notes |
| --- | --- | --- |
| `PR_AGENT_MODEL` | `claude-opus-5` | Any current Claude model id. |
| `PR_AGENT_THINKING` | `true` | Adaptive thinking. |
| `PR_AGENT_EMBEDDINGS` | `local` | `local`, `voyage`, `none`. |
| `PR_AGENT_VALIDATE` | empty | `';;'`-separated commands. |
| `PR_AGENT_RETRIEVAL_K` | `8` | Chunks fed to the planner. |
| `GITHUB_API_URL` | github.com | Point at GitHub Enterprise here. |

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

The test suite runs the whole graph end to end against a fake GitHub API and a
scripted chat model, so the routing, rollback, reply and filtering behaviour is
covered without a network or an API key. The web layer is covered through
FastAPI's test client, including the event stream and the concurrent-run guard.

### Layout

```
src/pr_agent/
├── cli.py             # typer commands: run, serve, index, search, comments
├── config.py          # env-backed settings
├── models.py          # PRRef, ReviewThread, Triage, ChangePlan, outcomes
├── github_client.py   # REST + GraphQL; groups comments into threads
├── workspace.py       # sandboxed file access, snapshot/rollback
├── tools.py           # the tools the model calls
├── prompts.py         # one prompt per stage
├── validation.py      # runs your checks
├── git_ops.py         # commit, push with backoff
├── llm.py             # Claude via langchain-anthropic
├── rag/
│   ├── splitter.py    # language-aware chunking with line metadata
│   ├── embeddings.py  # local / voyage / none
│   ├── index.py       # incremental build + persistence
│   └── retriever.py   # code-aware BM25 + RRF fusion
├── graph/
│   ├── state.py       # the state passed between nodes
│   ├── nodes.py       # one function per node
│   └── build.py       # wiring and routing
└── web/
    ├── app.py         # FastAPI routes, including the SSE event stream
    ├── runner.py      # background runs and live event fan-out
    ├── schemas.py     # request and response bodies
    └── static/        # the UI: one HTML, one CSS, one JS file, no build step
```

---

## Limitations

- **Forked PRs.** `--push` pushes to the head branch of your local checkout; if
  the PR comes from a fork you need push rights on that fork.
- **Large repositories.** The first index with local embeddings can take a
  while. Use `--embeddings none` to skip it, or keep `.pr_agent/index` around
  between runs — only changed files are re-embedded.
- **Outdated comments.** A comment anchored to a line that has since moved is
  still attempted; the agent works from the diff hunk and retrieval, which
  usually recovers, but not always.
- **Cross-cutting requests.** Comments asking for a repo-wide refactor are
  handled as one change and are the most likely to be rolled back by
  validation.
