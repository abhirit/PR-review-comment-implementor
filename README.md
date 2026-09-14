# PR Review Comment Implementor

An agent that reads the review comments on a GitHub pull request, works out
which ones ask for a code change, and implements them in your checkout —
grounded by retrieval over the repository, verified by your own test suite, and
rolled back when it cannot get a change to pass.

Built with **LangGraph** (the control flow), **LangChain** (model, tools and
text splitting), **Gemini** (`gemini-3.8-flash` by default, with Claude as an
alternative provider) and a **hybrid RAG** index over the repo.

```
pr-agent serve                                    # web UI on localhost:8765
pr-agent run "owner/repo#123" --auto-validate     # or drive it from the terminal
```

---

## What it actually does

First, once: the checkout is put on the PR's head branch. Anything
uncommitted is stashed before the switch and handed back at the end, so the
agent can borrow a checkout you were in the middle of using.

Then, for each review thread on the PR:

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
and resolve the ones it implemented — and, if you asked for it, put the
checkout back on the branch you were on with your stash popped.

### The graph

```
  load_pr ──▶ prepare_branch ──▶ index_repo ──▶ next_thread
                                                    │
                              (queue empty) ────────┼──▶ finalize ──▶ END
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

## The branch

The agent must edit the code the reviewer was actually looking at, so a run
begins by checking out the pull request's head branch:

1. If the working tree is dirty, it is stashed first — untracked files
   included, ignored files left alone.
2. The branch is checked out. If it is not in the checkout yet it is fetched
   from `origin`; a pull request opened from a fork is fetched through
   `pull/<number>/head`, which GitHub publishes on the base repository.
3. If the branch cannot be reached at all, **the run stops**. Implementing
   review comments against the wrong branch produces plausible-looking changes
   to code the reviewer never saw, which is worse than not running.

At the end, `--restore-branch` (the *Go back to your branch afterwards* box in
the UI, on by default for *Implement and publish*) returns the checkout to the
branch you started on and pops the stash. The stash entry is found by its
commit sha rather than by `stash@{0}`, so an entry someone else pushed in the
meantime is never mistaken for yours.

One exception: if a commit was made but the push failed, the checkout stays on
the PR branch. The work exists only there, and walking away from it would hide
that.

Pass `--no-checkout` (or clear the box) to manage the branch yourself.

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
- **Chat** — ask the agent about the run once it has finished: why a comment
  was skipped, what a diff does, or for another change on top. It has the whole
  run in context and the same sandboxed file tools, so "also rename that
  variable" is a request it can carry out. Changes made in chat are written to
  the checkout but not committed, and there is a toggle to make a message
  read-only when you only want an answer.
- **Retrieval** — query the index directly to check whether a comment's subject
  is findable before you run.
- **History** — replay any earlier run from this server session.

Progress streams over server-sent events, and every event is retained, so a
browser that connects late — or reconnects — replays the whole run rather than
showing a blank page.

It is a local tool: it binds to `127.0.0.1`, ships no authentication, and edits
files and runs your validation commands. Do not expose it to a network you do
not control. Two runs against one checkout are refused, since they would edit
the same files, and a run's chat opens only once that run has finished, for the
same reason.

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

**BM25 alone is the default.** Splitting compound identifiers recovers most of
the vocabulary overlap embeddings are usually brought in to fix, and whatever
retrieval misses the model can still reach with `search_repository` and
`read_file` during the implement step. Set `--embeddings google`,
`--embeddings local` or `--embeddings voyage` to add the dense half, at the
cost of a model download or extra API calls plus a vector store to keep in
sync.

Other details that matter in practice:

- **Language-aware chunking.** Files split on function and class boundaries
  using the splitter for their language, not at arbitrary character offsets.
- **Line numbers on every chunk**, so a retrieved excerpt is directly
  actionable.
- **Incremental embedding.** Chunking is redone each run (it is cheap);
  embedding is not, so only files whose content hash changed are re-embedded.
  Changing the embedding model or chunk size invalidates the store and forces a
  clean rebuild.
- **The index follows the agent's own edits.** Each implemented comment
  re-chunks the files it changed, so a later comment is never planned against
  code that has already been replaced.
- **The excerpt is never trusted for editing.** The model is told to
  `read_file` before it edits, because a retrieved chunk can be stale.

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

# optional: run on Claude instead of Gemini
pip install -e ".[anthropic]"

# optional: the web UI
pip install -e ".[web]"
```

Then copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY`
([AI Studio](https://aistudio.google.com/apikey)) and `GITHUB_TOKEN`.

> **Running on Claude instead.** Install the `[anthropic]` extra, then set
> `PR_AGENT_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. Everything else is
> unchanged: both providers are driven through the same LangChain interface.

> The embedding backend is configured separately from the chat model —
> Anthropic serves no embeddings endpoint at all, and Gemini's is a separate
> model and API call.

---

## Use

```bash
# See what the agent would consider, and act on nothing.
pr-agent comments "owner/repo#123"

# Dry run: edits nothing permanently, prints the plan and the replies it would post.
pr-agent run "owner/repo#123" --repo ./checkout --dry-run

# The normal run: implement, verify with your checks, commit locally.
pr-agent run "owner/repo#123" --repo ./checkout --auto-validate

# Full loop: implement, verify, commit, push to the PR branch, reply, resolve,
# then put the checkout back on the branch you were on.
pr-agent run "owner/repo#123" --repo ./checkout \
    --validate "ruff check ." --validate "pytest -q" \
    --push --reply --resolve --restore-branch

# One specific comment only.
pr-agent run "owner/repo#123" --comment-id 1234567890

# Inspect retrieval quality on its own.
pr-agent index --repo ./checkout
pr-agent search "where are review threads grouped" --repo ./checkout
```

The agent checks the PR branch out itself, stashing anything uncommitted
first, so there is no need to prepare the checkout by hand. If you would rather
do it yourself, pass `--no-checkout`.

### Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--repo`, `-r` | `.` | Local checkout to edit. |
| `--dry-run` | off | Do everything except commit, push and post. |
| `--no-commit` | commits | Leave changes in the working tree. |
| `--push` | off | Push the commit to the PR's head branch. |
| `--checkout` / `--no-checkout` | on | Check the PR branch out first, stashing uncommitted work. |
| `--restore-branch` | off | Afterwards, return to the original branch and unstash. |
| `--reply` | off | Post a reply on each thread that was handled. |
| `--resolve` | off | Also resolve implemented threads (implies `--reply`). |
| `--validate` | none | A check to run after each change. Repeatable. |
| `--auto-validate` | off | Detect checks from the repo (ruff, pytest, npm, go). |
| `--max-fix-attempts` | 3 | Retries after a failing check before rolling back. |
| `--provider` | `google` | `google` (Gemini) or `anthropic` (Claude). |
| `--embeddings` | `none` | `google`, `local`, `voyage` or `none`. |
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
- **Your uncommitted work is never edited over.** It goes to the stash before
  the branch switch, and comes back by sha, not by position.
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
| `PR_AGENT_PROVIDER` | `google` | `google` (Gemini) or `anthropic` (Claude). |
| `PR_AGENT_MODEL` | per provider | `gemini-3.8-flash` / `claude-opus-5`. Pro-tier Gemini models need a paid API plan. |
| `PR_AGENT_THINKING` | `true` | The model decides how much to think. Gemini 3 models always think. |
| `PR_AGENT_EMBEDDINGS` | `none` | `google`, `local`, `voyage`, `none`. |
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
├── chat.py            # the follow-up conversation about a finished run
├── git_ops.py         # commit, push, branch switching and stashing
├── llm.py             # Gemini or Claude, one LangChain interface
├── rag/
│   ├── splitter.py    # language-aware chunking with line metadata
│   ├── embeddings.py  # google / local / voyage / none
│   ├── index.py       # incremental build + persistence
│   └── retriever.py   # code-aware BM25 + RRF fusion
├── graph/
│   ├── state.py       # the state passed between nodes
│   ├── nodes.py       # one function per node
│   └── build.py       # wiring and routing
└── web/
    ├── app.py         # FastAPI routes: the SSE event stream and the chat
    ├── runner.py      # background runs and live event fan-out
    ├── schemas.py     # request and response bodies
    └── static/        # the UI: one HTML, one CSS, one JS file, no build step
```

---

## Limitations

- **Forked PRs.** The head branch is fetched through `pull/<number>/head`, so
  the checkout works, but `--push` still needs push rights on the fork.
- **Large repositories.** BM25-only indexing is fast, but if you turn
  embeddings on, the first index can take a while. Keep `.pr_agent/index`
  around between runs — only changed files are re-embedded.
- **Outdated comments.** A comment anchored to a line that has since moved is
  still attempted; the agent works from the diff hunk and retrieval, which
  usually recovers, but not always.
- **Cross-cutting requests.** Comments asking for a repo-wide refactor are
  handled as one change and are the most likely to be rolled back by
  validation.
