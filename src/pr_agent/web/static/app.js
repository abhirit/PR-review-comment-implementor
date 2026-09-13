/* PR Review Comment Implementor — local UI.
 *
 * No framework and no build step: the server ships this file as-is, so the
 * whole tool installs with pip.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  threads: [],
  selected: new Set(),
  runId: null,
  runStatus: null,
  source: null,
  outcomes: new Map(), // thread id -> rendered card element
  liveThread: null,
  stages: [],
  logLines: 0,
};

/* ------------------------------------------------------------------ utils */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text; // always textContent: never HTML
  return node;
}

function toast(message, isError) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("bad", Boolean(isError));
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { node.hidden = true; }, isError ? 8000 : 4000);
}

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = null;
  try { payload = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    const detail = (payload && payload.detail) || `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function relTime(seconds) {
  if (!seconds) return "";
  const delta = Math.max(0, Date.now() / 1000 - seconds);
  if (delta < 60) return `${Math.round(delta)}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  return `${Math.round(delta / 3600)}h ago`;
}

/* ------------------------------------------------------------------- tabs */

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  $$(".tabpanel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === name));
  if (name === "history") loadHistory();
}
$$(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

/* ----------------------------------------------------------------- config */

async function loadConfig() {
  const repoPath = $("#repo_path").value || ".";
  let config;
  try {
    config = await api(`/api/config?repo_path=${encodeURIComponent(repoPath)}`);
  } catch (err) {
    toast(`Could not read configuration: ${err.message}`, true);
    return;
  }

  const pills = $("#status-pills");
  pills.replaceChildren();

  const add = (label, ok, title) => {
    const pill = el("span", `pill ${ok ? "ok" : "bad"}`, label);
    if (title) pill.title = title;
    pills.appendChild(pill);
  };
  add(config.anthropic_key_set ? "Claude key" : "No Claude key", config.anthropic_key_set,
      config.anthropic_key_set ? "ANTHROPIC_API_KEY is set" : "Set ANTHROPIC_API_KEY and restart");
  add(config.github_token_set ? "GitHub token" : "No GitHub token", config.github_token_set,
      config.github_token_set ? "GITHUB_TOKEN is set" : "Set GITHUB_TOKEN and restart");

  const modelPill = el("span", "pill neutral", config.model);
  modelPill.title = "Model in use";
  pills.appendChild(modelPill);

  if (config.repo_is_git) {
    const branch = el("span", "pill neutral", config.repo_branch || "detached");
    branch.title = "Current branch of the checkout";
    pills.appendChild(branch);
  }

  const hint = $("#repo-hint");
  if (!config.repo_is_git) {
    hint.textContent = "Not a git checkout. The agent can still edit files, but cannot commit.";
  } else if (config.repo_dirty) {
    hint.textContent = `On ${config.repo_branch}, with uncommitted changes. Only files the agent edits are staged.`;
  } else {
    hint.textContent = `On ${config.repo_branch}. Check out the PR branch before running.`;
  }

  if (!$("#validate").value && config.detected_validate_commands.length) {
    $("#detect-checks").textContent =
      `Detect from repository (${config.detected_validate_commands.length} found)`;
  }
  state.config = config;
}

$("#detect-checks").addEventListener("click", async () => {
  await loadConfig();
  const detected = (state.config && state.config.detected_validate_commands) || [];
  if (!detected.length) {
    toast("No checks detected. Enter them manually.", true);
    return;
  }
  $("#validate").value = detected.join("\n");
  toast(`Filled in ${plural(detected.length, "check", "checks")}.`);
});

$("#repo_path").addEventListener("change", loadConfig);

/* ---------------------------------------------------------------- threads */

$("#load-threads").addEventListener("click", async () => {
  const pr = $("#pr").value.trim();
  if (!pr) { toast("Enter a pull request first.", true); return; }

  const button = $("#load-threads");
  button.disabled = true;
  button.textContent = "Loading…";
  try {
    setThreads(await api(`/api/comments?pr=${encodeURIComponent(pr)}`));
    showTab("threads");
    toast(`Loaded ${plural(state.threads.length, "review thread", "review threads")}.`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Load review threads";
  }
});

function setThreads(threads) {
  state.threads = threads;
  state.selected = new Set(threads.map((t) => t.id));
  renderThreads();
}

function plural(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

function renderThreads() {
  const list = $("#threads-list");
  list.replaceChildren();

  if (!state.threads.length) {
    list.appendChild(el("p", "empty", "No review comments on this pull request."));
    $("#thread-actions").hidden = true;
    return;
  }
  $("#thread-actions").hidden = false;
  updateThreadCount();

  for (const thread of state.threads) {
    const row = el("div", "thread");

    const box = el("input");
    box.type = "checkbox";
    box.checked = state.selected.has(thread.id);
    box.addEventListener("change", () => {
      if (box.checked) state.selected.add(thread.id);
      else state.selected.delete(thread.id);
      updateThreadCount();
    });

    const main = el("div", "thread-main");
    const head = el("div", "thread-head");
    head.appendChild(el("span", "thread-author", `@${thread.author}`));
    head.appendChild(el("span", "thread-loc",
      thread.path ? `${thread.path}${thread.line ? `:${thread.line}` : ""}` : "general comment"));
    if (thread.replies) {
      head.appendChild(el("span", "badge neutral", plural(thread.replies, "reply", "replies")));
    }
    if (thread.html_url) {
      const link = el("a", "thread-loc", "view on GitHub");
      link.href = thread.html_url;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      head.appendChild(link);
    }

    main.appendChild(head);
    main.appendChild(el("div", "thread-body", thread.body));
    row.appendChild(box);
    row.appendChild(main);
    list.appendChild(row);
  }
}

function updateThreadCount() {
  $("#thread-count").textContent =
    `${state.selected.size} of ${state.threads.length} selected`;
}

$("#select-all").addEventListener("click", () => {
  state.selected = new Set(state.threads.map((t) => t.id));
  renderThreads();
});
$("#select-none").addEventListener("click", () => {
  state.selected = new Set();
  renderThreads();
});

/* -------------------------------------------------------------- run setup */

function syncModeOptions() {
  const mode = $("input[name=mode]:checked").value;
  $("#publish-options").hidden = mode !== "full";
}
$$("input[name=mode]").forEach((r) => r.addEventListener("change", syncModeOptions));
syncModeOptions();

function buildRunRequest() {
  const mode = $("input[name=mode]:checked").value;
  const validateCommands = $("#validate").value
    .split("\n").map((line) => line.trim()).filter(Boolean);

  // Only send ids when the user has actually narrowed the set; an empty list
  // means "every thread" to the backend.
  const allSelected =
    !state.threads.length || state.selected.size === state.threads.length;

  return {
    pr: $("#pr").value.trim(),
    repo_path: $("#repo_path").value.trim() || ".",
    dry_run: mode === "dry",
    commit: mode !== "dry",
    push: mode === "full" && $("#push").checked,
    reply: mode === "full" && $("#reply").checked,
    resolve: mode === "full" && $("#resolve").checked,
    comment_ids: allSelected ? [] : Array.from(state.selected),
    ignore_authors: $("#ignore_authors").value.split(",").map((s) => s.trim()).filter(Boolean),
    self_login: $("#self_login").value.trim() || null,
    include_review_bodies: $("#include_review_bodies").checked,
    model: $("#model").value.trim() || null,
    embeddings: $("#embeddings").value || null,
    validate_commands: validateCommands,
    max_fix_attempts: Number($("#max_fix_attempts").value),
  };
}

$("#run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const request = buildRunRequest();
  if (!request.pr) { toast("Enter a pull request first.", true); return; }

  if (!request.validate_commands.length && !request.dry_run) {
    const proceed = confirm(
      "No verification checks are configured, so the agent's changes will not be " +
      "tested before they are kept.\n\nStart anyway?"
    );
    if (!proceed) return;
  }
  if (request.push || request.reply || request.resolve) {
    const actions = [
      request.push && "push a commit to the PR branch",
      request.reply && "post replies on GitHub",
      request.resolve && "resolve review threads",
    ].filter(Boolean).join(", ");
    if (!confirm(`This run will ${actions}.\n\nContinue?`)) return;
  }

  try {
    const { run_id: runId } = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    startWatching(runId, request);
  } catch (err) {
    toast(err.message, true);
  }
});

$("#cancel-run").addEventListener("click", async () => {
  if (!state.runId) return;
  try {
    const result = await api(`/api/runs/${state.runId}/cancel`, { method: "POST" });
    toast(result.detail);
  } catch (err) {
    toast(err.message, true);
  }
});

/* ------------------------------------------------------------- run stream */

const STAGES = [
  ["load_pr", "Load PR"],
  ["index_repo", "Index"],
  ["triage", "Triage"],
  ["retrieve", "Retrieve"],
  ["plan", "Plan"],
  ["implement", "Implement"],
  ["validate", "Validate"],
  ["finalize", "Finalise"],
];

function startWatching(runId, request) {
  if (state.source) state.source.close();

  state.runId = runId;
  state.outcomes = new Map();
  state.liveThread = null;
  state.logLines = 0;

  $("#run-empty").hidden = true;
  $("#run-view").hidden = false;
  $("#outcomes").replaceChildren();
  $("#run-banner").replaceChildren();
  $("#log").replaceChildren();
  $("#run-title").textContent = request ? request.pr : "Run";
  $("#run-meta").textContent = request
    ? [
        request.dry_run ? "dry run" : "implementing",
        request.push ? "push" : null,
        request.reply ? "reply" : null,
        request.resolve ? "resolve" : null,
        request.validate_commands.length
          ? `${request.validate_commands.length} check(s)`
          : "no checks",
      ].filter(Boolean).join(" · ")
    : "";

  renderPipeline(null);
  setRunStatus("starting");
  $("#cancel-run").hidden = false;
  $("#start-run").disabled = true;
  showTab("run");

  state.source = new EventSource(`/api/runs/${runId}/events`);
  state.source.onmessage = (message) => handleEvent(JSON.parse(message.data));
  state.source.onerror = () => {
    // The stream closes normally when a run ends; only surface a real drop.
    if (state.runStatus === "running" || state.runStatus === "starting") {
      toast("Lost the connection to the run. Reopen it from History to catch up.", true);
    }
    state.source.close();
  };
}

function setRunStatus(status) {
  state.runStatus = status;
  const badge = $("#run-status");
  badge.textContent = status;
  badge.className = `badge ${status}`;
  const finished = ["done", "failed", "cancelled"].includes(status);
  $("#cancel-run").hidden = finished;
  $("#start-run").disabled = !finished && Boolean(state.runId);
  const runBadge = $("#run-badge");
  runBadge.hidden = !(status === "running" || status === "starting");
  runBadge.textContent = "•";
}

function renderPipeline(activeNode) {
  const list = $("#pipeline");
  list.replaceChildren();
  const activeIndex = STAGES.findIndex(([node]) => node === activeNode);
  STAGES.forEach(([node, label], index) => {
    const item = el("li", "", label);
    if (activeIndex >= 0 && index < activeIndex) item.classList.add("is-done");
    if (node === activeNode) item.classList.add("is-active");
    list.appendChild(item);
  });
}

function appendLog(level, message) {
  const log = $("#log");
  const line = el("div", `lvl-${level}`, message);
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
  state.logLines += 1;
  $("#log-count").textContent = `(${state.logLines})`;
}

function handleEvent(event) {
  switch (event.type) {
    case "status":
      setRunStatus(event.status);
      break;

    case "log":
      appendLog(event.level, event.message);
      break;

    case "node":
      renderPipeline(event.node);
      break;

    case "pr":
      if (event.threads) setThreads(event.threads);
      banner("info", `${plural(event.threads.length, "review thread", "review threads")} to consider.`);
      break;

    case "index":
      banner("info", `Index: ${event.summary}`);
      break;

    case "thread_start":
      state.liveThread = event.thread;
      liveCard(event.thread);
      break;

    case "triage":
      updateLive((body) => {
        field(body, "Triage", `${event.action} — ${event.reason}`);
        if (event.queries && event.queries.length) {
          field(body, "Retrieval queries", event.queries.join(" · "));
        }
      }, event.action);
      break;

    case "plan":
      updateLive((body) => field(body, "Plan", event.plan));
      break;

    case "implement":
      updateLive((body) => {
        if (event.summary) field(body, event.phase === "fix" ? "Fix" : "Implementation", event.summary);
        if (event.files && event.files.length) chips(body, "Files changed", event.files);
        if (event.error) field(body, "Problem", event.error);
      });
      break;

    case "validation":
      updateLive((body) => {
        const label = event.ok ? "Checks passed" : `Checks failed: ${event.command}`;
        field(body, "Validation", label);
        if (!event.ok && event.output) output(body, event.output);
      });
      break;

    case "outcome":
      finishCard(event.outcome);
      break;

    case "finalize": {
      const parts = [];
      if (event.commit_sha) parts.push(`committed ${event.commit_sha.slice(0, 10)}`);
      if (event.pushed) parts.push("pushed");
      if (event.replies_posted) {
        parts.push(`${plural(event.replies_posted, "reply", "replies")} posted`);
      }
      if (parts.length) banner("ok", parts.join(" · "));
      break;
    }

    case "report":
      if (event.report) banner("ok", event.report, true);
      break;

    case "error":
      banner("bad", event.message);
      break;

    case "end":
      setRunStatus(event.status);
      if (state.source) state.source.close();
      loadHistory();
      break;

    default:
      break;
  }
}

/* ------------------------------------------------------------- run render */

function banner(kind, text, isPre) {
  const node = el("div", `banner ${kind}`);
  if (isPre) {
    node.appendChild(el("strong", "", "Report"));
    node.appendChild(el("pre", "", text));
  } else {
    node.textContent = text;
  }
  $("#run-banner").appendChild(node);
}

function field(parent, label, value) {
  const wrap = el("div");
  wrap.appendChild(el("div", "field-label", label));
  wrap.appendChild(el("div", "field-value", value));
  parent.appendChild(wrap);
}

function chips(parent, label, items) {
  const wrap = el("div");
  wrap.appendChild(el("div", "field-label", label));
  const row = el("div", "chips");
  items.forEach((item) => row.appendChild(el("span", "chip", item)));
  wrap.appendChild(row);
  parent.appendChild(wrap);
}

function output(parent, text) {
  parent.appendChild(el("pre", "output", text));
}

function renderDiff(parent, diffText) {
  const wrap = el("div");
  wrap.appendChild(el("div", "field-label", "Diff"));
  const box = el("div", "diff");
  for (const line of diffText.split("\n")) {
    let className = "";
    if (line.startsWith("+++") || line.startsWith("---")) className = "meta";
    else if (line.startsWith("@@")) className = "hunk";
    else if (line.startsWith("+")) className = "add";
    else if (line.startsWith("-")) className = "del";
    box.appendChild(el("div", className, line || " "));
  }
  wrap.appendChild(box);
  parent.appendChild(wrap);
}

function liveCard(thread) {
  const card = el("div", "outcome is-live");
  const head = el("div", "outcome-head");
  head.appendChild(el("span", "badge running", "working"));
  head.appendChild(el("span", "thread-author", `@${thread.author}`));
  head.appendChild(el("span", "thread-loc",
    thread.path ? `${thread.path}${thread.line ? `:${thread.line}` : ""}` : "general"));
  const body = el("div", "outcome-body");
  field(body, "Comment", thread.body);

  card.appendChild(head);
  card.appendChild(body);
  $("#outcomes").appendChild(card);
  state.outcomes.set(thread.id, card);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function updateLive(build, badgeText) {
  if (!state.liveThread) return;
  const card = state.outcomes.get(state.liveThread.id);
  if (!card) return;
  build(card.querySelector(".outcome-body"));
  if (badgeText) {
    const badge = card.querySelector(".badge");
    badge.textContent = badgeText;
    badge.className = `badge ${badgeText}`;
  }
}

function finishCard(outcome) {
  const card = state.outcomes.get(outcome.thread_id);
  if (!card) return;
  card.classList.remove("is-live");

  const implemented = outcome.files_changed && outcome.files_changed.length;
  const badge = card.querySelector(".badge");
  if (outcome.error) {
    badge.textContent = "failed";
    badge.className = "badge failed";
  } else if (implemented) {
    badge.textContent = "implemented";
    badge.className = "badge ok";
  } else {
    badge.textContent = outcome.action;
    badge.className = `badge ${outcome.action}`;
  }

  const body = card.querySelector(".outcome-body");
  if (outcome.diff) renderDiff(body, outcome.diff);
  if (outcome.attempts) field(body, "Fix attempts", String(outcome.attempts));
  if (outcome.error) field(body, "Error", outcome.error);
  if (outcome.reply) field(body, "Reply", outcome.reply);
  state.liveThread = null;
}

/* ---------------------------------------------------------------- history */

async function loadHistory() {
  let runs;
  try { runs = await api("/api/runs"); } catch { return; }

  const list = $("#history-list");
  list.replaceChildren();
  if (!runs.length) {
    list.appendChild(el("p", "empty", "No runs in this session yet."));
    return;
  }
  for (const run of runs) {
    const row = el("div", "history-row");
    row.appendChild(el("span", `badge ${run.status}`, run.status));
    row.appendChild(el("strong", "", run.pr));
    row.appendChild(el("span", "muted", `${run.implemented} implemented`));
    row.appendChild(el("span", "muted", relTime(run.created_at)));
    row.addEventListener("click", () => replayRun(run.id));
    list.appendChild(row);
  }
}
$("#refresh-history").addEventListener("click", loadHistory);

async function replayRun(runId) {
  let snapshot;
  try { snapshot = await api(`/api/runs/${runId}`); } catch (err) {
    toast(err.message, true);
    return;
  }
  startWatching(runId, snapshot.request);
  // Re-running the stored events rebuilds the view exactly as it was live.
  if (["done", "failed", "cancelled"].includes(snapshot.status) && state.source) {
    state.source.close();
    snapshot.events.forEach(handleEvent);
    setRunStatus(snapshot.status);
  }
  showTab("run");
}

/* ----------------------------------------------------------------- search */

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("#search-q").value.trim();
  const repoPath = $("#repo_path").value.trim() || ".";
  const results = $("#search-results");
  results.replaceChildren(el("p", "empty", "Searching…"));

  try {
    const hits = await api(
      `/api/search?q=${encodeURIComponent(query)}&repo_path=${encodeURIComponent(repoPath)}`
    );
    results.replaceChildren();
    if (!hits.length) {
      results.appendChild(el("p", "empty", "Nothing matched."));
      return;
    }
    for (const hit of hits) {
      const card = el("div", "hit");
      const head = el("div", "hit-head");
      head.appendChild(el("strong", "", `${hit.path}:${hit.start_line}-${hit.end_line}`));
      head.appendChild(el("span", "badge info", hit.source));
      head.appendChild(el("span", "muted", hit.score.toFixed(4)));
      card.appendChild(head);
      card.appendChild(el("pre", "", hit.content));
      results.appendChild(card);
    }
  } catch (err) {
    results.replaceChildren(el("p", "empty", err.message));
  }
});

/* ------------------------------------------------------------------- boot */

loadConfig();
loadHistory();
