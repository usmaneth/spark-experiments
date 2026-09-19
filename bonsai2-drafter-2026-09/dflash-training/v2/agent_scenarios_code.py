#!/usr/bin/env python3
"""agent_scenarios_code.py - fake tool results and tasks for the code agents (fern, pixel, anvil).

Every scenario is a function gen(rng, n) -> dict with keys task, call, result, kind, tag.
    task    the user (planner) task text
    call    {"name": tool name, "arguments": {...}} - the prior tool call
    result  the fake tool result text; n scales the number of repeated units
    kind    "call" when the natural next answer is another tool call, "report" when it is a report
Calling gen twice with an rng seeded the same way and a different n gives the same task and call
and a result of a different size. All text is original and ASCII only.
"""
import random

# --------------------------------------------------------------------------- shared vocab
PY_SERVICES = ["auth", "billing", "ingest", "notifier", "search", "export", "gateway", "ledger", "scheduler", "quota"]
PY_LIBS = ["common", "storage", "metrics", "queue", "cache"]
PY_FILES = ["handlers.py", "models.py", "tasks.py", "client.py", "config.py", "routes.py", "worker.py", "schema.py",
            "cli.py", "retry.py", "parsers.py", "emitters.py", "validators.py", "hooks.py"]
PY_FUNCS = ["parse_invoice", "build_retry", "send_batch", "load_config", "normalize_email", "chunk_rows", "sign_payload",
            "refresh_token", "compute_total", "apply_discount", "enqueue", "dequeue", "validate_schema", "render_template",
            "fetch_page", "merge_windows", "rotate_key", "flush_buffer", "resolve_tenant", "coerce_bool"]
PY_TESTS = ["test_retry_gives_up", "test_parse_invoice_rejects_negative", "test_chunk_rows_last_chunk", "test_config_env_override",
            "test_normalize_email_lowercases", "test_sign_payload_stable", "test_refresh_token_expiry", "test_compute_total_rounding",
            "test_enqueue_duplicate", "test_validate_schema_missing_field", "test_render_template_escapes", "test_fetch_page_timeout",
            "test_merge_windows_overlap", "test_rotate_key_keeps_old", "test_flush_buffer_empty", "test_resolve_tenant_unknown",
            "test_coerce_bool_strings", "test_apply_discount_cap", "test_dequeue_order", "test_send_batch_partial"]
MARKERS = ["TODO", "FIXME", "XXX", "HACK"]
MARKER_TEXT = ["handle the empty page case", "remove after the 3.2 migration", "this retries forever on 5xx",
               "the timeout is hard coded", "move this into libs/common", "cache the compiled regex",
               "the tenant id can be None here", "drop the legacy branch once the flag is gone",
               "the batch size should come from config", "this swallows the original exception",
               "replace the print with the logger", "the lock is held across the network call",
               "duplicate of the helper in storage", "does not handle a trailing slash", "validate the currency code",
               "the sort is unstable on ties", "this should be an async call", "add a unit test for the leap day case"]
RULES = [("F401", "`{mod}` imported but unused", "[*] "), ("E501", "Line too long ({n} > 100)", ""),
         ("F841", "Local variable `{var}` is assigned to but never used", ""), ("E722", "Do not use bare `except`", ""),
         ("B008", "Do not perform function call `{call}` in argument defaults", ""), ("UP035", "Import from `collections.abc` instead: `Iterable`", "[*] "),
         ("E402", "Module level import not at top of file", ""), ("I001", "Import block is un-sorted or un-formatted", "[*] "),
         ("B006", "Do not use mutable data structures for argument defaults", ""), ("SIM108", "Use ternary operator instead of `if`-`else`-block", "[*] "),
         ("E731", "Do not assign a `lambda` expression, use a `def`", ""), ("C401", "Unnecessary generator (rewrite as a `set` comprehension)", "[*] "),
         ("N806", "Variable `{var}` in function should be lowercase", ""), ("RET504", "Unnecessary assignment to `{var}` before `return` statement", "[*] ")]
MODS = ["os", "sys", "json", "time", "typing.Optional", "re", "pathlib.Path", "datetime", "logging", "itertools"]
VARS = ["result", "tmp", "resp", "count", "payload", "rows", "cfg", "total", "buf", "token"]
NAMES = ["alice", "ben", "chloe", "dan", "elena", "farid", "grace", "hiro", "ines", "jamal", "kira", "leo", "maya", "noor"]


def pyfile(rng):
    if rng.random() < 0.7:
        return f"services/{rng.choice(PY_SERVICES)}/{rng.choice(PY_FILES)}"
    return f"libs/{rng.choice(PY_LIBS)}/{rng.choice(PY_FILES)}"


def sha(rng):
    return "".join(rng.choice("0123456789abcdef") for _ in range(7))


def pick(rng, xs):
    return rng.choice(xs)


def num(rng, a, b):
    return rng.randint(a, b)


def sample(rng, xs, k):
    return rng.sample(xs, min(k, len(xs)))


# --------------------------------------------------------------------------- FERN (python)
def fern_grep_markers(rng, n):
    marks = sample(rng, MARKERS, num(rng, 2, 3))
    scope = pick(rng, ["services/", "libs/", "services/ tools/", "libs/ services/"])
    pat = "|".join(marks)
    lines = []
    for _ in range(max(6, n)):
        f = pyfile(rng)
        lines.append(f"{f}:{num(rng, 8, 420)}:    # {pick(rng, marks)}: {pick(rng, MARKER_TEXT)}")
    lines.sort()
    variant = rng.randrange(4)
    if variant == 0:
        task = (f"Find every {' and '.join(marks)} marker under {scope.strip()} and write a cleanup plan. Group the items by file, "
                f"give each item a size of small, medium or large, and sort the groups by total size, largest first. Use a markdown table.")
        kind = "report"
    elif variant == 1:
        task = (f"List the {' / '.join(marks)} markers in {scope.strip()}. Then open the file with the most markers and quote the "
                f"surrounding function for each one, so the planner can decide which ones to keep.")
        kind = "call"
    elif variant == 2:
        task = (f"How many {pick(rng, marks)} markers are left under {scope.strip()}, and which service owns the most? "
                f"Answer with a count per directory and a one line recommendation.")
        kind = "report"
    else:
        task = (f"Remove the stale {pick(rng, marks)} comments under {scope.strip()} that talk about a migration or a flag. "
                f"Start with the file that has the most of them; read it before you change it.")
        kind = "call"
    call = {"name": "grep_repo", "arguments": {"pattern": pat, "path_glob": scope.split()[0] + "**/*.py", "max_matches": 200}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_grep_markers"}


def fern_grep_symbol(rng, n):
    old, new = pick(rng, [("legacy_retry", "build_retry"), ("get_conn", "connection"), ("to_json", "dumps"),
                          ("log_warn", "logger.warning"), ("read_cfg", "load_config"), ("md5_sign", "sign_payload"),
                          ("split_chunks", "chunk_rows"), ("Tenant.find", "resolve_tenant")])
    lines = []
    for _ in range(max(5, n)):
        f = pyfile(rng)
        ln = num(rng, 10, 380)
        ctx = pick(rng, [f"    result = {old}(payload, retries={num(rng, 1, 5)})", f"    return {old}(row)", f"from libs.common.legacy import {old.split('.')[0]}",
                         f"        {pick(rng, VARS)} = {old}(self.client, timeout={num(rng, 5, 60)})", f"    if {old}(cfg) is None:",
                         f"    items = [{old}(x) for x in rows]", f"    with {old}(url) as conn:"])
        lines.append(f"{f}:{ln}:{ctx}")
    lines.sort()
    v = rng.randrange(3)
    if v == 0:
        task = (f"Replace every call of `{old}` with `{new}` in services/ and libs/. `{new}` has the same signature. "
                f"Find the call sites first, then change one file at a time and run the tests of that service.")
        kind = "call"
    elif v == 1:
        task = (f"Report every use of `{old}` in the monorepo: file, line and the enclosing service. Say which services would need a "
                f"code change to move to `{new}`, and estimate the size of each change as small, medium or large.")
        kind = "report"
    else:
        task = (f"`{old}` is deprecated. Find its callers, open the one in libs/ first, and check whether it is safe to switch it to "
                f"`{new}` without a behavior change. Quote the caller before you change anything.")
        kind = "call"
    call = {"name": "grep_repo", "arguments": {"pattern": old.replace(".", r"\.") + r"\(", "path_glob": "**/*.py"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_grep_symbol"}


def _pytest_block(rng, testname, path):
    kinds = rng.randrange(5)
    fn = pick(rng, PY_FUNCS)
    lines = ["_" * 20 + f" {testname} " + "_" * 20, ""]
    if kinds == 0:
        a, b = num(rng, 1, 9), num(rng, 1, 9)
        lines += [f"    def {testname}():", f"        calls = _run_with_failures({a})", f">       assert calls == {a}", f"E       assert {a + 1} == {a}", ""]
        lines += [f"{path}:{num(rng, 10, 120)}: AssertionError"]
    elif kinds == 1:
        lines += [f"    def {testname}():", f"        out = {fn}({pick(rng, ['payload', 'rows', 'cfg', 'text'])})", f">       assert out == {pick(rng, ['expected', '[1, 2, 3]', 'Decimal(\"10.50\")', '\"a@b.io\"'])}",
                  f"E       AssertionError: assert {pick(rng, ['None', '[1, 2]', 'Decimal(\"10.5\")', '\"A@b.io\"'])} == {pick(rng, ['expected', '[1, 2, 3]', 'Decimal(\"10.50\")', '\"a@b.io\"'])}", "",
                  f"{path}:{num(rng, 10, 120)}: AssertionError"]
    elif kinds == 2:
        src = pyfile(rng)
        lines += [f"    def {testname}():", f">       {fn}({pick(rng, ['{}', 'None', '\"\"', '[]'])})", "", f"{path}:{num(rng, 10, 120)}: ", "_ " * 30, "",
                  f"    def {fn}({pick(rng, ['payload', 'rows', 'cfg'])}):", f">       return {pick(rng, ['payload[\"id\"]', 'rows[0]', 'cfg.timeout', 'int(payload)'])}",
                  f"E       {pick(rng, ['KeyError: ' + chr(39) + 'id' + chr(39), 'IndexError: list index out of range', 'AttributeError: ' + chr(39) + 'NoneType' + chr(39) + ' object has no attribute ' + chr(39) + 'timeout' + chr(39), 'TypeError: int() argument must be a string, a bytes-like object or a real number, not ' + chr(39) + 'NoneType' + chr(39)])}", "",
                  f"{src}:{num(rng, 10, 300)}: {pick(rng, ['KeyError', 'IndexError', 'AttributeError', 'TypeError'])}"]
    elif kinds == 3:
        lines += [f"    def {testname}(tmp_path):", f"        cfg = load_config(tmp_path / 'settings.yaml', env={{'{pick(rng, ['TIMEOUT', 'BATCH_SIZE', 'LOG_LEVEL'])}': '{num(rng, 1, 99)}'}})",
                  f">       assert cfg.{pick(rng, ['timeout', 'batch_size', 'log_level'])} == {num(rng, 1, 99)}", f"E       assert {num(rng, 1, 99)} == {num(rng, 1, 99)}", f"E        +  where {num(rng, 1, 99)} = Config(...).{pick(rng, ['timeout', 'batch_size'])}", "",
                  f"{path}:{num(rng, 10, 120)}: AssertionError"]
    else:
        lines += [f"    async def {testname}():", f"        async with timeout({num(rng, 1, 5)}):", f">           await {fn}(client)", "",
                  f"{path}:{num(rng, 10, 120)}: ", "_ " * 30, "", f"E       asyncio.exceptions.TimeoutError", "",
                  f"{pyfile(rng)}:{num(rng, 10, 300)}: TimeoutError"]
    return lines


def fern_pytest(rng, n):
    svc = pick(rng, PY_SERVICES)
    nfail = min(3, max(1, n // 4))
    tests = sample(rng, PY_TESTS, nfail)
    path = f"tests/{svc}/test_{pick(rng, ['handlers', 'tasks', 'client', 'config', 'models'])}.py"
    total = num(rng, 18, 90)
    dots = list("." * (total - nfail))
    for _ in range(nfail):
        dots.insert(num(rng, 0, len(dots)), "F")
    lines = ["============================= test session starts ==============================",
             f"platform linux -- Python 3.12.{num(rng, 3, 9)}, pytest-8.{num(rng, 2, 4)}.{num(rng, 0, 3)}, pluggy-1.{num(rng, 4, 6)}.0",
             f"rootdir: /work/northwind", f"configfile: pyproject.toml", f"collected {total} items", "",
             f"{path} {''.join(dots)}" + f"{'':>{max(1, 60 - total)}}[100%]", "", "=================================== FAILURES ==================================="]
    for t in tests:
        lines += _pytest_block(rng, t, path)
        lines.append("")
    lines += ["=========================== short test summary info ============================"]
    for t in tests:
        lines.append(f"FAILED {path}::{t} - {pick(rng, ['AssertionError', 'KeyError', 'TypeError', 'TimeoutError', 'AttributeError'])}")
    lines.append(f"========================= {nfail} failed, {total - nfail} passed in {num(rng, 1, 30)}.{num(rng, 10, 99)}s =========================")
    # optional source excerpt
    if n > 8:
        src = pyfile(rng)
        fn = pick(rng, PY_FUNCS)
        lines += ["", f"$ sed -n {num(rng, 20, 60)},{num(rng, 61, 90)}p {src}",
                  f"def {fn}({pick(rng, ['payload, retries=3', 'rows, size=100', 'path, env=None', 'client, timeout=30'])}):",
                  f"    {pick(rng, ['calls = 0', 'out = []', 'cfg = _defaults()', 'deadline = time.monotonic() + timeout'])}",
                  f"    {pick(rng, ['while calls <= retries:', 'for i in range(0, len(rows), size):', 'data = yaml.safe_load(open(path))', 'while time.monotonic() < deadline:'])}",
                  f"        {pick(rng, ['calls += 1', 'out.append(rows[i:i + size])', 'cfg.update(data or {})', 'resp = client.get()'])}",
                  f"        {pick(rng, ['try:', 'if not out[-1]:', 'if env:', 'if resp.ok:'])}",
                  f"            {pick(rng, ['_send(payload)', 'break', 'cfg.update({k.lower(): v for k, v in env.items()})', 'return resp.json()'])}",
                  f"        {pick(rng, ['except OSError:', '', '', 'time.sleep(1)'])}",
                  f"            {pick(rng, ['time.sleep(1)', 'pass', 'continue', 'raise'])}",
                  f"    return {pick(rng, ['calls', 'out', 'Config(**cfg)', 'None'])}"]
    v = rng.randrange(3)
    if v == 0:
        task = (f"The test run for the {svc} service has {nfail} failure{'s' if nfail > 1 else ''}. Find the root cause of each one and fix the code under "
                f"services/{svc}/ or libs/. Do not change the tests. Run the same test file again after the fix.")
        kind = "call"
    elif v == 1:
        task = (f"Explain each failure in the {svc} test output in two sentences: what the test expects, what the code does. Then propose the "
                f"fix as a unified diff per file. Do not apply anything.")
        kind = "report"
    else:
        task = (f"Triage the failing tests in tests/{svc}/. For each one say whether the test or the code is wrong, with the line that shows it. "
                f"Then open the source file behind the first failure and quote the function.")
        kind = "call"
    call = {"name": "exec_command", "arguments": {"command": f"python3 -m pytest tests/{svc}/ -q -x --tb=short 2>&1 | tail -{num(rng, 60, 120)}" if rng.random() < 0.5 else f"python3 -m pytest {path} -q 2>&1"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_pytest"}


def fern_ruff(rng, n):
    scope = pick(rng, ["services/" + pick(rng, PY_SERVICES) + "/", "libs/", "tools/", "services/"])
    files = [pyfile(rng) for _ in range(num(rng, 3, 6))]
    lines = []
    for _ in range(max(6, n)):
        f = pick(rng, files)
        code, msg, fix = pick(rng, RULES)
        msg = msg.format(mod=pick(rng, MODS), n=num(rng, 101, 140), var=pick(rng, VARS), call=pick(rng, ['dict()', 'Depends()', 'now()']))
        lines.append(f"{f}:{num(rng, 1, 400)}:{num(rng, 1, 80)}: {code} {fix}{msg}")
    lines.sort()
    nfix = sum(1 for l in lines if "[*]" in l)
    lines += [f"Found {len(lines)} errors.", f"[*] {nfix} fixable with the `--fix` option."]
    if n > 10:
        f = pick(rng, files)
        lines += ["", f"$ sed -n 1,14p {f}", "import os", "import json", "import sys", f"from typing import Optional, List",
                  f"from libs.common import logging as log", "", f"{pick(rng, VARS).upper()} = {num(rng, 10, 500)}", "",
                  f"def {pick(rng, PY_FUNCS)}({pick(rng, ['rows', 'payload', 'cfg'])}, {pick(rng, ['opts={}', 'retries=3', 'size=100'])}):",
                  f"    {pick(rng, VARS)} = {pick(rng, ['[]', 'None', '0'])}", "    try:", f"        return _{pick(rng, PY_FUNCS)}({pick(rng, ['rows', 'payload', 'cfg'])})",
                  "    except:", "        return None"]
    v = rng.randrange(3)
    if v == 0:
        task = (f"Resolve the ruff findings under {scope}. Fix the file with the most findings first, by hand, with no behavior change. "
                f"Do not add noqa comments. Run ruff on that file again when you are done.")
        kind = "call"
    elif v == 1:
        task = (f"Group the ruff findings for {scope} by rule code with a count, say which rules are safe to autofix, and list the files "
                f"that need a manual edit. Do not change any file.")
        kind = "report"
    else:
        task = (f"The lint gate for {scope} is red. Open the file with the most findings, quote the lines the findings point at, and prepare "
                f"the fixed file. Keep imports that the tests use.")
        kind = "call"
    call = {"name": "exec_command", "arguments": {"command": f"ruff check {scope} --output-format concise 2>&1 | head -{num(rng, 40, 80)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_ruff"}


def fern_mypy(rng, n):
    scope = pick(rng, ["libs/", "services/" + pick(rng, PY_SERVICES) + "/", "libs/ services/"])
    files = [pyfile(rng) for _ in range(num(rng, 3, 5))]
    errs = ['Incompatible return value type (got "Optional[str]", expected "str")  [return-value]',
            'Argument 1 to "{fn}" has incompatible type "Optional[dict[str, Any]]"; expected "dict[str, Any]"  [arg-type]',
            'Item "None" of "Optional[Config]" has no attribute "timeout"  [union-attr]',
            'Need type annotation for "{var}" (hint: "{var}: list[<type>] = ...")  [var-annotated]',
            'Missing return statement  [return]', 'Incompatible types in assignment (expression has type "int", variable has type "str")  [assignment]',
            'Function is missing a type annotation for one or more arguments  [no-untyped-def]',
            '"{fn}" does not return a value (it only ever returns None)  [func-returns-value]',
            'Unsupported operand types for + ("int" and "str")  [operator]', 'Name "{var}" already defined on line {n}  [no-redef]',
            'Returning Any from function declared to return "int"  [no-any-return]']
    lines = []
    for _ in range(max(5, n)):
        f = pick(rng, files)
        e = pick(rng, errs).format(fn=pick(rng, PY_FUNCS), var=pick(rng, VARS), n=num(rng, 5, 60))
        lines.append(f"{f}:{num(rng, 5, 400)}: error: {e}")
    lines.sort()
    lines.append(f"Found {len(lines)} errors in {len(set(l.split(':')[0] for l in lines))} files (checked {num(rng, 40, 160)} source files)")
    v = rng.randrange(2)
    if v == 0:
        task = (f"Make mypy pass for {scope}. Fix the errors in the file with the most of them first. Prefer a narrow type fix over a cast. "
                f"Open the file before you change it.")
        kind = "call"
    else:
        task = (f"Classify the mypy errors for {scope}: which ones show a real bug, which ones are missing annotations, and which ones "
                f"need a change in libs/common. One table, sorted by file.")
        kind = "report"
    call = {"name": "exec_command", "arguments": {"command": f"mypy {scope} --ignore-missing-imports 2>&1 | head -{num(rng, 40, 80)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_mypy"}


def fern_list_dir(rng, n):
    svc = pick(rng, PY_SERVICES)
    root = pick(rng, [f"services/{svc}", "libs", "tools", "services"])
    entries = []
    if root == "services":
        for s in sample(rng, PY_SERVICES, min(len(PY_SERVICES), max(4, n // 2))):
            entries.append(f"services/{s}/")
            for f in sample(rng, PY_FILES, num(rng, 2, 4)):
                entries.append(f"services/{s}/{f}")
            entries.append(f"services/{s}/__init__.py")
    else:
        entries.append(f"{root}/")
        for f in sample(rng, PY_FILES + ["__init__.py", "README.md", "py.typed", "fixtures/", "migrations/"], min(14, max(5, n))):
            entries.append(f"{root}/{f}")
        for d in ["fixtures", "migrations"]:
            if f"{root}/{d}/" in entries:
                for k in range(num(rng, 2, 5)):
                    entries.append(f"{root}/{d}/{pick(rng, ['0001_init', '0002_add_index', '0003_rename', 'sample', 'invoice_small', 'events'])}{k}.{pick(rng, ['sql', 'json', 'yaml'])}")
    entries = sorted(set(entries))
    lines = [f"{'d' if e.endswith('/') else '-'} {num(rng, 100, 9000) if not e.endswith('/') else 4096:>6}  {e}" for e in entries]
    v = rng.randrange(3)
    if v == 0:
        task = (f"Write a Layout section for the README of {root}/. One line per directory, then a markdown table of the files with the columns "
                f"name, purpose and size. Only use entries from the listing.")
        kind = "report"
    elif v == 1:
        task = (f"Find where the configuration loader of {root}/ lives, based on the listing, then open it and quote the function that reads "
                f"the environment.")
        kind = "call"
    else:
        task = (f"Which files under {root}/ have no test next to them under tests/? Compare against the listing and give a list; then open the "
                f"largest untested module.")
        kind = "call"
    call = {"name": "list_dir", "arguments": {"path": root, "depth": 2}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_list_dir"}


def _py_diff(rng, path, hunks):
    fn = pick(rng, PY_FUNCS)
    lines = [f"diff --git a/{path} b/{path}", f"index {sha(rng)}..{sha(rng)} 100644", f"--- a/{path}", f"+++ b/{path}"]
    for _ in range(hunks):
        a = num(rng, 10, 300)
        lines.append(f"@@ -{a},{num(rng, 6, 9)} +{a},{num(rng, 6, 11)} @@ def {fn}({pick(rng, ['payload', 'rows', 'cfg', 'client'])}):")
        lines += [f"     {pick(rng, ['calls = 0', 'out = []', 'deadline = time.monotonic() + timeout', 'cfg = _defaults()'])}",
                  f"-    {pick(rng, ['while calls <= retries:', 'for row in rows:', 'if resp.status == 200:', 'data = json.loads(raw)'])}",
                  f"+    {pick(rng, ['while calls < retries:', 'for row in rows or []:', 'if 200 <= resp.status < 300:', 'data = json.loads(raw or \"{}\")'])}",
                  f"         {pick(rng, ['calls += 1', 'out.append(_norm(row))', 'return resp.json()', 'cfg.update(data)'])}",
                  f"-        {pick(rng, ['time.sleep(1)', 'log.info(row)', 'except:', 'return cfg'])}",
                  f"+        {pick(rng, ['time.sleep(min(30, 2 ** calls))', 'log.debug(\"row %s\", row.get(\"id\"))', 'except OSError as e:', 'return Config(**cfg)'])}",
                  f"+        {pick(rng, ['log.warning(\"retry %d\", calls)', 'if row.get(\"id\") is None:', '    log.warning(\"fetch failed: %s\", e)', '# TODO: validate keys'])}",
                  f"     {pick(rng, ['return calls', 'return out', 'raise', ''])}"]
    return lines


def fern_git_diff(rng, n):
    files = [pyfile(rng) for _ in range(min(4, max(1, n // 5)))]
    lines = []
    for f in files:
        lines += _py_diff(rng, f, num(rng, 1, 2))
    if n > 12:
        t = f"tests/{pick(rng, PY_SERVICES)}/test_{pick(rng, ['tasks', 'client', 'config'])}.py"
        lines += [f"diff --git a/{t} b/{t}", f"index {sha(rng)}..{sha(rng)} 100644", f"--- a/{t}", f"+++ b/{t}",
                  f"@@ -{num(rng, 10, 90)},4 +{num(rng, 10, 90)},{num(rng, 8, 12)} @@", f"     assert calls == 3", "+", "+",
                  f"+def {pick(rng, PY_TESTS)}():", f"+    out = {pick(rng, PY_FUNCS)}({pick(rng, ['[]', 'None', '{}'])})",
                  f"+    assert out == {pick(rng, ['[]', 'None', '0'])}"]
    branch = f"{pick(rng, NAMES)}/{pick(rng, ['retry-backoff', 'config-env', 'null-rows', 'status-range', 'batch-size'])}"
    v = rng.randrange(3)
    if v == 0:
        task = (f"Review the diff of branch {branch} against main. List every behavior change, flag anything that can raise or loop, and "
                f"say whether the tests cover the change. Number the findings and give file and line for each.")
        kind = "report"
    elif v == 1:
        task = (f"The diff on {branch} changes the retry loop bound. Check whether the existing tests still hold; open the test file for "
                f"the touched service and quote the assertions that depend on the count.")
        kind = "call"
    else:
        task = (f"Write the commit message for the change on {branch}: an imperative subject under 60 characters, a blank line, and a body "
                f"that says what changed and why, based only on the diff.")
        kind = "report"
    call = {"name": "exec_command", "arguments": {"command": f"git diff main...{branch} -- services/ libs/ tests/ 2>&1 | head -{num(rng, 80, 160)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_git_diff"}


COMMIT_SUBJECTS = ["Add exponential backoff to the retry helper", "Fix a None tenant in the quota check", "Read the batch size from config",
                   "Drop the legacy invoice parser", "Log the request id on every handler", "Raise the default timeout to 30 s",
                   "Add the export job for monthly statements", "Cache the compiled route regex", "Validate the currency code on input",
                   "Rename dequeue_all to drain", "Add a health endpoint to the gateway", "Fix the leap day bug in merge_windows",
                   "Bump the storage client to 4.2", "Remove the print statements from the worker", "Add tests for the discount cap",
                   "Return 429 instead of 503 on quota exhaustion", "Move the email normalizer into libs/common", "Fix the unstable sort in the ledger",
                   "Retry on 502 and 504 as well", "Add a dry-run flag to the export cli"]


def fern_git_log(rng, n):
    tag = f"v{num(rng, 1, 4)}.{num(rng, 0, 12)}.0"
    lines = []
    for _ in range(max(5, n // 2)):
        s = pick(rng, COMMIT_SUBJECTS)
        lines.append(f"{sha(rng)} {s}")
        for f in [pyfile(rng) for _ in range(num(rng, 1, 3))] + ([f"tests/{pick(rng, PY_SERVICES)}/test_{pick(rng, ['tasks', 'client', 'handlers'])}.py"] if rng.random() < 0.6 else []):
            lines.append(f" {f:<44} | {num(rng, 2, 60):>3} {'+' * num(rng, 1, 8)}{'-' * num(rng, 0, 4)}")
        lines.append(f" {num(rng, 1, 4)} files changed, {num(rng, 3, 90)} insertions(+), {num(rng, 0, 40)} deletions(-)")
    v = rng.randrange(3)
    if v == 0:
        task = (f"Write the release notes for the next version from the git log since {tag}. Headings Added, Changed and Fixed. One bullet per "
                f"commit, written for an operator of the services, not for a developer. Skip test only commits.")
        kind = "report"
    elif v == 1:
        task = (f"From the log since {tag}, find the commit that touched libs/common most recently and show its full diff, then say whether "
                f"it changed a public function signature.")
        kind = "call"
    else:
        task = (f"Build a table of the commits since {tag} with the columns hash, subject, services touched and risk (low, medium, high). "
                f"Risk is high when the commit touches libs/common or a config file.")
        kind = "report"
    call = {"name": "exec_command", "arguments": {"command": f"git log --no-merges --stat --format='%h %s' {tag}..HEAD 2>&1 | head -{num(rng, 60, 120)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_git_log"}


LOG_LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]
LOG_MSGS = {
    "INFO": ["request id={rid} path=/v1/{svc}/{ep} status=200 ms={ms}", "job {job} finished rows={rows} ms={ms}", "health: db=ok cache=ok queue={q}",
             "worker {w} picked task {rid}", "config reloaded from deploy/config/prod.yaml", "connection pool size={q}"],
    "WARN": ["retry {k}/3 for upstream {svc} after {ms} ms", "queue depth {q} above soft limit 200", "slow query {ms} ms on {tbl}",
             "token for tenant {tid} expires in {k} h", "cache miss ratio {pct}% in the last minute"],
    "ERROR": ["request id={rid} failed: upstream {svc} returned 503", "job {job} failed: {exc}", "worker {w} exited with code 137",
              "cannot connect to db: connection refused (attempt {k})", "{exc} in {fn}", "tenant {tid}: quota exceeded, dropping batch of {rows}"],
    "DEBUG": ["cache key {rid} ttl={ms}", "batch of {rows} rows for tenant {tid}"],
}
EXCS = ["KeyError: 'tenant_id'", "TimeoutError", "ValueError: invalid currency 'EU'", "OSError: [Errno 28] No space left on device",
        "json.decoder.JSONDecodeError: Expecting value", "ConnectionResetError"]


def _log_line(rng, t0, i, svc):
    lvl = pick(rng, LOG_LEVELS)
    msg = pick(rng, LOG_MSGS[lvl]).format(rid="".join(rng.choice("0123456789abcdef") for _ in range(4)), svc=svc, ep=pick(rng, ["items", "orders", "export", "auth", "search"]),
                                          ms=num(rng, 3, 9000), job=f"{svc}.{pick(rng, ['nightly', 'hourly', 'sync'])}", rows=num(rng, 10, 90000), q=num(rng, 0, 900),
                                          w=num(rng, 1, 8), k=num(rng, 1, 3), tbl=pick(rng, ["orders", "invoices", "events"]), tid=num(rng, 10, 999), pct=num(rng, 5, 95),
                                          exc=pick(rng, EXCS), fn=pick(rng, PY_FUNCS))
    ts = t0 + i * num(rng, 1, 40)
    h, m, s = 8 + ts // 3600, (ts // 60) % 60, ts % 60
    return f"2026-09-{num(rng, 10, 18):02d}T{h:02d}:{m:02d}:{s:02d}Z {lvl:<5} {svc} {msg}"


def fern_logs(rng, n):
    svc = pick(rng, PY_SERVICES)
    t0 = num(rng, 0, 3000)
    lines = [_log_line(rng, t0, i, svc) for i in range(max(8, n))]
    v = rng.randrange(3)
    if v == 0:
        task = (f"Write an incident note from the {svc} log excerpt: a timeline table with UTC times, the likely cause, the user impact and "
                f"three follow-up actions with an owner. Under 300 words. Mark everything you cannot prove as not verified.")
        kind = "report"
    elif v == 1:
        task = (f"The {svc} log shows errors. Find the code that emits the first ERROR line, open it, and quote the branch that produces the "
                f"message. Then say what input triggers it.")
        kind = "call"
    else:
        task = (f"Count the log lines per level in the {svc} excerpt and list the distinct ERROR messages with their first timestamp. "
                f"One table. Then say which one to look at first and why.")
        kind = "report"
    call = {"name": "exec_command", "arguments": {"command": f"tail -{num(rng, 40, 120)} logs/{svc}.log 2>&1" if rng.random() < 0.5 else f"grep -v DEBUG logs/{svc}.log | tail -{num(rng, 40, 100)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "fern_logs"}


FERN = [fern_grep_markers, fern_grep_symbol, fern_pytest, fern_ruff, fern_mypy, fern_list_dir, fern_git_diff, fern_git_log, fern_logs]

# --------------------------------------------------------------------------- PIXEL (node / typescript)
COMPONENTS = ["Sidebar", "DataTable", "DateRangePicker", "Toast", "Modal", "UserMenu", "SearchBox", "Pagination", "StatusBadge",
              "FileUpload", "Breadcrumbs", "TokenChart", "SettingsForm", "InviteDialog", "TagInput", "ThemeToggle"]
HOOKS = ["useDebounce", "usePagination", "useSession", "useTheme", "useFetch", "useLocalStorage", "useHotkeys", "useToast"]
TS_ERRS = ["error TS2322: Type 'string | undefined' is not assignable to type 'string'.",
           "error TS2339: Property '{p}' does not exist on type '{t}Props'.",
           "error TS2345: Argument of type 'null' is not assignable to parameter of type '{t}'.",
           "error TS7006: Parameter '{v}' implicitly has an 'any' type.",
           "error TS2741: Property '{p}' is missing in type '{{ label: string; }}' but required in type '{t}Props'.",
           "error TS18048: '{v}' is possibly 'undefined'.",
           "error TS2554: Expected {a} arguments, but got {b}.",
           "error TS6133: '{v}' is declared but its value is never read.",
           "error TS2769: No overload matches this call."]
ESLINT_RULES = [("@typescript-eslint/no-unused-vars", "'{v}' is defined but never used"), ("react-hooks/exhaustive-deps", "React Hook useEffect has a missing dependency: '{v}'"),
                ("no-console", "Unexpected console statement"), ("@typescript-eslint/no-explicit-any", "Unexpected any. Specify a different type"),
                ("react/jsx-key", "Missing \"key\" prop for element in iterator"), ("eqeqeq", "Expected '===' and instead saw '=='"),
                ("prefer-const", "'{v}' is never reassigned. Use 'const' instead"), ("import/order", "There should be at least one empty line between import groups"),
                ("react-hooks/rules-of-hooks", "React Hook \"{h}\" is called conditionally"), ("no-nested-ternary", "Do not nest ternary expressions")]
TS_VARS = ["rows", "onClose", "value", "selected", "page", "items", "err", "timer", "ref", "user"]


def tsfile(rng, comp=None):
    c = comp or pick(rng, COMPONENTS)
    k = rng.random()
    if k < 0.55:
        return f"src/components/{c}/{c}.tsx"
    if k < 0.75:
        return f"src/hooks/{pick(rng, HOOKS)}.ts"
    if k < 0.9:
        return f"src/api/{pick(rng, ['client', 'sessions', 'tokens', 'usage', 'workspaces'])}.ts"
    return f"src/store/{pick(rng, ['session', 'theme', 'workspace', 'notifications'])}.ts"


def pixel_vitest(rng, n):
    comp = pick(rng, COMPONENTS)
    test = f"src/components/{comp}/{comp}.test.tsx"
    nfail = min(3, max(1, n // 5))
    names = sample(rng, ["renders the empty state", "calls onClose on escape", "formats the date range", "keeps the selection after a page change",
                         "shows the error toast", "debounces the search input", "matches the snapshot", "disables the button while loading",
                         "sorts by the clicked column", "renders 50 rows without a warning"], nfail + num(rng, 2, 5))
    fails = names[:nfail]
    lines = [f" RUN  v3.{num(rng, 0, 2)}.{num(rng, 0, 9)} /work/quartz-console", "",
             f" {'x' if True else ''} {test} ({len(names)} tests | {nfail} failed) {num(rng, 100, 2500)}ms"]
    for nm in names:
        lines.append(f"   {'x' if nm in fails else 'ok'} {comp} > {nm} {num(rng, 1, 60)}ms" if nm not in fails or 'timeout' not in nm else f"   x {comp} > {nm} {num(rng, 5000, 5100)}ms")
    lines += [""]
    for i, nm in enumerate(fails):
        k = rng.randrange(4)
        lines += [f" FAIL  {test} > {comp} > {nm}"]
        if k == 0:
            lines += [f"AssertionError: expected {pick(rng, ['\"1h 5m\"', '\"12 Sep 2026\"', '3', 'true', '\"Loading...\"'])} to {pick(rng, ['be', 'deeply equal', 'contain'])} {pick(rng, ['\"1h 05m\"', '\"2026-09-12\"', '2', 'false', '\"Loading\"'])}",
                      "", f"- Expected", f"+ Received", "", f"- {pick(rng, ['\"1h 05m\"', '\"2026-09-12\"', 'false'])}", f"+ {pick(rng, ['\"1h 5m\"', '\"12 Sep 2026\"', 'true'])}", "",
                      f" > {test}:{num(rng, 10, 90)}:{num(rng, 5, 40)}"]
        elif k == 1:
            lines += [f"Error: Test timed out in 5000ms.", "If this is a long-running test, pass a timeout value as the last argument or configure it globally with \"testTimeout\".",
                      f" > {test}:{num(rng, 10, 90)}:{num(rng, 5, 40)}"]
        elif k == 2:
            lines += [f"Error: Snapshot `{comp} > {nm} 1` mismatched", "", "- Expected", "+ Received", "", f"  <div", f"-   class=\"{comp.lower()} {pick(rng, ['is-open', 'compact', 'dark'])}\"",
                      f"+   class=\"{comp.lower()}\"", "  >", f" > {test}:{num(rng, 10, 90)}:{num(rng, 5, 40)}"]
        else:
            lines += [f"TestingLibraryElementError: Unable to find an element with the text: {pick(rng, ['No results', 'Save changes', 'Invite sent', 'Page 2 of 5'])}. This could be because the text is broken up by multiple elements.",
                      "", "Ignored nodes: comments, script, style", f"<body>", f"  <div>", f"    <div class=\"{comp.lower()}\">", f"      {pick(rng, ['Loading...', 'No items', 'Save', 'Page 1 of 5'])}", "    </div>", "  </div>", "</body>",
                      f" > {test}:{num(rng, 10, 90)}:{num(rng, 5, 40)}"]
        lines.append("")
    lines += [f" Test Files  1 failed | {num(rng, 8, 40)} passed ({num(rng, 9, 41)})", f"      Tests  {nfail} failed | {num(rng, 60, 300)} passed ({num(rng, 63, 303)})",
              f"   Start at  {num(rng, 8, 17):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d}", f"   Duration  {num(rng, 2, 30)}.{num(rng, 10, 99)}s"]
    if n > 12:
        lines += ["", f"$ sed -n 1,{num(rng, 22, 34)}p {test}", f"import {{ render, screen, fireEvent }} from \"@testing-library/react\";", f"import {{ {comp} }} from \"./{comp}\";",
                  f"import {{ {pick(rng, HOOKS)} }} from \"../../hooks/{pick(rng, HOOKS)}\";", "", f"describe(\"{comp}\", () => {{",
                  f"  it(\"{fails[0]}\", {'async ' if 'debounce' in fails[0] else ''}() => {{", f"    render(<{comp} {pick(rng, ['rows={[]}', 'open', 'value=\"\"', 'page={1} total={5}'])} />);",
                  f"    {pick(rng, ['expect(screen.getByText(\"No results\")).toBeInTheDocument();', 'fireEvent.keyDown(document, { key: \"Escape\" });', 'expect(screen.getByRole(\"button\")).toBeDisabled();', 'expect(container).toMatchSnapshot();'])}",
                  "  });", "});"]
    v = rng.randrange(3)
    if v == 0:
        task = (f"The {comp} test file has {nfail} failing test{'s' if nfail > 1 else ''}. Diagnose each failure from the output, fix the component or "
                f"the hook it uses, and run the file again. Do not update a snapshot by hand; report it if the snapshot is wrong.")
        kind = "call"
    elif v == 1:
        task = (f"Explain the {nfail} failure{'s' if nfail > 1 else ''} in {test} to the lead agent: what the test expects, what the component renders, "
                f"and which side is wrong. Two sentences per failure, then a proposed change for each in a diff block. Change nothing.")
        kind = "report"
    else:
        task = (f"Triage the {comp} test failures. Open the component and quote the code path for the first failing test. Say whether the fix "
                f"belongs in the component, the hook or the test.")
        kind = "call"
    call = {"name": "npm_run", "arguments": {"script": "test", "args": f"-- {test} --run --reporter=verbose 2>&1 | tail -{num(rng, 60, 110)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_vitest"}


def pixel_eslint(rng, n):
    files = [tsfile(rng) for _ in range(num(rng, 3, 5))]
    scope = pick(rng, ["src/components/", "src/", "src/hooks/ src/api/"])
    by = {}
    for _ in range(max(6, n)):
        f = pick(rng, files)
        rule, msg = pick(rng, ESLINT_RULES)
        msg = msg.format(v=pick(rng, TS_VARS), h=pick(rng, HOOKS))
        by.setdefault(f, []).append((num(rng, 1, 300), num(rng, 1, 60), rule, msg))
    lines = []
    total = 0
    for f in sorted(by):
        lines.append(f"/work/quartz-console/{f}")
        for ln, col, rule, msg in sorted(by[f]):
            lines.append(f"  {ln}:{col}  {'error' if rule.startswith('react-hooks/rules') or rule == 'react/jsx-key' else 'warning'}  {msg}  {rule}")
            total += 1
        lines.append("")
    nerr = sum(1 for l in lines if "  error  " in l)
    lines.append(f"x {total} problems ({nerr} errors, {total - nerr} warnings)")
    lines.append(f"  {num(rng, 1, max(1, total // 2))} errors and {num(rng, 1, max(1, total // 2))} warnings potentially fixable with the `--fix` option.")
    v = rng.randrange(3)
    top = max(by, key=lambda f: len(by[f]))
    if v == 0:
        task = (f"Clean up the ESLint findings under {scope.strip()}. Fix the file with the most findings by hand, with no behavior change and "
                f"no eslint-disable comments. Run lint on that file again.")
        kind = "call"
    elif v == 1:
        task = (f"Summarize the ESLint output for {scope.strip()}: a table of rule, count and whether --fix handles it. Then say which finding is a "
                f"real bug and why. Do not edit anything.")
        kind = "report"
    else:
        task = (f"Open {top} and quote every line the linter flagged, then fix the exhaustive-deps and rules-of-hooks findings first. "
                f"Leave the style warnings for later.")
        kind = "call"
    call = {"name": "npm_run", "arguments": {"script": "lint", "args": f"-- {scope} 2>&1 | head -{num(rng, 50, 90)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_eslint"}


def pixel_tsc(rng, n):
    files = [tsfile(rng) for _ in range(num(rng, 2, 5))]
    lines = []
    for _ in range(max(4, n)):
        f = pick(rng, files)
        comp = f.split("/")[2] if "/components/" in f else pick(rng, COMPONENTS)
        e = pick(rng, TS_ERRS).format(p=pick(rng, ["onSelect", "rows", "dense", "onClose", "initialPage"]), t=comp, v=pick(rng, TS_VARS), a=num(rng, 1, 2), b=num(rng, 2, 4))
        lines.append(f"{f}({num(rng, 5, 300)},{num(rng, 1, 60)}): {e}")
    lines.sort()
    lines += ["", f"Found {len(lines)} errors in {len(files)} files.", "", "Errors  Files"]
    for f in files:
        c = sum(1 for l in lines if l.startswith(f))
        if c:
            lines.append(f"     {c}  {f}:{num(rng, 5, 300)}")
    v = rng.randrange(2)
    if v == 0:
        task = (f"Make `npm run typecheck` pass. Start with {files[0]}: open it, quote the lines the errors point at, and fix them with narrow types. "
                f"No `any` and no non-null assertions. Run typecheck again.")
        kind = "call"
    else:
        task = (f"Explain each of the {sum(1 for l in lines if ': error TS' in l)} TypeScript errors in the output in one sentence and say whether it "
                f"hides a runtime bug. Group by file. Propose the minimal fix for each as a code snippet. Do not change files.")
        kind = "report"
    call = {"name": "npm_run", "arguments": {"script": "typecheck", "args": f"2>&1 | head -{num(rng, 40, 80)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_tsc"}


def pixel_search(rng, n):
    target = pick(rng, [("useFetch(", "the useFetch hook"), ("<Modal", "the Modal component"), ("localStorage.", "direct localStorage access"),
                        ("theme.colors.", "hard coded theme color lookups"), ("any", "explicit any types"), ("console.log(", "console.log calls"),
                        ("fetch(", "raw fetch calls"), ("dangerouslySetInnerHTML", "raw HTML injection")])
    lines = []
    for _ in range(max(5, n)):
        f = tsfile(rng)
        ctx = pick(rng, [f"  const {{ data, error }} = {target[0]}\"/api/{pick(rng, ['usage', 'tokens', 'sessions'])}\");",
                         f"      {target[0]} {pick(rng, ['open={open}', 'title=\"Invite\"', 'size=\"lg\"'])} onClose={{() => set{pick(rng, ['Open', 'Show'])}(false)}}>",
                         f"    {target[0]}getItem(\"{pick(rng, ['theme', 'token', 'lastWorkspace'])}\");", f"  color: {target[0]}{pick(rng, ['primary', 'muted', 'danger'])},",
                         f"  {pick(rng, TS_VARS)}: {target[0]}", f"    {target[0]}\"{pick(rng, ['render', 'rows', 'selected'])}\", {pick(rng, TS_VARS)});",
                         f"  const res = await {target[0]}`${{BASE}}/{pick(rng, ['usage', 'me', 'keys'])}`);"])
        lines.append(f"{f}:{num(rng, 5, 300)}:{ctx}")
    lines.sort()
    v = rng.randrange(3)
    if v == 0:
        task = (f"Find every use of {target[1]} under src/ and replace it with {pick(rng, ['the shared api client', 'the useStorage hook', 'the design tokens', 'the logger from src/lib/log.ts', 'the Dialog primitive'])}. "
                f"Change one file at a time and run its test after each edit. Start with the file that has the most matches.")
        kind = "call"
    elif v == 1:
        task = (f"Count the uses of {target[1]} per directory under src/ and rank the files. Say which three files to migrate first and why, "
                f"in a short table. Do not edit anything.")
        kind = "report"
    else:
        task = (f"Open the first file in the search result that uses {target[1]} and quote the surrounding function. Then say whether the "
                f"replacement needs a new prop or a new hook.")
        kind = "call"
    call = {"name": "search_source", "arguments": {"pattern": target[0].replace("(", r"\(").replace(".", r"\."), "glob": "src/**/*.ts*"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_search"}


def pixel_build(rng, n):
    lines = [f"> quartz-console@{num(rng, 2, 5)}.{num(rng, 0, 20)}.{num(rng, 0, 9)} build", "> tsc -b && vite build", "",
             f"vite v{num(rng, 6, 7)}.{num(rng, 0, 3)}.{num(rng, 0, 9)} building for production...", f"transforming ({num(rng, 800, 2400)}) src/main.tsx"]
    for _ in range(max(3, n // 3)):
        k = rng.randrange(4)
        if k == 0:
            lines.append(f"(!) {tsfile(rng)} is dynamically imported by src/routes.tsx but also statically imported by {tsfile(rng)}, dynamic import will not move module into another chunk.")
        elif k == 1:
            lines.append(f"[plugin:vite:esbuild] {tsfile(rng)}:{num(rng, 5, 200)}:{num(rng, 1, 40)}: warning: Duplicate key \"{pick(rng, TS_VARS)}\" in object literal")
        elif k == 2:
            lines.append(f"warning: \"{pick(rng, ['default', 'formatRate', 'useLegacyTheme'])}\" is not exported by \"{tsfile(rng)}\", imported by \"{tsfile(rng)}\".")
        else:
            lines.append(f"(!) Some chunks are larger than 500 kB after minification. Consider code splitting.")
    lines += [f"x {num(rng, 400, 1200)} modules transformed.", "rendering chunks...", "computing gzip size..."]
    for nm in sample(rng, ["index", "vendor", "charts", "settings", "editor", "workspace"], num(rng, 3, 6)):
        lines.append(f"dist/assets/{nm}-{sha(rng)}.js   {num(rng, 20, 900)}.{num(rng, 10, 99)} kB | gzip: {num(rng, 5, 300)}.{num(rng, 10, 99)} kB")
    if rng.random() < 0.6:
        f = tsfile(rng)
        lines += [f"x Build failed in {num(rng, 3, 20)}.{num(rng, 10, 99)}s", f"error during build:", f"[vite]: Rollup failed to resolve import \"{pick(rng, ['@/lib/format', 'dayjs/plugin/utc', './tokens', 'recharts/es6'])}\" from \"{f}\".",
                  "This is most likely unintended because it can break your application at runtime.", "If you do want to externalize this module explicitly add it to `build.rollupOptions.external`"]
        failed = True
    else:
        lines.append(f"x built in {num(rng, 3, 20)}.{num(rng, 10, 99)}s")
        failed = False
    v = rng.randrange(2)
    nmod = [l for l in lines if "modules transformed" in l][0].split()[1]
    if v == 0 and failed:
        task = (f"The production build fails after {nmod} modules. Find the file with the unresolved import, open it, and fix the import path. "
                f"Then run the build again and quote the last line.")
        kind = "call"
    else:
        task = (f"Read the build output ({nmod} modules, {sum(1 for l in lines if l.startswith('dist/'))} chunks) and write a short note for the lead agent: "
                f"whether the build passed, the warnings grouped by type, and the chunks over 500 kB with a suggestion for each. Under 200 words.")
        kind = "report"
    call = {"name": "npm_run", "arguments": {"script": "build", "args": f"2>&1 | tail -{num(rng, 30, 60)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_build"}


def pixel_list(rng, n):
    comp = pick(rng, COMPONENTS)
    root = pick(rng, [f"src/components/{comp}", "src/hooks", "src/api", "src/components"])
    lines = []
    if root == "src/components":
        for c in sample(rng, COMPONENTS, min(len(COMPONENTS), max(5, n))):
            lines.append(f"{c}/")
    elif root == "src/hooks":
        for h in sample(rng, HOOKS, min(len(HOOKS), max(4, n // 2))):
            lines.append(f"{h}.ts")
            if rng.random() < 0.6:
                lines.append(f"{h}.test.ts")
        lines.append("index.ts")
    elif root == "src/api":
        for a in ["client.ts", "sessions.ts", "tokens.ts", "usage.ts", "workspaces.ts", "types.ts", "index.ts", "__tests__/"]:
            lines.append(a)
        lines.append("mocks/")
    else:
        for f in [f"{comp}.tsx", f"{comp}.test.tsx", f"{comp}.module.css", "index.ts", "__snapshots__/", f"{comp}.stories.tsx", "helpers.ts", "types.ts"]:
            if rng.random() < 0.8:
                lines.append(f)
    lines = sorted(set(lines))
    lines = [f"{'d' if e.endswith('/') else 'f'}  {num(rng, 200, 30000) if not e.endswith('/') else '-':>6}  {e}" for e in lines]
    if n > 10:
        lines += ["", f"$ cat {root}/index.ts"]
        for e in sample(rng, COMPONENTS if root == 'src/components' else HOOKS, num(rng, 4, 9)):
            lines.append(f"export {{ {e} }} from \"./{e}{'/' + e if root == 'src/components' else ''}\";")
        lines += ["", f"$ wc -l {root}/*.ts*"]
        for e in sample(rng, COMPONENTS if root == 'src/components' else HOOKS, num(rng, 4, 9)):
            lines.append(f"  {num(rng, 20, 900):>5} {root}/{e}{'/' + e + '.tsx' if root == 'src/components' else '.ts'}")
    if n > 20:
        lines += ["", "$ sed -n 1,24p package.json", "{", f"  \"name\": \"quartz-console\",", f"  \"version\": \"{num(rng, 2, 5)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}\",", "  \"scripts\": {",
                  "    \"dev\": \"vite\",", "    \"build\": \"tsc -b && vite build\",", "    \"test\": \"vitest\",", "    \"lint\": \"eslint src --ext .ts,.tsx\",", "    \"typecheck\": \"tsc --noEmit\",",
                  "    \"format:check\": \"prettier --check src\"", "  },", "  \"dependencies\": {", f"    \"react\": \"^19.{num(rng, 0, 2)}.0\",", f"    \"react-dom\": \"^19.{num(rng, 0, 2)}.0\",",
                  f"    \"zustand\": \"^5.{num(rng, 0, 3)}.0\",", f"    \"dayjs\": \"^1.11.{num(rng, 10, 14)}\"", "  }", "}"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Describe the layout of {root}/ for the lead agent from the listing: one line per entry with its role, and a note on which "
                f"entries lack a test. Do not open files.")
        kind = "report"
    else:
        task = f"Open the main source file under {root}/ from the listing and quote its exported props or functions, so the lead agent can plan the change."
        kind = "call"
    call = {"name": "list_files", "arguments": {"path": root}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "pixel_list"}


PIXEL = [pixel_vitest, pixel_eslint, pixel_tsc, pixel_search, pixel_build, pixel_list]

# --------------------------------------------------------------------------- ANVIL (go / rust)
GO_PKGS = ["conn", "route", "tls", "config", "parser", "metrics", "ratelimit", "health", "upstream", "buffer"]
GO_TESTS = ["TestDialTimeout", "TestRouteLongestPrefix", "TestReloadKeepsConnections", "TestParseFrameShort", "TestRateLimitBurst",
            "TestHealthDegraded", "TestUpstreamRetry", "TestBufferGrow", "TestConfigDefaults", "TestTLSRotate", "TestRouteWildcard",
            "TestConnCloseIdempotent", "TestMetricsLabels", "TestParseFrameChecksum"]
RS_TESTS = ["parse_frame_rejects_short_header", "parse_frame_accepts_max_len", "checksum_roundtrip", "frame_iter_stops_at_eof",
            "header_flags_decode", "payload_len_overflow", "varint_boundaries", "frame_equality"]
GO_ERRS = ["cannot use {v} (variable of type {t1}) as {t2} value in {ctx}", "undefined: {id}", "{id} declared and not used",
           "missing return", "cannot use nil as {t2} value in return statement", "too many arguments in call to {id}",
           "invalid operation: {v} == nil (mismatched types {t1} and untyped nil)", "{id}.{f} undefined (type {t1} has no field or method {f})",
           "imported and not used: \"{imp}\""]
GO_TYPES = ["*Conn", "int", "string", "time.Duration", "[]byte", "error", "*Route", "context.Context", "uint32"]
GO_IDS = ["dialWithTimeout", "matchPrefix", "reloadCerts", "parseFrame", "allowBurst", "checkUpstream", "grow", "defaults", "rotate", "lookup"]


def gopath(rng, pkg=None):
    p = pkg or pick(rng, GO_PKGS)
    return f"internal/{p}/{pick(rng, [p, 'handler', 'pool', 'table', 'reload', 'limits', p + '_test'])}.go"


def anvil_go_test(rng, n):
    pkg = pick(rng, GO_PKGS)
    nfail = min(3, max(1, n // 6))
    fails = sample(rng, GO_TESTS, nfail)
    lines = []
    for t in fails:
        k = rng.randrange(4)
        lines.append(f"--- FAIL: {t} ({num(rng, 0, 3)}.{num(rng, 0, 99):02d}s)")
        f = gopath(rng, pkg)
        if k == 0:
            lines.append(f"    {f.split('/')[-1]}:{num(rng, 20, 300)}: got {pick(rng, ['3', '\"/api/v1\"', 'nil', '0', 'true'])}, want {pick(rng, ['2', '\"/api/v1/\"', 'ErrClosed', '1', 'false'])}")
        elif k == 1:
            lines += [f"    {f.split('/')[-1]}:{num(rng, 20, 300)}: unexpected error: {pick(rng, ['dial tcp 127.0.0.1:0: connect: connection refused', 'context deadline exceeded', 'frame too short: 3 bytes', 'tls: bad certificate', 'route not found: /v2'])}"]
        elif k == 2:
            lines += [f"panic: runtime error: {pick(rng, ['index out of range [4] with length 4', 'invalid memory address or nil pointer dereference', 'slice bounds out of range [:8] with capacity 6'])} [recovered]",
                      f"\tpanic: runtime error: {pick(rng, ['index out of range [4] with length 4', 'invalid memory address or nil pointer dereference'])}", "",
                      f"goroutine {num(rng, 7, 90)} [running]:", "testing.tRunner.func1.2(...)", f"\t/usr/local/go/src/testing/testing.go:{num(rng, 1600, 1700)}",
                      f"harbor/internal/{pkg}.{pick(rng, GO_IDS)}(0xc000{sha(rng)[:6]}, {{0x{sha(rng)[:5]}, 0x{num(rng, 1, 9)}}})",
                      f"\t/work/harbor/{f}:{num(rng, 20, 300)} +0x{sha(rng)[:3]}", f"harbor/internal/{pkg}.{t}(0xc000{sha(rng)[:6]})",
                      f"\t/work/harbor/{f.replace('.go', '_test.go') if not f.endswith('_test.go') else f}:{num(rng, 20, 200)} +0x{sha(rng)[:3]}"]
        else:
            lines += [f"    {f.split('/')[-1]}:{num(rng, 20, 300)}: timeout after {num(rng, 1, 10)}s waiting for {pick(rng, ['reload', 'close', 'ack', 'drain'])}",
                      f"    {f.split('/')[-1]}:{num(rng, 20, 300)}: goroutine leak: {num(rng, 1, 6)} goroutines still running"]
    lines.append("FAIL")
    lines.append(f"FAIL\tharbor/internal/{pkg}\t{num(rng, 0, 12)}.{num(rng, 100, 999)}s")
    for p in sample(rng, [x for x in GO_PKGS if x != pkg], num(rng, 2, 5)):
        lines.insert(0, f"ok  \tharbor/internal/{p}\t{num(rng, 0, 4)}.{num(rng, 100, 999)}s")
    lines.append("FAIL")
    if n > 14:
        f = gopath(rng, pkg)
        idf = pick(rng, GO_IDS)
        lines += ["", f"$ sed -n {num(rng, 30, 60)},{num(rng, 61, 84)}p {f}", f"func {idf}({pick(rng, ['ctx context.Context, addr string', 'r *Route, path string', 'b []byte', 'c *Conn'])}) ({pick(rng, ['*Conn, error', 'bool', '*Frame, error', 'error'])}) {{",
                  f"\t{pick(rng, ['d := net.Dialer{Timeout: c.timeout}', 'if len(b) < 4 {', 'for i := range r.prefixes {', 'c.mu.Lock()'])}",
                  f"\t{pick(rng, ['conn, err := d.DialContext(ctx, \"tcp\", addr)', '\treturn nil, ErrShort', '\tif strings.HasPrefix(path, r.prefixes[i]) {', 'defer c.mu.Unlock()'])}",
                  f"\t{pick(rng, ['if err != nil {', '}', '\t\treturn true', 'if c.closed {'])}", f"\t\t{pick(rng, ['return nil, err', 'n := binary.BigEndian.Uint32(b[:4])', '}', 'return nil'])}",
                  f"\t{pick(rng, ['}', 'return &Frame{Len: n, Payload: b[4:n]}, nil', '}', '}'])}", f"\treturn {pick(rng, ['&Conn{c: conn}, nil', 'false', 'c.close()', 'nil'])}", "}"]
    v = rng.randrange(3)
    if v == 0:
        task = (f"The {pkg} package has {nfail} failing test{'s' if nfail > 1 else ''}. Find the cause of each in the non-test code, patch it, and run "
                f"`go test ./internal/{pkg}/ -count=1` again. Do not change the tests.")
        kind = "call"
    elif v == 1:
        task = (f"Explain the {pkg} test failures: for each one, the test's expectation, the observed value or panic, and the function to look at. "
                f"Then give a proposed patch per failure as a unified diff. Do not apply anything.")
        kind = "report"
    else:
        task = (f"Trace the first failure in internal/{pkg}: search for the function named in the output, print it, and quote the branch that "
                f"produces the wrong result. Say whether the fix is one line or larger.")
        kind = "call"
    call = {"name": "run_cmd", "arguments": {"command": f"go test ./internal/... -count=1 2>&1 | tail -{num(rng, 40, 90)}" if rng.random() < 0.5 else f"go test ./internal/{pkg}/ -count=1 -v 2>&1 | grep -v '^=== RUN' | tail -{num(rng, 40, 80)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_go_test"}


def anvil_cargo_test(rng, n):
    nfail = min(3, max(1, n // 6))
    fails = sample(rng, RS_TESTS, nfail)
    total = num(rng, 20, 60)
    lines = [f"   Compiling parser v0.{num(rng, 3, 9)}.{num(rng, 0, 9)} (/work/harbor/parser)",
             f"    Finished `test` profile [unoptimized + debuginfo] target(s) in {num(rng, 2, 40)}.{num(rng, 10, 99)}s",
             f"     Running unittests src/lib.rs (target/debug/deps/parser-{sha(rng)}{sha(rng)[:5]})", "", f"running {total} tests"]
    for t in sample(rng, RS_TESTS, min(len(RS_TESTS), nfail + 3)):
        lines.append(f"test tests::{t} ... {'FAILED' if t in fails else 'ok'}")
    lines += ["", "failures:", ""]
    for t in fails:
        k = rng.randrange(3)
        lines.append(f"---- tests::{t} stdout ----")
        if k == 0:
            lines += [f"thread 'tests::{t}' panicked at parser/src/{pick(rng, ['lib.rs', 'frame.rs', 'header.rs', 'varint.rs'])}:{num(rng, 20, 400)}:{num(rng, 5, 40)}:",
                      f"assertion `left == right` failed", f"  left: {pick(rng, ['Err(Short)', 'Ok(Frame { len: 8, flags: 0 })', '12', 'Some(3)'])}",
                      f" right: {pick(rng, ['Ok(Frame { len: 4, flags: 1 })', 'Err(Overflow)', '13', 'None'])}"]
        elif k == 1:
            lines += [f"thread 'tests::{t}' panicked at parser/src/{pick(rng, ['lib.rs', 'frame.rs', 'varint.rs'])}:{num(rng, 20, 400)}:{num(rng, 5, 40)}:",
                      f"{pick(rng, ['range end index 8 out of range for slice of length 6', 'attempt to add with overflow', 'called `Option::unwrap()` on a `None` value', 'attempt to shift left with overflow'])}"]
        else:
            lines += [f"thread 'tests::{t}' panicked at parser/src/{pick(rng, ['lib.rs', 'frame.rs'])}:{num(rng, 20, 400)}:{num(rng, 5, 40)}:",
                      f"assertion failed: {pick(rng, ['frame.len() <= MAX_LEN', 'iter.next().is_none()', 'flags & 0x80 == 0', 'checksum(&buf) == expected'])}"]
        lines += ["note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace", ""]
    lines += ["failures:"] + [f"    tests::{t}" for t in fails] + ["", f"test result: FAILED. {total - nfail} passed; {nfail} failed; 0 ignored; 0 measured; 0 filtered out; finished in {num(rng, 0, 3)}.{num(rng, 10, 99)}s", "",
                                                                     "error: test failed, to rerun pass `-p parser --lib`"]
    if n > 14:
        lines += ["", f"$ sed -n {num(rng, 40, 80)},{num(rng, 81, 104)}p parser/src/frame.rs",
                  f"pub fn parse_frame(buf: &[u8]) -> Result<Frame, ParseError> {{", "    if buf.len() < HEADER_LEN {", "        return Err(ParseError::Short);", "    }",
                  f"    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;", f"    let flags = buf[4];",
                  f"    {pick(rng, ['let payload = &buf[HEADER_LEN..len];', 'let payload = &buf[HEADER_LEN..HEADER_LEN + len];', 'if len > MAX_LEN { return Err(ParseError::Overflow); }'])}",
                  f"    {pick(rng, ['let sum = checksum(payload);', 'let sum = checksum(&buf[..len]);', 'let payload = &buf[HEADER_LEN..];'])}",
                  "    Ok(Frame { len, flags, payload: payload.to_vec(), sum })", "}"]
    v = rng.randrange(3)
    if v == 0:
        task = (f"`cargo test -p parser` has {nfail} failing test{'s' if nfail > 1 else ''} ({', '.join(fails)}). Fix the parser code so they pass without a "
                f"change to the tests, then run the same command and quote the result line.")
        kind = "call"
    elif v == 1:
        task = (f"For each failing parser test ({', '.join(fails)}), explain the panic or the assertion in two sentences and name the function and "
                f"line to fix. Then write the fix as a unified diff. Do not apply it.")
        kind = "report"
    else:
        task = (f"Print the function behind the failure of {fails[0]} and quote the slice or arithmetic expression that panics. Say what input "
                f"length triggers it and whether the fix needs a bounds check or a saturating operation.")
        kind = "call"
    call = {"name": "run_cmd", "arguments": {"command": f"cargo test -p parser 2>&1 | tail -{num(rng, 50, 90)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_cargo_test"}


def anvil_go_build(rng, n):
    pkg = pick(rng, GO_PKGS)
    lines = [f"# harbor/internal/{pkg}"]
    for _ in range(max(3, n // 2)):
        f = gopath(rng, pkg)
        e = pick(rng, GO_ERRS).format(v=pick(rng, ["c", "r", "buf", "timeout", "n", "cfg"]), t1=pick(rng, GO_TYPES), t2=pick(rng, GO_TYPES),
                                     ctx=pick(rng, ["argument to " + pick(rng, GO_IDS), "assignment", "struct literal", "return statement"]),
                                     id=pick(rng, GO_IDS), f=pick(rng, ["Close", "Len", "Prefix", "Deadline", "Reload"]), imp=pick(rng, ["fmt", "os", "strings", "time", "bytes"]))
        lines.append(f"{f}:{num(rng, 5, 400)}:{num(rng, 1, 60)}: {e}")
    if rng.random() < 0.5:
        p2 = pick(rng, [x for x in GO_PKGS if x != pkg])
        lines += [f"# harbor/internal/{p2}", f"{gopath(rng, p2)}:{num(rng, 5, 400)}:{num(rng, 1, 60)}: {pick(rng, GO_ERRS).format(v='x', t1='int', t2='string', ctx='assignment', id=pick(rng, GO_IDS), f='Len', imp='fmt')}"]
    lines.append(f"$ echo exit=$?")
    lines.append("exit=1")
    v = rng.randrange(2)
    if v == 0:
        task = (f"`go build ./...` fails in internal/{pkg}. Fix the first error only, print the surrounding code before you patch, rebuild, and "
                f"report whether the other errors were a cascade.")
        kind = "call"
    else:
        task = (f"Read the build errors for internal/{pkg} and classify each as a cascade of an earlier error or an independent problem. "
                f"For each independent one, name the file, the line and the one line fix. Do not patch.")
        kind = "report"
    call = {"name": "build", "arguments": {"target": "./...", "release": False}} if rng.random() < 0.5 else {"name": "run_cmd", "arguments": {"command": "go build ./... 2>&1; echo exit=$?"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_go_build"}


def anvil_clippy(rng, n):
    lints = [("needless_return", "unneeded `return` statement"), ("manual_range_contains", "manual `!RangeInclusive::contains` implementation"),
             ("len_zero", "length comparison to zero"), ("redundant_clone", "redundant clone"), ("unwrap_used", "used `unwrap()` on an `Option` value"),
             ("too_many_arguments", "this function has too many arguments (9/7)"), ("needless_borrow", "this expression creates a reference which is immediately dereferenced by the compiler"),
             ("collapsible_if", "this `if` statement can be collapsed"), ("cast_possible_truncation", "casting `u64` to `u32` may truncate the value"),
             ("large_enum_variant", "large size difference between variants")]
    lines = [f"    Checking parser v0.{num(rng, 3, 9)}.{num(rng, 0, 9)} (/work/harbor/parser)"]
    for _ in range(max(4, n // 3)):
        name, msg = pick(rng, lints)
        f = f"parser/src/{pick(rng, ['lib.rs', 'frame.rs', 'header.rs', 'varint.rs', 'iter.rs'])}"
        ln = num(rng, 10, 400)
        lines += [f"{'error' if rng.random() < 0.5 else 'warning'}: {msg}", f"  --> {f}:{ln}:{num(rng, 5, 30)}", "   |",
                  f"{ln:<3}|     {pick(rng, ['return Ok(frame);', 'if x >= 0 && x <= 255 {', 'if buf.len() == 0 {', 'let b = buf.clone();', 'let n = it.next().unwrap();', 'let len = total as u32;'])}",
                  f"   |     {'^' * num(rng, 8, 24)}", "   |",
                  f"   = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#{name}", ""]
    nerr = sum(1 for l in lines if l.startswith("error:"))
    nwarn = sum(1 for l in lines if l.startswith("warning:"))
    if nerr:
        lines += [f"error: could not compile `parser` (lib) due to {nerr} previous error{'s' if nerr > 1 else ''}; {nwarn} warning{'s' if nwarn != 1 else ''} emitted"]
    else:
        lines += [f"warning: `parser` (lib) generated {nwarn} warning{'s' if nwarn != 1 else ''}"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Make `cargo clippy -p parser -- -D warnings` pass {pick(rng, ['before the merge', 'for the 0.' + str(num(rng, 3, 9)) + ' release', 'so the lint gate is green', 'on the current branch'])}: {nerr} error{'s' if nerr != 1 else ''} and {nwarn} warning{'s' if nwarn != 1 else ''} now. "
                f"Fix each finding in place with no behavior change. Print each function before you patch it. Rerun clippy at the end.")
        kind = "call"
    else:
        task = (f"Group the {nerr + nwarn} clippy findings by lint name with a count and a one line explanation of each lint. Flag the ones that can "
                f"change behavior if fixed carelessly. Do not patch.")
        kind = "report"
    call = {"name": "run_cmd", "arguments": {"command": f"cargo clippy -p parser -- -D warnings 2>&1 | head -{num(rng, 60, 100)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_clippy"}


def anvil_rg(rng, n):
    ident = pick(rng, GO_IDS + ["ErrClosed", "MaxFrameLen", "defaultTimeout", "route.Table", "conn.Pool", "metrics.Inc"])
    lines = []
    for _ in range(max(5, n)):
        f = gopath(rng)
        ctx = pick(rng, [f"\tif err := {ident}(ctx, addr); err != nil {{", f"\treturn {ident}", f"\tvar _ = {ident}",
                         f"\tt.Run(\"{ident}\", func(t *testing.T) {{", f"\tif n > {ident} {{", f"\tp := {ident}.New({pick(rng, ['cfg', '8', 'timeout'])})",
                         f"\t{ident}.WithLabelValues(\"{pick(rng, ['5xx', 'reload', 'dial'])}\").Inc()", f"// {ident} is the upper bound of a frame."])
        lines.append(f"{f}:{num(rng, 5, 400)}:{ctx}")
    lines.sort()
    v = rng.randrange(3)
    if v == 0:
        task = (f"Trace every use of `{ident}` in the Go code. Print the definition first, then say for each call site whether it handles the "
                f"error or the bound correctly. Numbered findings with file and line.")
        kind = "call"
    elif v == 1:
        task = (f"Count the references to `{ident}` per package and say which packages would break if its signature gained a context argument. "
                f"Answer with a table. Do not patch.")
        kind = "report"
    else:
        task = (f"Rename `{ident}` to `{ident[0].lower() + ident[1:] + 'V2' if ident[0].isupper() else ident + 'V2'}` across internal/. Start with the definition: "
                f"print it, then patch it, then patch each call site. Build after the last patch.")
        kind = "call"
    call = {"name": "rg", "arguments": {"pattern": r"\b" + ident.replace(".", r"\.") + r"\b", "glob": "internal/**/*.go", "max_matches": 200}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_rg"}


def anvil_vet_race(rng, n):
    pkg = pick(rng, ["conn", "upstream", "ratelimit", "buffer", "health"])
    lines = ["==================", "WARNING: DATA RACE"]
    f = gopath(rng, pkg)
    lines += [f"Write at 0x00c000{sha(rng)[:6]} by goroutine {num(rng, 7, 40)}:", f"  harbor/internal/{pkg}.(*{pick(rng, ['Pool', 'Limiter', 'Checker', 'Buffer'])}).{pick(rng, ['Put', 'Allow', 'mark', 'grow', 'reset'])}()",
              f"      /work/harbor/{f}:{num(rng, 20, 300)} +0x{sha(rng)[:3]}", f"  harbor/internal/{pkg}.{pick(rng, GO_TESTS)}.func1()",
              f"      /work/harbor/{f.replace('.go', '_test.go')}:{num(rng, 20, 200)} +0x{sha(rng)[:3]}", "",
              f"Previous read at 0x00c000{sha(rng)[:6]} by goroutine {num(rng, 7, 40)}:", f"  harbor/internal/{pkg}.(*{pick(rng, ['Pool', 'Limiter', 'Checker', 'Buffer'])}).{pick(rng, ['Get', 'Len', 'status', 'Bytes'])}()",
              f"      /work/harbor/{f}:{num(rng, 20, 300)} +0x{sha(rng)[:3]}", f"  harbor/internal/{pkg}.{pick(rng, GO_TESTS)}()",
              f"      /work/harbor/{f.replace('.go', '_test.go')}:{num(rng, 20, 200)} +0x{sha(rng)[:3]}", "", f"Goroutine {num(rng, 7, 40)} (running) created at:",
              f"  harbor/internal/{pkg}.{pick(rng, GO_TESTS)}()", f"      /work/harbor/{f.replace('.go', '_test.go')}:{num(rng, 20, 200)} +0x{sha(rng)[:3]}", "=================="]
    for _ in range(max(0, n // 6 - 1)):
        lines += [f"--- FAIL: {pick(rng, GO_TESTS)} ({num(rng, 0, 2)}.{num(rng, 10, 99)}s)", "    testing.go:1490: race detected during execution of test"]
    lines += ["FAIL", f"FAIL\tharbor/internal/{pkg}\t{num(rng, 1, 9)}.{num(rng, 100, 999)}s", "FAIL"]
    if n > 10:
        lines += ["", f"$ go vet ./internal/{pkg}/", f"{f}:{num(rng, 20, 300)}:{num(rng, 2, 20)}: {pick(rng, ['loopclosure: loop variable c captured by func literal', 'copylocks: assignment copies lock value to p2: internal/' + pkg + '.Pool contains sync.Mutex', 'lostcancel: the cancel function returned by context.WithTimeout should be called, not discarded, to avoid a context leak', 'printf: Errorf format %d has arg addr of wrong type string'])}"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"The race detector flags internal/{pkg} in {pick(rng, ['the nightly run', 'CI job ' + str(num(rng, 1000, 9999)), 'the -race gate', 'the last three runs'])}. Print the two functions in the report, find the shared field, and patch the code to "
                f"guard it with the existing mutex. Rerun `go test ./internal/{pkg}/ -race -count=1`.")
        kind = "call"
    else:
        task = (f"Explain the data race in internal/{pkg} ({pick(rng, ['seen once', 'seen ' + str(num(rng, 2, 9)) + ' times this week', 'new since the pool change', 'flaky on the CI runner'])}) in plain terms: which field, which two code paths, and why the test exposes it. "
                f"Propose the smallest fix as a diff and name any risk of a deadlock. Do not patch.")
        kind = "report"
    call = {"name": "run_cmd", "arguments": {"command": f"go test ./internal/{pkg}/ -race -count=1 2>&1 | head -{num(rng, 40, 70)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_race"}


def anvil_git_diff(rng, n):
    pkg = pick(rng, GO_PKGS)
    f = gopath(rng, pkg)
    idf = pick(rng, GO_IDS)
    lines = [f"diff --git a/{f} b/{f}", f"index {sha(rng)}..{sha(rng)} 100644", f"--- a/{f}", f"+++ b/{f}"]
    for _ in range(max(1, n // 8)):
        a = num(rng, 20, 300)
        lines += [f"@@ -{a},{num(rng, 7, 9)} +{a},{num(rng, 8, 12)} @@ func {idf}({pick(rng, ['ctx context.Context, addr string', 'b []byte', 'c *Conn', 'r *Route, path string'])}) {pick(rng, ['(*Conn, error)', '(*Frame, error)', 'error', 'bool'])} {{",
                  f" \t{pick(rng, ['c.mu.Lock()', 'if len(b) < 4 {', 'd := net.Dialer{}', 'for _, p := range r.prefixes {'])}",
                  f"-\t{pick(rng, ['defer c.mu.Unlock()', '\treturn nil, ErrShort', 'conn, err := d.Dial(\"tcp\", addr)', '\tif strings.HasPrefix(path, p) {'])}",
                  f"+\t{pick(rng, ['// unlock happens in close()', '\treturn nil, fmt.Errorf(\"frame too short: %d bytes\", len(b))', 'conn, err := d.DialContext(ctx, \"tcp\", addr)', '\tif path == p || strings.HasPrefix(path, p+\"/\") {'])}",
                  f" \t{pick(rng, ['if c.closed {', '}', 'if err != nil {', '\t\treturn true'])}",
                  f"+\t\t{pick(rng, ['c.mu.Unlock()', 'n := binary.BigEndian.Uint32(b)', 'return nil, err', 'log.Printf(\"matched %s\", p)'])}",
                  f"+\t\t{pick(rng, ['return ErrClosed', 'if int(n) > len(b) {', '}', '// TODO: metrics'])}",
                  f" \t{pick(rng, ['}', '\treturn nil, ErrOverflow', 'return &Conn{c: conn, timeout: c.timeout}, nil', '}'])}"]
    if n > 12:
        t = f.replace(".go", "_test.go") if not f.endswith("_test.go") else f
        lines += [f"diff --git a/{t} b/{t}", f"index {sha(rng)}..{sha(rng)} 100644", f"--- a/{t}", f"+++ b/{t}", f"@@ -{num(rng, 10, 100)},3 +{num(rng, 10, 100)},{num(rng, 8, 14)} @@",
                  f" func {pick(rng, GO_TESTS)}(t *testing.T) {{", "+\tt.Parallel()", f"+\t{pick(rng, ['c := newConn(t)', 'r := NewTable()', 'b := make([]byte, 3)'])}",
                  f"+\t{pick(rng, ['if err := c.Close(); err != nil {', 'r.Add(\"/api\", nil)', '_, err := parseFrame(b)'])}",
                  f"+\t\t{pick(rng, ['t.Fatal(err)', 'if !r.Match(\"/api/v1\") { t.Fatal(\"no match\") }', 'if err == nil { t.Fatal(\"want error\") }'])}", "+\t}", " }"]
    branch = f"{pick(rng, NAMES)}/{pick(rng, ['unlock-order', 'frame-len-check', 'dial-context', 'prefix-boundary'])}"
    v = rng.randrange(2)
    if v == 0:
        task = (f"Review the diff on {branch} for internal/{pkg}. List the behavior changes, any lock or error handling mistake, and whether the "
                f"added test covers the change. Numbered findings with file and line. Do not patch.")
        kind = "report"
    else:
        task = (f"The diff on {branch} moves an unlock. Print the whole function after the change and decide whether every return path unlocks. "
                f"Patch it if a path is missing, then run the package tests with -race.")
        kind = "call"
    call = {"name": "run_cmd", "arguments": {"command": f"git diff main...{branch} -- internal/{pkg}/ 2>&1 | head -{num(rng, 60, 120)}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "anvil_git_diff"}


ANVIL = [anvil_go_test, anvil_cargo_test, anvil_go_build, anvil_clippy, anvil_rg, anvil_vet_race, anvil_git_diff]
