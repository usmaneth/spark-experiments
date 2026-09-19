#!/usr/bin/env python3
"""tools_catalog.py - the tool catalog and the request templates for build_prompts_tools.py.

Every tool has a JSON-schema parameter block and a list of request templates. A template is
a callable that takes a random.Random and returns one user request whose natural answer is
one call of that tool. The MIXED_NOCALL and MIXED_CLARIFY lists hold requests where the
natural answer is a normal reply or a clarifying question.

All text is original and ASCII only.
"""
import random

# --------------------------------------------------------------------------- fillers
CITIES = ["Lyon", "Osaka", "Denver", "Lagos", "Porto", "Austin", "Delhi", "Seoul", "Perth", "Quito",
          "Bergen", "Cusco", "Hanoi", "Malmo", "Nairobi", "Tbilisi", "Reykjavik", "Valencia", "Calgary",
          "Adelaide", "Bologna", "Krakow", "Santiago", "Kyoto", "Tallinn", "Marrakesh", "Zurich", "Dublin",
          "Helsinki", "Montreal", "Cape Town", "Ljubljana", "Antwerp", "Sapporo", "Bilbao"]
COUNTRIES = {"Lyon": "FR", "Osaka": "JP", "Denver": "US", "Lagos": "NG", "Porto": "PT", "Austin": "US",
             "Delhi": "IN", "Seoul": "KR", "Perth": "AU", "Quito": "EC", "Bergen": "NO", "Cusco": "PE",
             "Hanoi": "VN", "Malmo": "SE", "Nairobi": "KE", "Tbilisi": "GE", "Reykjavik": "IS",
             "Valencia": "ES", "Calgary": "CA", "Adelaide": "AU", "Bologna": "IT", "Krakow": "PL",
             "Santiago": "CL", "Kyoto": "JP", "Tallinn": "EE", "Marrakesh": "MA", "Zurich": "CH",
             "Dublin": "IE", "Helsinki": "FI", "Montreal": "CA", "Cape Town": "ZA", "Ljubljana": "SI",
             "Antwerp": "BE", "Sapporo": "JP", "Bilbao": "ES"}
NAMES = ["Alice", "Ben", "Chloe", "Dan", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kira", "Leo",
         "Maya", "Noor", "Omar", "Priya", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yara", "Zane",
         "Bea", "Cyrus", "Dalia", "Eyal", "Freya", "Gus", "Hana", "Ivo", "Juno", "Kofi", "Lena", "Milo"]
DOMAINS = ["northwind.dev", "acme-labs.io", "example.org", "brightpath.co", "tinfoil.app", "quartz.team",
           "lumen.systems", "harbor-eng.net"]
TEAMS = ["web", "mobile", "backend", "data", "support", "platform", "infra", "payments", "search", "growth"]
SERVICES = ["api-gateway", "auth-service", "billing-worker", "ingest", "notifier", "search-indexer",
            "checkout", "image-resizer", "webhook-relay", "report-builder", "session-store", "rate-limiter",
            "export-service", "ledger", "scheduler", "media-proxy", "feature-flags", "geo-lookup"]
NAMESPACES = ["prod", "staging", "dev", "payments", "data-platform", "monitoring", "edge", "batch"]
BUCKETS = ["acme-backups", "media-uploads-prod", "raw-events", "invoice-archive", "ml-artifacts",
           "static-assets", "log-export", "customer-exports", "nightly-snapshots"]
TABLES = {"orders": "id, customer_id, total, status, created_at",
          "customers": "id, name, email, country, created_at",
          "invoices": "id, customer_id, amount, due_date, paid_at",
          "sessions": "id, user_id, started_at, duration_s, device",
          "events": "id, user_id, name, payload, ts",
          "subscriptions": "id, customer_id, plan, status, renews_at",
          "tickets": "id, team, priority, status, opened_at, closed_at",
          "deployments": "id, service, version, env, deployed_at, ok",
          "products": "id, sku, name, price, stock",
          "shipments": "id, order_id, carrier, shipped_at, delivered_at",
          "page_views": "id, path, user_id, ts, referrer",
          "refunds": "id, order_id, amount, reason, created_at"}
FILE_PATHS = ["notes/meeting-2026-09-12.md", "src/config/settings.py", "docs/runbook.md", "README.md",
              "logs/app.log", "data/customers.csv", "scripts/deploy.sh", "infra/main.tf", ".env.example",
              "reports/q3-summary.md", "src/api/routes.py", "package.json", "Makefile", "docs/CHANGELOG.md",
              "config/nginx.conf", "tests/test_billing.py", "data/2026-09-events.jsonl", "src/lib/retry.ts",
              "pyproject.toml", "docker-compose.yml", "k8s/deployment.yaml", "notes/todo.txt",
              "src/worker/queue.go", "cmd/server/main.go", "app/models/user.rb", "etc/cron.d/backup"]
DIRS = ["src", "docs", "tests", "scripts", "data", "logs", "infra", "notes", "reports", "build",
        "config", "assets", "migrations", "tmp", "vendor", "k8s", "backups", "src/api", "src/lib"]
REPOS = ["acme/billing", "acme/webapp", "northwind/ingest", "brightpath/mobile", "acme/infra",
         "quartz/search", "harbor/edge-proxy", "lumen/data-tools", "tinfoil/cli", "acme/docs"]
LANGS = ["Spanish", "French", "German", "Japanese", "Portuguese", "Italian", "Korean", "Dutch", "Polish",
         "Turkish", "Swedish", "Arabic", "Hindi", "Vietnamese"]
LANG_CODES = {"Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja", "Portuguese": "pt",
              "Italian": "it", "Korean": "ko", "Dutch": "nl", "Polish": "pl", "Turkish": "tr", "Swedish": "sv",
              "Arabic": "ar", "Hindi": "hi", "Vietnamese": "vi"}
PACKAGES = {"pip": ["requests", "httpx", "pydantic", "numpy", "polars", "rich", "typer", "fastapi", "sqlalchemy",
                    "pytest-xdist", "black", "ruff", "pyyaml", "boto3", "orjson"],
            "npm": ["zod", "vitest", "eslint", "prettier", "express", "fastify", "dayjs", "commander", "chalk",
                    "axios", "typescript", "tsx", "pino", "dotenv"],
            "cargo": ["serde", "tokio", "clap", "anyhow", "reqwest", "rayon", "regex", "tracing"],
            "apt": ["jq", "ripgrep", "htop", "tmux", "curl", "shellcheck", "sqlite3", "fd-find"],
            "brew": ["jq", "ripgrep", "gh", "fzf", "bat", "shellcheck", "tmux", "hyperfine"]}
SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "SHOP", "ASML", "SAP", "TSM"]
METRICS = ["p95 latency", "error rate", "requests per second", "cpu usage", "memory usage", "queue depth",
           "cache hit ratio", "5xx count", "open connections", "disk io wait"]
TIMEZONES = ["Asia/Tokyo", "Europe/Berlin", "America/Denver", "Australia/Sydney", "Africa/Nairobi",
             "America/Sao_Paulo", "Asia/Kolkata", "Europe/London", "Pacific/Auckland", "America/Toronto"]
URLS = ["https://api.northwind.dev/v2/status", "https://status.acme-labs.io/api/incidents",
        "https://example.org/reports/latest.json", "https://brightpath.co/blog/postgres-tuning",
        "https://quartz.team/docs/rate-limits", "https://lumen.systems/changelog",
        "https://harbor-eng.net/api/health", "https://tinfoil.app/pricing",
        "https://api.acme-labs.io/v1/orders?status=open", "https://docs.example.org/guides/webhooks"]
EMAIL_SUBJECTS = ["Q3 roadmap review", "invoice #4471 overdue", "on-call handover", "release 2.8 notes",
                  "vendor contract renewal", "office move logistics", "load test results", "design review notes",
                  "customer escalation: Flux Ltd", "hiring loop feedback", "postmortem draft", "data retention policy"]
TICKET_ISSUES = [
    ("checkout button does nothing on Safari 17", "web"),
    ("push notifications arrive twice on Android 15", "mobile"),
    ("nightly export job fails with a timeout after 30 minutes", "data"),
    ("password reset email lands in spam for gmail users", "backend"),
    ("search results ignore the language filter", "search"),
    ("invoice PDF shows the wrong currency symbol for CHF", "payments"),
    ("dashboard chart overlaps the legend on 13 inch screens", "web"),
    ("api returns 502 for uploads larger than 25 MB", "platform"),
    ("dark mode toggle resets after logout", "mobile"),
    ("customer import drops rows with a trailing comma", "data"),
    ("rate limiter counts health checks as user traffic", "infra"),
    ("signup form accepts an empty country field", "growth"),
    ("webhook retries stop after one attempt", "backend"),
    ("csv export truncates addresses longer than 80 characters", "support"),
    ("staging deploy takes 40 minutes since Monday", "infra"),
    ("session expires while a form is still open", "web"),
]
UNITS = [("kilometers", "miles", 42), ("miles", "kilometers", 26.2), ("celsius", "fahrenheit", 37.5),
         ("fahrenheit", "celsius", 98.6), ("kilograms", "pounds", 72), ("pounds", "kilograms", 185),
         ("liters", "gallons", 50), ("gallons", "liters", 12), ("meters", "feet", 1800), ("feet", "meters", 5280),
         ("inches", "centimeters", 27), ("centimeters", "inches", 180), ("kilowatt hours", "megajoules", 350),
         ("hectares", "acres", 12.5), ("knots", "kilometers per hour", 18), ("psi", "bar", 32),
         ("ounces", "grams", 16), ("cups", "milliliters", 2.5)]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "SEK", "INR", "BRL", "KRW", "MXN"]
CRONS = ["0 3 * * *", "*/15 * * * *", "30 6 * * 1", "0 */4 * * *", "45 23 * * 5", "0 0 1 * *", "15 8 * * 1-5"]
DEPLOYMENTS = ["web-frontend", "api", "worker", "cron-runner", "search-indexer", "auth", "gateway", "ingest",
               "billing", "notifier", "image-resizer", "report-builder"]
LOG_WORDS = ["timeout", "connection reset", "OOMKilled", "permission denied", "deadlock", "rate limited",
             "certificate expired", "disk full", "segfault", "checksum mismatch"]


def pick(rng, xs):
    return rng.choice(xs)


def day(rng):
    return rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next Monday", "tomorrow",
                       "next Thursday", "Saturday", "Sunday"])


def clock(rng):
    h = rng.randint(7, 18)
    m = rng.choice(["00", "15", "30", "45"])
    return f"{h}:{m}"


def date_iso(rng):
    return f"2026-{rng.randint(9, 12):02d}-{rng.randint(1, 28):02d}"


def email(rng, name=None):
    n = (name or pick(rng, NAMES)).lower()
    return f"{n}@{pick(rng, DOMAINS)}"


def num(rng, lo, hi):
    return rng.randint(lo, hi)


# --------------------------------------------------------------------------- tool definitions
def T(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def S(desc, **kw):
    d = {"type": "string", "description": desc}
    d.update(kw)
    return d


def I(desc, **kw):
    d = {"type": "integer", "description": desc}
    d.update(kw)
    return d


def N(desc, **kw):
    d = {"type": "number", "description": desc}
    d.update(kw)
    return d


def B(desc):
    return {"type": "boolean", "description": desc}


def A(desc, item="string"):
    return {"type": "array", "items": {"type": item}, "description": desc}


TOOLS = {}
REQUESTS = {}


def tool(defn, templates):
    name = defn["function"]["name"]
    TOOLS[name] = defn
    REQUESTS[name] = templates
    return defn


# weather --------------------------------------------------------------------
tool(T("get_weather", "Get the current weather for a city.",
       {"city": S("City name, for example Lisbon"), "country": S("Two letter country code, optional"),
        "units": S("Temperature unit, default celsius", enum=["celsius", "fahrenheit"])}, ["city"]),
     [lambda r: f"Is it raining in {pick(r, CITIES)} at the moment? I want to know if I need an umbrella for the walk to the station.",
      lambda r: f"How warm is it in {pick(r, CITIES)} right now, in fahrenheit? My cousin lives there and I want to know if it is shorts weather.",
      lambda r: (lambda c: f"Give me the current conditions in {c}, {COUNTRIES[c]} - temperature, wind and humidity.")(pick(r, CITIES)),
      lambda r: f"What is the temperature outside in {pick(r, CITIES)} now? I am about to go for a run and want to pick the right layers.",
      lambda r: f"My flight lands in {pick(r, CITIES)} in an hour. What is the weather there now?",
      lambda r: f"Check whether it is windy in {pick(r, CITIES)} right now. We plan to fly a drone this afternoon."])

tool(T("get_forecast", "Get the daily weather forecast for a city for the next days.",
       {"city": S("City name"), "days": I("Number of days, 1 to 10", minimum=1, maximum=10),
        "units": S("Temperature unit", enum=["celsius", "fahrenheit"])}, ["city", "days"]),
     [lambda r: f"What does the {num(r, 3, 7)} day forecast look like for {pick(r, CITIES)}? We are camping there this weekend.",
      lambda r: f"Will it rain in {pick(r, CITIES)} over the next {num(r, 2, 5)} days? I need to plan when to paint the fence.",
      lambda r: f"Give me the forecast for {pick(r, CITIES)} for the coming week in fahrenheit.",
      lambda r: f"Is there any snow expected in {pick(r, CITIES)} in the next {num(r, 3, 10)} days? I am deciding on winter tyres.",
      lambda r: f"Show me the {num(r, 2, 4)} day outlook for {pick(r, CITIES)} so I can pick a day for the outdoor wedding photos."])

# docs / search ---------------------------------------------------------------
tool(T("search_docs", "Search the internal documentation and return matching pages with a short excerpt.",
       {"query": S("Free text search query"),
        "space": S("Documentation space to search, default all", enum=["engineering", "billing", "support", "all"]),
        "limit": I("Maximum number of results, default 5", minimum=1, maximum=20)}, ["query"]),
     [lambda r: f"Look in the {pick(r, ['engineering', 'billing', 'support'])} docs for how we {pick(r, ['rotate database credentials', 'issue a partial refund', 'onboard a new vendor', 'roll back a failed deploy', 'handle a chargeback', 'add a feature flag', 'archive a workspace', 'set up a sandbox tenant'])} and give me the gist.",
      lambda r: f"Where is the internal page about {pick(r, ['the incident severity levels', 'the retry policy for webhooks', 'the pricing tiers', 'the data retention schedule', 'the release checklist', 'sla credits', 'the api key rotation procedure', 'the staging environment reset'])}? Return the top {num(r, 2, 6)} hits.",
      lambda r: f"Search our documentation for '{pick(r, ['tenant migration', 'rate limit override', 'invoice reissue', 'ssh bastion access', 'canary rollout', 'pagerduty rotation', 'gdpr export', 'oauth scopes'])}' and summarize the first result.",
      lambda r: f"Find the support article that explains how a customer can {pick(r, ['change their billing email', 'download past invoices', 'transfer ownership of a project', 'enable two factor auth', 'restore a deleted board', 'cancel a trial'])}.",
      lambda r: f"Pull up whatever the engineering handbook says about {pick(r, ['branch naming', 'schema migrations', 'secrets in ci', 'flaky test quarantine', 'on-call handoff', 'log retention'])}. I need the exact rule, not a paraphrase."])

tool(T("web_search", "Search the public web and return titles, urls and snippets.",
       {"query": S("Search query"), "num_results": I("Number of results, default 5", minimum=1, maximum=20),
        "recency": S("Restrict to recent results", enum=["day", "week", "month", "year", "any"])}, ["query"]),
     [lambda r: f"Search the web for the latest news about {pick(r, ['the postgres 18 release', 'the sqlite wasm build', 'rust 2024 edition adoption', 'the eu ai act timeline', 'llama.cpp speculative decoding', 'the kubernetes 1.35 release', 'usb4 version 2 laptops', 'the nvidia gb10 desktop'])} from the last week.",
      lambda r: f"Find {num(r, 3, 8)} recent articles that compare {pick(r, ['duckdb and clickhouse', 'bun and deno', 'terraform and pulumi', 'argo cd and flux', 'litestream and rqlite', 'grpc and connect', 'pnpm and yarn berry'])}.",
      lambda r: f"What are people saying online about {pick(r, ['the new macbook thermal throttling', 'the rpi 5 pcie hat', 'the framework laptop 16 gpu', 'nix flakes stability', 'the go 1.25 gc changes'])}? Search and give me the highlights.",
      lambda r: f"Do a web search for '{pick(r, ['zsh completion slow startup', 'nginx 499 status meaning', 'docker buildx cache github actions', 'systemd timer vs cron', 'psql copy csv header', 'jq group_by sum'])}' and tell me the top result.",
      lambda r: f"Search for the official release notes of {pick(r, ['python 3.14', 'node 24', 'go 1.25', 'rust 1.90', 'postgres 18', 'redis 8', 'kubernetes 1.34', 'terraform 1.13'])} and list the headline changes."])

tool(T("summarize_url", "Fetch a web page and return a summary of its main content.",
       {"url": S("Page url"), "max_sentences": I("Summary length in sentences, default 5", minimum=1, maximum=20),
        "language": S("Summary language code, default en")}, ["url"]),
     [lambda r: f"Summarize {pick(r, URLS)} in {num(r, 3, 8)} sentences.",
      lambda r: f"What is on this page? {pick(r, URLS)} - give me the short version.",
      lambda r: f"Read {pick(r, URLS)} and give me a {num(r, 2, 6)} sentence summary in {pick(r, LANGS)}.",
      lambda r: f"I do not have time to read {pick(r, URLS)}. Boil it down to the key points."])

# sql -------------------------------------------------------------------------
tool(T("run_sql", "Run a read-only SQL query against the analytics database and return the rows.",
       {"query": S("A single SELECT statement"), "database": S("Database to query, default analytics", enum=["analytics", "billing"]),
        "row_limit": I("Maximum rows to return, default 100", minimum=1, maximum=1000)}, ["query"]),
     [lambda r: (lambda t: f"How many rows in the {t} table have status = '{pick(r, ['failed', 'pending', 'cancelled', 'open', 'shipped'])}'? Columns: {TABLES[t]}.")(pick(r, ['orders', 'tickets', 'subscriptions', 'deployments'])),
      lambda r: (lambda t: f"Give me the top {num(r, 3, 15)} {t} by {pick(r, ['total', 'amount', 'duration_s', 'price'])} from last month. The {t} table has columns {TABLES[t]}.")(pick(r, ['orders', 'invoices', 'sessions', 'products'])),
      lambda r: f"What was the average order total per country in {pick(r, ['August', 'July', 'Q2', 'the last 30 days'])}? Tables: orders({TABLES['orders']}) and customers({TABLES['customers']}).",
      lambda r: (lambda t: f"Count the {t} per day for the last {num(r, 7, 30)} days on the {pick(r, ['analytics', 'billing'])} database. Columns of {t}: {TABLES[t]}.")(pick(r, ['events', 'page_views', 'refunds', 'shipments'])),
      lambda r: f"Which {num(r, 5, 20)} customers opened the most tickets this year? Use customers({TABLES['customers']}) and tickets({TABLES['tickets']}); tickets.customer_id links them.",
      lambda r: f"List every deployment of {pick(r, SERVICES)} to prod in the last {num(r, 7, 60)} days with its version and whether it succeeded. Table deployments: {TABLES['deployments']}.",
      lambda r: (lambda t: f"Find the {t} rows created between {date_iso(r)} and {date_iso(r)} that have no {pick(r, ['paid_at', 'delivered_at', 'closed_at'])}. Columns: {TABLES[t]}.")(pick(r, ['invoices', 'shipments', 'tickets']))])

tool(T("read_spreadsheet", "Read a range of cells from a spreadsheet file and return the rows.",
       {"file": S("Path or id of the spreadsheet"), "sheet": S("Sheet name, default the first sheet"),
        "range": S("A1 style range, for example A1:D50"), "header": B("Treat the first row as a header")}, ["file"]),
     [lambda r: f"Open {pick(r, ['budget-2026.xlsx', 'headcount.xlsx', 'vendor-list.xlsx', 'inventory.xlsx', 'okrs-q4.xlsx'])} and read the first {num(r, 20, 100)} rows of the {pick(r, ['Summary', 'Raw', 'Q4', 'Totals', 'Sheet1'])} sheet.",
      lambda r: f"What is in cells A1 to {pick(r, ['F', 'H', 'D'])}{num(r, 10, 60)} of the {pick(r, ['Forecast', 'Actuals', 'Pipeline'])} tab in {pick(r, ['sales-tracker.xlsx', 'renewals.xlsx', 'capacity.xlsx'])}?",
      lambda r: f"Load the {pick(r, ['Costs', 'Usage', 'Churn'])} sheet from {pick(r, ['finance/monthly.xlsx', 'ops/usage-2026-08.xlsx', 'cs/churn.xlsx'])} with the header row and show me the rows."])

# ticketing -------------------------------------------------------------------
tool(T("create_ticket", "Create a ticket in the issue tracker and return its id.",
       {"title": S("Short ticket title"), "body": S("Ticket description in markdown"),
        "priority": S("Priority", enum=["low", "medium", "high", "urgent"]),
        "team": S("Team that owns the ticket", enum=TEAMS), "labels": A("Optional labels")}, ["title", "body", "priority"]),
     [lambda r: (lambda iss: f"Raise a {pick(r, ['medium', 'high', 'low'])} priority ticket for the {iss[1]} team: {iss[0]}. Add what a reproduction would look like.")(pick(r, TICKET_ISSUES)),
      lambda r: (lambda iss: f"Log a bug: {iss[0]}. Assign it to {iss[1]}, priority {pick(r, ['low', 'medium'])}, label it {pick(r, ['regression', 'ux', 'perf', 'flaky', 'customer-reported'])}.")(pick(r, TICKET_ISSUES)),
      lambda r: (lambda iss: f"Please create an urgent ticket. {iss[0][0].upper() + iss[0][1:]}. Three customers reported it in the last hour. Owner team is {iss[1]}.")(pick(r, TICKET_ISSUES)),
      lambda r: f"Open a {pick(r, ['low', 'medium'])} priority task for {pick(r, TEAMS)}: {pick(r, ['upgrade the base image to debian 13', 'add retries to the export job', 'document the new env vars', 'remove the deprecated v1 endpoints', 'add a health check to the notifier', 'rotate the staging tls cert'])}. Label it {pick(r, ['chore', 'tech-debt', 'docs', 'security'])}.",
      lambda r: (lambda iss: f"File this as a ticket with priority {pick(r, ['high', 'urgent'])} for {iss[1]}: {iss[0]}. Note in the body that it started after release {num(r, 2, 9)}.{num(r, 0, 12)}.")(pick(r, TICKET_ISSUES))])

tool(T("update_ticket", "Change the status, assignee or add a comment on an existing ticket.",
       {"ticket_id": S("Ticket id, for example WEB-1234"), "status": S("New status", enum=["open", "in_progress", "blocked", "done", "wont_fix"]),
        "assignee": S("User name of the new assignee"), "comment": S("Comment to add"), "priority": S("New priority", enum=["low", "medium", "high", "urgent"])}, ["ticket_id"]),
     [lambda r: f"Move {pick(r, TEAMS).upper()}-{num(r, 100, 4999)} to {pick(r, ['in progress', 'blocked', 'done'])} and leave a comment that says {pick(r, ['the fix is on the release branch', 'waiting on the vendor', 'verified on staging', 'cannot reproduce on 2.8'])}.",
      lambda r: f"Assign ticket {pick(r, TEAMS).upper()}-{num(r, 100, 4999)} to {pick(r, NAMES).lower()} and bump the priority to {pick(r, ['high', 'urgent'])}.",
      lambda r: f"Close {pick(r, TEAMS).upper()}-{num(r, 100, 4999)} as {pick(r, ['done', 'wont fix'])}. Add a note: {pick(r, ['duplicate of an older report', 'shipped in 2.9.1', 'the behavior is by design', 'fixed by the config change on Friday'])}.",
      lambda r: f"Add a comment to {pick(r, TEAMS).upper()}-{num(r, 100, 4999)}: '{pick(r, ['repro steps attached, happens only with the beta flag', 'customer confirmed the workaround works', 'blocked on the schema migration', 'needs design input before we continue'])}'."])

# calendar / email ------------------------------------------------------------
tool(T("list_events", "List calendar events in a time window.",
       {"calendar": S("Calendar name, default primary"), "start": S("Window start, ISO 8601"),
        "end": S("Window end, ISO 8601"), "query": S("Optional text filter on the title")}, ["start", "end"]),
     [lambda r: f"What is on my calendar {pick(r, ['tomorrow', 'on Thursday', 'next Monday', 'this Friday afternoon', 'on ' + date_iso(r)])}?",
      lambda r: f"Do I have any meetings between {clock(r)} and {clock(r)} {day(r)}? I want to book a dentist slot.",
      lambda r: f"List everything on the {pick(r, ['team', 'oncall', 'releases', 'interviews'])} calendar for the week of {date_iso(r)}.",
      lambda r: f"Am I free {day(r)} at {clock(r)}? Check my calendar for that hour.",
      lambda r: f"Show my events for the next {num(r, 2, 5)} days that mention '{pick(r, ['review', 'standup', 'retro', 'planning', 'customer', 'interview'])}'."])

tool(T("create_event", "Create a calendar event and invite attendees.",
       {"title": S("Event title"), "start": S("Start time, ISO 8601"), "end": S("End time, ISO 8601"),
        "attendees": A("Email addresses to invite"), "location": S("Room or link"), "description": S("Notes")}, ["title", "start", "end"]),
     [lambda r: f"Book a {num(r, 30, 60)} minute {pick(r, ['sync', 'design review', 'retro', '1:1', 'planning session', 'demo'])} with {email(r)} {day(r)} at {clock(r)}.",
      lambda r: f"Put '{pick(r, ['dentist', 'car service', 'parent teacher meeting', 'flu shot', 'passport renewal'])}' on my calendar for {date_iso(r)} from {clock(r)} to {clock(r)}.",
      lambda r: f"Schedule the {pick(r, ['release go/no-go', 'incident review', 'roadmap workshop', 'vendor call'])} for {day(r)} {clock(r)}, one hour, in {pick(r, ['room Bergen', 'the big meeting room', 'room 4B', 'the boardroom'])}, and invite {email(r)} and {email(r)}.",
      lambda r: f"Add a recurring-looking placeholder: '{pick(r, ['focus time', 'lunch walk', 'inbox zero', 'reading hour'])}' {day(r)} {clock(r)} for 45 minutes. Just this one instance is fine.",
      lambda r: f"Create an event named '{pick(r, ['Q4 kickoff', 'security training', 'all hands', 'launch party'])}' on {date_iso(r)} at {clock(r)} for {num(r, 1, 3)} hours with {email(r)} invited."])

tool(T("send_email", "Send an email from the user's account.",
       {"to": A("Recipient addresses"), "subject": S("Subject line"), "body": S("Plain text body"),
        "cc": A("Cc addresses"), "attachments": A("File paths to attach")}, ["to", "subject", "body"]),
     [lambda r: f"Email {email(r)} with subject '{pick(r, EMAIL_SUBJECTS)}' and tell them {pick(r, ['the meeting moved to ' + clock(r), 'the numbers are in the shared folder', 'I will be out on ' + day(r), 'the draft is ready for review', 'we approved the budget'])}.",
      lambda r: f"Send a short note to {email(r)} and cc {email(r)}: the {pick(r, ['deploy', 'migration', 'audit', 'load test'])} is done and {pick(r, ['nothing broke', 'two alerts fired but cleared', 'we need a follow up on Monday'])}. Subject: {pick(r, EMAIL_SUBJECTS)}.",
      lambda r: f"Write to {pick(r, NAMES)} at {email(r)} to confirm {pick(r, ['the interview on ' + day(r), 'the delivery address', 'the invoice amount of ' + str(num(r, 200, 9000)) + ' EUR', 'the new start date'])}. Keep it to three sentences.",
      lambda r: f"Mail the file {pick(r, FILE_PATHS)} to {email(r)} with the subject '{pick(r, EMAIL_SUBJECTS)}' and a one line body.",
      lambda r: f"Reply to {email(r)} about '{pick(r, EMAIL_SUBJECTS)}': say yes, thank them, and ask for {pick(r, ['the updated timeline', 'the invoice number', 'a copy of the contract', 'the dial-in link'])}."])

tool(T("search_email", "Search the user's mailbox and return matching messages.",
       {"query": S("Search terms"), "folder": S("Mailbox folder", enum=["inbox", "sent", "archive", "all"]),
        "since": S("Only messages after this date, ISO 8601"), "limit": I("Maximum messages, default 10", minimum=1, maximum=50)}, ["query"]),
     [lambda r: f"Find the email from {email(r)} about {pick(r, ['the invoice', 'the contract', 'the offsite', 'the api keys', 'the renewal'])} from the last {num(r, 2, 8)} weeks.",
      lambda r: f"Search my inbox for messages mentioning '{pick(r, EMAIL_SUBJECTS)}' since {date_iso(r)}.",
      lambda r: f"Did anyone send me a {pick(r, ['tracking number', 'boarding pass', 'signed nda', 'purchase order', 'calendar invite for the retro'])} recently? Look through my mail.",
      lambda r: f"Show the last {num(r, 3, 10)} emails I sent to {email(r)}.",
      lambda r: f"Look in the archive for the thread with {pick(r, NAMES)} about {pick(r, ['the salary band', 'the office lease', 'the postgres upgrade', 'the vendor security review'])}."])

tool(T("lookup_contact", "Look up a person in the address book.",
       {"name": S("Full or partial name"), "field": S("Field to return, default all", enum=["email", "phone", "team", "manager", "all"])}, ["name"]),
     [lambda r: f"What is {pick(r, NAMES)}'s phone number?",
      lambda r: f"Who is {pick(r, NAMES)}'s manager? Check the address book.",
      lambda r: f"Find the email address for {pick(r, NAMES)} {pick(r, ['Okafor', 'Lindqvist', 'Haddad', 'Moreau', 'Tanaka', 'Novak', 'Reyes', 'Bauer'])}.",
      lambda r: f"Which team is {pick(r, NAMES)} on? Look them up.",
      lambda r: f"Get me the contact card for {pick(r, NAMES)} from {pick(r, ['finance', 'legal', 'the design team', 'the Berlin office'])}."])

# files -----------------------------------------------------------------------
tool(T("cat_file", "Read a text file and return its content.",
       {"path": S("File path"), "max_lines": I("Return at most this many lines from the start", minimum=1),
        "encoding": S("Text encoding, default utf-8")}, ["path"]),
     [lambda r: f"Show me the contents of {pick(r, FILE_PATHS)}.",
      lambda r: f"Open {pick(r, FILE_PATHS)} and read the first {num(r, 20, 120)} lines.",
      lambda r: f"What does {pick(r, FILE_PATHS)} say? I need to check {pick(r, ['the port number', 'the retry count', 'who owns the module', 'the default timeout', 'the version pin', 'the cron schedule'])}.",
      lambda r: f"Print {pick(r, FILE_PATHS)} so I can see whether the {pick(r, ['debug flag', 'api base url', 'log level', 'feature toggle', 'bucket name'])} is set.",
      lambda r: f"Read the file at {pick(r, FILE_PATHS)} for me."])

tool(T("save_file", "Write text to a file, creating it when needed.",
       {"path": S("File path"), "content": S("Full text to write"), "overwrite": B("Replace an existing file, default false"),
        "append": B("Append instead of replace")}, ["path", "content"]),
     [lambda r: f"Save a file named {pick(r, ['notes/standup.md', 'todo.txt', 'scratch/ideas.md', 'docs/decisions.md', 'notes/ ' + date_iso(r) + '.md']).replace(' ', '')} with these lines: {pick(r, ['1. fix the flaky test 2. ping legal 3. book travel', 'decision: keep postgres, revisit in Q1', 'call the landlord, renew the domain, submit expenses', 'draft the offsite agenda and share by Friday'])}.",
      lambda r: f"Write a hello world {pick(r, ['python', 'bash', 'go', 'node'])} script to {pick(r, ['scripts/hello.py', 'scripts/hello.sh', 'cmd/hello/main.go', 'scripts/hello.js'])}.",
      lambda r: f"Create {pick(r, ['.env.example', 'config/local.yaml', 'CONTRIBUTING.md', 'docs/faq.md'])} with {pick(r, ['the three variables API_URL, API_KEY and LOG_LEVEL with placeholder values', 'a two line header and a todo marker', 'a title and a short paragraph that says the doc is a stub', 'a yaml block with debug true and port 8080'])}.",
      lambda r: f"Append the line '{pick(r, ['- reviewed the pr for the cache module', '- rotated the staging key', '- 2026-09-18: backup verified', '- reminder: renew the certificate'])}' to {pick(r, ['notes/log.md', 'CHANGELOG.md', 'ops/journal.txt'])}.",
      lambda r: f"Overwrite {pick(r, ['README.md', 'docs/status.md', 'notes/plan.md'])} with a short section titled '{pick(r, ['Status', 'Next steps', 'Known issues'])}' and two bullet points about {pick(r, ['the migration', 'the launch', 'the audit', 'the beta'])}."])

tool(T("list_directory", "List files and folders in a directory.",
       {"path": S("Directory path"), "recursive": B("Include subdirectories"), "pattern": S("Glob pattern filter, for example *.log"),
        "show_hidden": B("Include dot files")}, ["path"]),
     [lambda r: f"What files are in {pick(r, DIRS)}/?",
      lambda r: f"List every {pick(r, ['*.log', '*.py', '*.md', '*.yaml', '*.csv', '*.sh', '*.json'])} file under {pick(r, DIRS)}, recursively.",
      lambda r: f"Show me the directory listing of {pick(r, DIRS)} including hidden files.",
      lambda r: f"Are there any {pick(r, ['backup', 'tmp', 'old', 'draft', 'export'])} files left in {pick(r, DIRS)}? List the folder.",
      lambda r: f"List {pick(r, DIRS)}/ so I can see which {pick(r, ['migrations', 'reports', 'snapshots', 'configs', 'scripts'])} exist."])

tool(T("move_file", "Move or rename a file or directory.",
       {"source": S("Existing path"), "destination": S("New path"), "overwrite": B("Replace the destination when it exists")}, ["source", "destination"]),
     [lambda r: f"Rename {pick(r, FILE_PATHS)} to {pick(r, ['archive/old-notes.md', 'docs/runbook-v2.md', 'README.old.md', 'config/settings.bak', 'scripts/deploy-legacy.sh'])}.",
      lambda r: f"Move {pick(r, FILE_PATHS)} into the {pick(r, ['archive', 'backups', 'tmp', 'attic', 'old'])} folder.",
      lambda r: f"Put the {pick(r, ['data/customers.csv', 'reports/q3-summary.md', 'logs/app.log'])} file under {pick(r, DIRS)}/{date_iso(r)}/ - keep the file name.",
      lambda r: f"Rename the directory {pick(r, DIRS)} to {pick(r, ['legacy', 'v1', 'old-src', 'scratch', 'archive-2025'])}."])

tool(T("delete_file", "Delete a file or an empty directory. Use recursive for a directory tree.",
       {"path": S("Path to delete"), "recursive": B("Delete a directory and its content"), "force": B("Ignore missing files")}, ["path"]),
     [lambda r: f"Delete {pick(r, ['tmp/scratch.txt', 'build/output.tar.gz', 'logs/app.log.1', 'data/duplicate.csv', 'notes/old-draft.md', 'dist/bundle.old.js'])}.",
      lambda r: f"Remove the {pick(r, ['build', 'tmp', 'node_modules', '.pytest_cache', 'dist', '__pycache__'])} directory and everything in it.",
      lambda r: f"Get rid of {pick(r, ['core.1234', 'nohup.out', '.DS_Store', 'debug.log', 'a.out'])} in the project root.",
      lambda r: f"Delete the empty folder {pick(r, DIRS)}/{pick(r, ['unused', 'empty', 'old', 'placeholder'])}."])

# shell / git -----------------------------------------------------------------
tool(T("run_command", "Run one shell command and return stdout, stderr and the exit code.",
       {"command": S("Command line"), "cwd": S("Working directory, default the project root"),
        "timeout_s": I("Timeout in seconds, default 60", minimum=1, maximum=600)}, ["command"]),
     [lambda r: f"Run the {pick(r, ['unit tests', 'linter', 'type checker', 'build', 'formatter check'])} and show me the output.",
      lambda r: f"How much disk space is free on this machine? Use df.",
      lambda r: f"Check which version of {pick(r, ['python', 'node', 'go', 'docker', 'terraform', 'kubectl', 'rustc', 'psql'])} is installed here.",
      lambda r: f"Count the lines of code in {pick(r, DIRS)} ({pick(r, ['*.py', '*.go', '*.ts', '*.rs', '*.sh'])} files only).",
      lambda r: f"Show the last {num(r, 20, 100)} lines of {pick(r, ['logs/app.log', '/var/log/syslog', 'logs/worker.log', 'nohup.out'])}.",
      lambda r: f"Run `{pick(r, ['make lint', 'npm run build', 'pytest -x -q', 'go vet ./...', 'cargo check', 'terraform validate', 'docker compose ps', 'uptime'])}` in the project and report the exit code."])

tool(T("git_status", "Show the working tree status of a git repository.",
       {"repo": S("Repository path, default the current one"), "short": B("Use the short format")}, []),
     [lambda r: f"Which files have I changed in {pick(r, ['this repo', 'the current checkout', pick(r, REPOS)])}? Show the git status.",
      lambda r: f"Is my working tree clean? Check git status{pick(r, ['', ' in the short format', ' for ' + pick(r, REPOS)])}.",
      lambda r: f"Are there untracked files in the repo right now?",
      lambda r: f"Before I commit, show me what git thinks is modified."])

tool(T("git_commit", "Stage files and create a commit.",
       {"repo": S("Repository path"), "message": S("Commit message"), "files": A("Paths to stage; empty means all changes"),
        "amend": B("Amend the previous commit")}, ["message"]),
     [lambda r: f"Commit everything with the message '{pick(r, ['Fix the retry backoff in the exporter', 'Add the staging config', 'Update the runbook for the new alert', 'Bump the base image to debian 13', 'Remove the unused cache module', 'Document the release flow'])}'.",
      lambda r: f"Stage {pick(r, FILE_PATHS)} and {pick(r, FILE_PATHS)} and commit as '{pick(r, ['Tidy the config loader', 'Handle the empty response case', 'Rename the worker queue', 'Add a smoke test'])}'.",
      lambda r: f"Make a commit in {pick(r, REPOS)} with the message '{pick(r, ['Pin pyyaml to 6.0.2', 'Refresh the lock file', 'Add the nginx timeout', 'Fix a typo in the FAQ'])}'.",
      lambda r: f"Amend my last commit to include {pick(r, FILE_PATHS)}; keep the message '{pick(r, ['Add the export job', 'Fix the flaky login test', 'Update the deploy script'])}'."])

tool(T("git_log", "Show the commit history of a repository.",
       {"repo": S("Repository path"), "max_count": I("Number of commits, default 20", minimum=1, maximum=500),
        "path": S("Limit to commits that touch this path"), "since": S("Only commits after this date"),
        "author": S("Filter by author")}, []),
     [lambda r: f"Show the last {num(r, 5, 40)} commits in {pick(r, REPOS)}.",
      lambda r: f"Who changed {pick(r, FILE_PATHS)} recently? Show the commit history for that file.",
      lambda r: f"List the commits by {pick(r, NAMES).lower()} since {date_iso(r)}.",
      lambda r: f"What went into the repo this week? Give me the git log since {pick(r, ['Monday', 'last Friday', date_iso(r)])}.",
      lambda r: f"Print the {num(r, 3, 15)} most recent commits that touched {pick(r, DIRS)}/."])

tool(T("git_diff", "Show the diff of the working tree or between refs.",
       {"repo": S("Repository path"), "ref": S("Ref or range, for example main..feature"), "path": S("Limit to this path"),
        "staged": B("Show the staged changes only"), "stat": B("Show only the file stat summary")}, []),
     [lambda r: f"Show me the diff of my uncommitted changes{pick(r, ['', ' in ' + pick(r, FILE_PATHS), ' as a stat summary'])}.",
      lambda r: f"What changed between {pick(r, ['main', 'develop', 'release/2.8'])} and {pick(r, ['feature/retry', 'fix/timeout', 'HEAD', 'hotfix/cert'])}? Show the diff.",
      lambda r: f"Diff the staged changes so I can write the commit message.",
      lambda r: f"Compare {pick(r, ['v2.7.0', 'v2.8.1', 'v3.0.0-rc1'])} to HEAD for the {pick(r, DIRS)} directory."])

tool(T("create_pull_request", "Open a pull request on the code host and return its url.",
       {"repo": S("Repository in owner/name form"), "title": S("Pull request title"), "head": S("Source branch"),
        "base": S("Target branch, default main"), "body": S("Description in markdown"), "draft": B("Open as a draft")}, ["repo", "title", "head"]),
     [lambda r: f"Open a PR in {pick(r, REPOS)} from {pick(r, ['feature/retry', 'fix/timeout', 'chore/deps', 'docs/runbook', 'hotfix/cert'])} into main titled '{pick(r, ['Add retry with jitter to the exporter', 'Fix the upload timeout', 'Bump dependencies', 'Update the runbook', 'Renew the staging cert'])}'.",
      lambda r: f"Create a draft pull request for branch {pick(r, ['wip/cache', 'spike/duckdb', 'exp/new-parser'])} in {pick(r, REPOS)}. Title: '{pick(r, ['WIP: cache layer', 'Spike: duckdb for reports', 'Experiment: streaming parser'])}'.",
      lambda r: f"Raise a pull request from {pick(r, ['fix/null-country', 'fix/double-push', 'fix/csv-trailing-comma'])} against {pick(r, ['main', 'develop', 'release/2.9'])} in {pick(r, REPOS)}; title it after the branch and mention it closes {pick(r, TEAMS).upper()}-{num(r, 100, 4999)}."])

tool(T("list_pull_requests", "List pull requests of a repository.",
       {"repo": S("Repository in owner/name form"), "state": S("Filter by state", enum=["open", "closed", "merged", "all"]),
        "author": S("Filter by author"), "limit": I("Maximum results, default 20", minimum=1, maximum=100)}, ["repo"]),
     [lambda r: f"What pull requests are open in {pick(r, REPOS)}?",
      lambda r: f"List the PRs {pick(r, NAMES).lower()} merged in {pick(r, REPOS)}.",
      lambda r: f"Show me the {num(r, 5, 30)} most recent closed pull requests for {pick(r, REPOS)}.",
      lambda r: f"Are there any open PRs waiting in {pick(r, REPOS)} right now? Give me the list."])

# http ------------------------------------------------------------------------
tool(T("http_get", "Perform an HTTP GET request and return status, headers and body.",
       {"url": S("Request url"), "headers": {"type": "object", "description": "Extra request headers"},
        "timeout_s": I("Timeout in seconds, default 30", minimum=1, maximum=300)}, ["url"]),
     [lambda r: f"Fetch {pick(r, URLS)} and show me the response.",
      lambda r: f"Is {pick(r, URLS)} up? Do a GET and tell me the status code.",
      lambda r: f"Get {pick(r, URLS)} with the header Accept: application/json and show me the body.",
      lambda r: f"Hit the health endpoint {pick(r, ['https://api.northwind.dev/healthz', 'https://harbor-eng.net/api/health', 'http://localhost:8080/health', 'https://status.acme-labs.io/ping'])} and report what it returns.",
      lambda r: f"Download the JSON at {pick(r, URLS)} and tell me how many items it has."])

tool(T("http_post", "Perform an HTTP POST request with a body and return the response.",
       {"url": S("Request url"), "body": S("Request body as text"), "content_type": S("Content-Type header, default application/json"),
        "headers": {"type": "object", "description": "Extra request headers"}}, ["url", "body"]),
     [lambda r: f"POST {{\"event\": \"{pick(r, ['deploy.finished', 'user.signup', 'invoice.paid', 'build.failed'])}\", \"service\": \"{pick(r, SERVICES)}\"}} to {pick(r, ['https://hooks.acme-labs.io/ingest', 'https://api.northwind.dev/v2/events', 'http://localhost:9000/webhook'])}.",
      lambda r: f"Send a test webhook to {pick(r, ['https://webhook-relay.quartz.team/test', 'https://api.example.org/hooks/ping'])} with the JSON body {{\"ping\": true, \"from\": \"{pick(r, NAMES).lower()}\"}}.",
      lambda r: f"Call {pick(r, ['https://api.acme-labs.io/v1/jobs', 'https://api.northwind.dev/v2/exports'])} with a POST and the body {{\"type\": \"{pick(r, ['reindex', 'export', 'rebuild', 'sync'])}\", \"target\": \"{pick(r, SERVICES)}\"}} and show me the reply.",
      lambda r: f"Post the form fields name={pick(r, NAMES).lower()}&plan={pick(r, ['pro', 'team', 'free'])} to {pick(r, ['https://tinfoil.app/api/signup', 'http://localhost:3000/api/signup'])} as application/x-www-form-urlencoded."])

tool(T("dns_lookup", "Resolve a domain name and return its DNS records.",
       {"domain": S("Domain name"), "record_type": S("Record type, default A", enum=["A", "AAAA", "MX", "TXT", "CNAME", "NS"]),
        "resolver": S("DNS server to ask, optional")}, ["domain"]),
     [lambda r: f"What are the MX records for {pick(r, DOMAINS)}?",
      lambda r: f"Resolve {pick(r, ['api', 'www', 'mail', 'cdn', 'status'])}.{pick(r, DOMAINS)} and give me the IP.",
      lambda r: f"Check the TXT records on {pick(r, DOMAINS)} - I want to see if the SPF entry is there.",
      lambda r: f"Which name servers does {pick(r, DOMAINS)} use?",
      lambda r: f"Does {pick(r, ['app', 'docs', 'shop'])}.{pick(r, DOMAINS)} have a CNAME? Look it up with {pick(r, ['1.1.1.1', '8.8.8.8', '9.9.9.9'])}."])

# math / conversion -------------------------------------------------------------
tool(T("calculator", "Evaluate an arithmetic expression and return the result.",
       {"expression": S("Expression using + - * / ^ ( ) and functions like sqrt, log, sin"),
        "precision": I("Decimal places in the result, default 6", minimum=0, maximum=20)}, ["expression"]),
     [lambda r: f"What is {num(r, 1000, 99999)} * {num(r, 11, 999)} / {num(r, 3, 97)}? I need the exact value.",
      lambda r: f"Compute ({num(r, 100, 999)} + {num(r, 100, 999)}) ^ 2 - {num(r, 1000, 9999)} for me.",
      lambda r: f"How much is {num(r, 12, 89)}.{num(r, 1, 9)}% of {num(r, 1000, 250000)}? Give me two decimals.",
      lambda r: f"Work out the square root of {num(r, 1000, 999999)} to {num(r, 3, 8)} decimal places.",
      lambda r: f"If I split {num(r, 1000, 99999)} across {num(r, 3, 47)} people, how much does each get? Calculate it exactly.",
      lambda r: f"Calculate {num(r, 2, 9)}^{num(r, 10, 40)} mod {num(r, 7, 997)}."])

tool(T("convert_units", "Convert a quantity between units of measure.",
       {"value": N("The quantity"), "from_unit": S("Source unit, for example km"), "to_unit": S("Target unit, for example mi")}, ["value", "from_unit", "to_unit"]),
     [lambda r: (lambda u: f"How many {u[1]} is {u[2]} {u[0]}?")(pick(r, UNITS)),
      lambda r: (lambda u: f"Convert {u[2]} {u[0]} to {u[1]} for me.")(pick(r, UNITS)),
      lambda r: (lambda u: f"The recipe says {u[2]} {u[0]}. What is that in {u[1]}?")(pick(r, UNITS)),
      lambda r: (lambda u: f"I have a reading of {u[2]} {u[0]} and the form wants {u[1]}. What do I enter?")(pick(r, UNITS))])

tool(T("convert_currency", "Convert an amount between currencies at the current or a historical rate.",
       {"amount": N("Amount to convert"), "from_currency": S("ISO 4217 code"), "to_currency": S("ISO 4217 code"),
        "date": S("Rate date, ISO 8601, default today")}, ["amount", "from_currency", "to_currency"]),
     [lambda r: f"How much is {num(r, 50, 20000)} {pick(r, CURRENCIES)} in {pick(r, CURRENCIES)} today?",
      lambda r: f"Convert {num(r, 100, 9000)} {pick(r, CURRENCIES)} to {pick(r, CURRENCIES)} at the rate from {date_iso(r)}.",
      lambda r: f"The invoice is {num(r, 500, 50000)} {pick(r, CURRENCIES)}. What is that in {pick(r, CURRENCIES)} right now?",
      lambda r: f"What does {num(r, 10, 500)} {pick(r, CURRENCIES)} come to in {pick(r, CURRENCIES)}? I am budgeting for a trip."])

tool(T("get_timezone_time", "Get the current date and time in a time zone.",
       {"timezone": S("IANA time zone name, for example Europe/Berlin"), "format": S("Output format", enum=["iso", "12h", "24h"])}, ["timezone"]),
     [lambda r: f"What time is it right now in {pick(r, TIMEZONES)}?",
      lambda r: f"Give me the current time in {pick(r, CITIES)} ({pick(r, TIMEZONES)}) in 24 hour format.",
      lambda r: f"Is it still office hours in {pick(r, TIMEZONES)}? Check the local time there.",
      lambda r: f"What is today's date and time in {pick(r, TIMEZONES)}, ISO format?"])

tool(T("get_stock_quote", "Get the latest price of a stock or ETF.",
       {"symbol": S("Ticker symbol"), "exchange": S("Exchange code, optional"), "currency": S("Quote currency, optional")}, ["symbol"]),
     [lambda r: f"What is {pick(r, SYMBOLS)} trading at right now?",
      lambda r: f"Get me the current price of {pick(r, SYMBOLS)} in {pick(r, ['EUR', 'USD', 'GBP'])}.",
      lambda r: f"How did {pick(r, SYMBOLS)} close today? Quote please.",
      lambda r: f"Pull the latest quote for {pick(r, SYMBOLS)} on {pick(r, ['NASDAQ', 'NYSE', 'XETRA', 'LSE'])}."])

# translation / notes / reminders --------------------------------------------------
tool(T("translate_text", "Translate text into a target language.",
       {"text": S("Text to translate"), "target_language": S("Target language code, for example de"),
        "source_language": S("Source language code, optional; detected when missing"), "formal": B("Use the formal register")}, ["text", "target_language"]),
     [lambda r: f"Translate '{pick(r, ['The meeting is moved to Thursday at ten.', 'Please send the signed contract by Friday.', 'Where is the nearest pharmacy?', 'Thank you for your patience, the refund is on its way.', 'The server will be down for maintenance tonight.', 'Could you send me the invoice again?'])}' into {pick(r, LANGS)}.",
      lambda r: f"How do you say '{pick(r, ['see you tomorrow', 'the package has not arrived', 'I would like to cancel my subscription', 'the elevator is out of order', 'we are closed on Sundays'])}' in {pick(r, LANGS)}?",
      lambda r: f"Put this in formal {pick(r, LANGS)}: '{pick(r, ['We regret the delay and will ship your order today.', 'Your account has been verified.', 'Please find the agenda attached.', 'The event starts at nine.'])}'",
      lambda r: f"Translate to {pick(r, LANGS)}: {pick(r, ['Out of office until the 24th. For urgent matters contact the support desk.', 'Password reset requested. If this was not you, ignore this message.', 'Two tickets for the 8 pm show, please.', 'Warning: this action cannot be undone.'])}"])

tool(T("create_note", "Create a note in the notes app.",
       {"title": S("Note title"), "body": S("Note text in markdown"), "tags": A("Tags"), "notebook": S("Notebook name, default Inbox")}, ["title", "body"]),
     [lambda r: f"Make a note titled '{pick(r, ['gift ideas', 'books to read', 'retro takeaways', 'q4 goals', 'hackathon pitch', 'garden plan'])}' with: {pick(r, ['a wool scarf, noise cancelling headphones, a coffee grinder', 'The Left Hand of Darkness, Piranesi, The Dispossessed', 'fewer meetings, clearer owners, faster reviews', 'ship the v3 api, cut infra cost by 15 percent, hire two engineers', 'a cli that turns screenshots into tickets', 'tomatoes in the south bed, herbs by the door'])}.",
      lambda r: f"Save a quick note: {pick(r, ['the wifi password for the office is on the fridge', 'call the plumber about the leak on Tuesday', 'dentist moved to the 3rd at 9', 'car needs new tyres before winter', 'return the library books by the 20th'])}. Tag it {pick(r, ['home', 'todo', 'admin', 'personal'])}.",
      lambda r: f"Jot down in my {pick(r, ['Work', 'Ideas', 'Journal', 'Inbox'])} notebook: '{pick(r, ['prototype the streaming parser next sprint', 'ask legal about the data residency question', 'the demo went well, follow up with the two leads', 'try a four day week experiment in Q1'])}'."])

tool(T("set_reminder", "Create a reminder that fires at a time.",
       {"text": S("What to remind about"), "time": S("When to fire, ISO 8601 or natural text like tomorrow 9am"),
        "repeat": S("Repeat rule", enum=["none", "daily", "weekly", "monthly"])}, ["text", "time"]),
     [lambda r: f"Remind me to {pick(r, ['submit the expense report', 'water the plants', 'renew the domain', 'send the invoice to Flux Ltd', 'check the backup job', 'call the bank', 'take the bins out'])} {day(r)} at {clock(r)}.",
      lambda r: f"Set a reminder for {date_iso(r)} {clock(r)}: {pick(r, ['passport appointment', 'tax filing deadline', 'renew the car insurance', 'team offsite starts', 'certificate expires'])}.",
      lambda r: f"Every {pick(r, ['Monday', 'Friday', 'day', 'month'])} at {clock(r)}, remind me to {pick(r, ['review the on-call log', 'rotate the api key', 'update the status page', 'back up the laptop', 'stretch'])}.",
      lambda r: f"Ping me in {num(r, 2, 6)} hours to {pick(r, ['check the deploy', 'reply to the recruiter', 'restart the export', 'pick up the parcel'])}."])

# packages / k8s / cloud / docker -----------------------------------------------------
tool(T("install_package", "Install a package with a package manager.",
       {"name": S("Package name"), "manager": S("Package manager", enum=["pip", "npm", "cargo", "apt", "brew"]),
        "version": S("Version constraint, optional"), "dev": B("Install as a development dependency")}, ["name", "manager"]),
     [lambda r: (lambda m: f"Install {pick(r, PACKAGES[m])} with {m}.")(pick(r, list(PACKAGES))),
      lambda r: (lambda m: f"Add {pick(r, PACKAGES[m])} {pick(r, ['as a dev dependency', 'version ' + str(num(r, 1, 6)) + '.' + str(num(r, 0, 12)), 'to the project'])} using {m}.")(pick(r, ['pip', 'npm', 'cargo'])),
      lambda r: (lambda m: f"I need {pick(r, PACKAGES[m])} on this box. Install it via {m}.")(pick(r, ['apt', 'brew'])),
      lambda r: (lambda m: f"Can you {m} install {pick(r, PACKAGES[m])} for me? Pin it to {num(r, 1, 9)}.{num(r, 0, 20)}.x.")(pick(r, ['pip', 'npm']))])

tool(T("k8s_get_pods", "List pods in a Kubernetes namespace with their status.",
       {"namespace": S("Namespace, default default"), "selector": S("Label selector, for example app=api"),
        "all_namespaces": B("List pods from every namespace"), "field_selector": S("Field selector, for example status.phase=Failed")}, []),
     [lambda r: f"Which pods are running in the {pick(r, NAMESPACES)} namespace?",
      lambda r: f"Show me the pods for app={pick(r, DEPLOYMENTS)} in {pick(r, NAMESPACES)} and their status.",
      lambda r: f"Are any pods in a crash loop in {pick(r, NAMESPACES)}? List them.",
      lambda r: f"List every failed pod across all namespaces.",
      lambda r: f"How many replicas of {pick(r, DEPLOYMENTS)} are up in {pick(r, NAMESPACES)} right now? Check the pods."])

tool(T("k8s_logs", "Fetch logs from a pod container.",
       {"pod": S("Pod name"), "namespace": S("Namespace"), "container": S("Container name, optional"),
        "tail_lines": I("Number of lines from the end, default 100", minimum=1, maximum=5000), "previous": B("Logs of the previous crashed container")}, ["pod"]),
     [lambda r: f"Get the last {num(r, 50, 500)} log lines from pod {pick(r, DEPLOYMENTS)}-{r.randrange(10**5, 10**6)}-{pick(r, ['x7k2q', 'p9d4z', 'm2n8w', 'q1r5t'])} in {pick(r, NAMESPACES)}.",
      lambda r: f"Show me the logs of the previous crashed container in {pick(r, DEPLOYMENTS)}-{num(r, 0, 4)} in namespace {pick(r, NAMESPACES)}.",
      lambda r: f"Tail {num(r, 100, 1000)} lines from the {pick(r, ['sidecar', 'app', 'proxy', 'init'])} container of {pick(r, DEPLOYMENTS)}-{r.randrange(10**5, 10**6)}-{pick(r, ['a8b3c', 'k4l9m', 'z2y7x'])} in {pick(r, NAMESPACES)}.",
      lambda r: f"What is {pick(r, DEPLOYMENTS)}-{r.randrange(10**5, 10**6)}-{pick(r, ['h5j1k', 'v6w2u'])} logging in {pick(r, NAMESPACES)}? Grab the last {num(r, 30, 200)} lines."])

tool(T("k8s_scale", "Scale a Kubernetes deployment to a number of replicas.",
       {"deployment": S("Deployment name"), "namespace": S("Namespace"), "replicas": I("Target replica count", minimum=0, maximum=200)}, ["deployment", "replicas"]),
     [lambda r: f"Scale {pick(r, DEPLOYMENTS)} in {pick(r, NAMESPACES)} to {num(r, 2, 12)} replicas.",
      lambda r: f"Bring {pick(r, DEPLOYMENTS)} down to {num(r, 0, 1)} replica in {pick(r, NAMESPACES)} for the maintenance window.",
      lambda r: f"We expect a traffic spike. Set {pick(r, DEPLOYMENTS)} ({pick(r, NAMESPACES)}) to {num(r, 8, 30)} replicas.",
      lambda r: f"Double {pick(r, DEPLOYMENTS)} from 3 to 6 replicas in the {pick(r, NAMESPACES)} namespace."])

tool(T("s3_list_objects", "List objects in a cloud storage bucket.",
       {"bucket": S("Bucket name"), "prefix": S("Key prefix filter"), "max_keys": I("Maximum keys, default 100", minimum=1, maximum=1000)}, ["bucket"]),
     [lambda r: f"What is in the {pick(r, BUCKETS)} bucket under {pick(r, ['2026/09/', 'daily/', 'exports/', 'tenant-42/', 'raw/'])}?",
      lambda r: f"List the objects in {pick(r, BUCKETS)}, at most {num(r, 20, 300)}.",
      lambda r: f"Do we have any files with the prefix {pick(r, ['backup-2026-09', 'invoice-', 'snapshot/', 'model-v3/'])} in {pick(r, BUCKETS)}?",
      lambda r: f"Show me the latest keys in {pick(r, BUCKETS)}/{pick(r, ['logs/', 'archive/', 'incoming/'])}."])

tool(T("s3_copy", "Copy an object or a prefix between storage locations.",
       {"source": S("Source uri, for example s3://bucket/key"), "destination": S("Destination uri"), "recursive": B("Copy every key under the prefix")}, ["source", "destination"]),
     [lambda r: f"Copy s3://{pick(r, BUCKETS)}/{pick(r, ['exports/2026-09-17.csv', 'snapshots/db-2026-09-15.dump', 'reports/q3.pdf'])} to s3://{pick(r, BUCKETS)}/{pick(r, ['archive/', 'incoming/', 'shared/'])}.",
      lambda r: f"Mirror the whole {pick(r, ['2026/09/', 'tenant-7/', 'models/v3/'])} prefix from {pick(r, BUCKETS)} into {pick(r, BUCKETS)}.",
      lambda r: f"Move a copy of s3://{pick(r, BUCKETS)}/{pick(r, ['logs/app-2026-09-18.gz', 'media/hero.png', 'exports/users.parquet'])} to my local ./downloads/ folder.",
      lambda r: f"Push the local file {pick(r, FILE_PATHS)} up to s3://{pick(r, BUCKETS)}/{pick(r, ['uploads/', 'manual/', 'adhoc/'])}."])

tool(T("docker_ps", "List containers on the host.",
       {"all": B("Include stopped containers"), "filter": S("Filter, for example name=api or status=exited")}, []),
     [lambda r: f"Which containers are running on this host?",
      lambda r: f"List all containers, including the stopped ones.",
      lambda r: f"Is the {pick(r, SERVICES)} container up? Check docker.",
      lambda r: f"Show me the containers that exited."])

tool(T("docker_logs", "Fetch logs from a container.",
       {"container": S("Container name or id"), "tail": I("Lines from the end, default 100", minimum=1, maximum=10000),
        "since": S("Only logs after this time, for example 10m or an ISO timestamp"), "timestamps": B("Prefix lines with timestamps")}, ["container"]),
     [lambda r: f"Show the last {num(r, 50, 500)} lines of logs from the {pick(r, SERVICES)} container.",
      lambda r: f"What has {pick(r, SERVICES)} logged in the last {num(r, 5, 60)} minutes? Include timestamps.",
      lambda r: f"Tail the {pick(r, SERVICES)} container logs, {num(r, 100, 1000)} lines, since {clock(r)} today.",
      lambda r: f"Grab {num(r, 20, 200)} lines from container {pick(r, ['api_1', 'db_1', 'worker_1', 'proxy_1', 'redis_1'])} with timestamps."])

tool(T("schedule_job", "Schedule a recurring job with a cron expression.",
       {"name": S("Job name"), "cron": S("Cron expression, five fields"), "command": S("Command to run"),
        "timezone": S("Time zone for the schedule, default UTC"), "enabled": B("Start enabled, default true")}, ["name", "cron", "command"]),
     [lambda r: f"Schedule a job called {pick(r, ['nightly-backup', 'cache-warm', 'report-mailer', 'log-rotate', 'index-rebuild'])} that runs `{pick(r, ['scripts/backup.sh', 'python3 jobs/warm_cache.py', 'make report', 'logrotate /etc/logrotate.conf', 'scripts/reindex.sh'])}` at {pick(r, ['3 am every day', '6:30 every Monday', 'every 15 minutes', 'midnight on the 1st', 'every 4 hours'])}.",
      lambda r: f"Add a cron job named {pick(r, ['cleanup-tmp', 'sync-s3', 'ping-status', 'renew-cert'])} with the schedule {pick(r, CRONS)} running {pick(r, ['find /tmp -mtime +7 -delete', 'aws s3 sync ./exports s3://acme-backups/exports', 'curl -fsS https://status.acme-labs.io/ping', 'certbot renew -q'])}.",
      lambda r: f"Create a scheduled task '{pick(r, ['weekly-digest', 'db-vacuum', 'metrics-rollup'])}' in the {pick(r, TIMEZONES)} time zone: {pick(r, CRONS)}, command `{pick(r, ['python3 jobs/digest.py', 'psql -c vacuum', 'scripts/rollup.sh'])}`."])

tool(T("query_metrics", "Query a time series metric for a service and return aggregated values.",
       {"metric": S("Metric name, for example http_request_duration_seconds"), "service": S("Service name"),
        "range": S("Time range, for example 1h, 24h, 7d"), "aggregation": S("Aggregation", enum=["avg", "max", "min", "p50", "p95", "p99", "sum", "count"]),
        "step": S("Resolution, for example 1m")}, ["metric", "range"]),
     [lambda r: f"What was the {pick(r, METRICS)} of {pick(r, SERVICES)} over the last {pick(r, ['hour', '6 hours', '24 hours', 'week'])}?",
      lambda r: f"Show me the p99 latency for {pick(r, SERVICES)} for the past {num(r, 2, 48)} hours at one minute resolution.",
      lambda r: f"Did the error rate of {pick(r, SERVICES)} go up in the last {num(r, 15, 120)} minutes? Pull the metric.",
      lambda r: f"Give me the max {pick(r, ['memory usage', 'cpu usage', 'queue depth', 'open connections'])} of {pick(r, SERVICES)} over {pick(r, ['24h', '7d', '3d'])}.",
      lambda r: f"How many requests per second did {pick(r, SERVICES)} serve on average over the last {pick(r, ['hour', 'day'])}?"])

tool(T("generate_password", "Generate a random password.",
       {"length": I("Password length, default 20", minimum=8, maximum=128), "symbols": B("Include punctuation"),
        "count": I("How many passwords, default 1", minimum=1, maximum=20)}, []),
     [lambda r: f"Generate a {num(r, 16, 48)} character password with symbols.",
      lambda r: f"I need {num(r, 2, 6)} random passwords, {num(r, 12, 32)} characters, letters and digits only.",
      lambda r: f"Make me a strong password for the {pick(r, ['staging db', 'wifi', 'vpn', 'admin panel', 'backup archive'])}, {num(r, 20, 40)} characters.",
      lambda r: f"Give me a new random passphrase-style password, at least {num(r, 24, 64)} characters."])

tool(T("run_tests", "Run the project's test suite and return the summary.",
       {"path": S("Directory or file to test, default the whole project"), "pattern": S("Test name filter, for example test_login"),
        "verbose": B("Show every test name"), "fail_fast": B("Stop at the first failure")}, []),
     [lambda r: f"Run the tests in {pick(r, ['tests/test_billing.py', 'tests/api', 'tests/unit', 'src/lib', 'tests/test_retry.py'])}.",
      lambda r: f"Execute only the tests that match {pick(r, ['test_login', 'test_export', 'retry', 'timeout', 'test_parse'])} and stop at the first failure.",
      lambda r: f"Run the whole test suite verbosely and tell me how many pass.",
      lambda r: f"Kick off the {pick(r, ['integration', 'unit', 'smoke'])} tests under {pick(r, ['tests/', 'spec/', 'test/'])} and report the result."])


ALL_TOOL_NAMES = list(TOOLS)


# --------------------------------------------------------------------------- mixed: normal reply, no call
SYSTEM_SHORT = [
    "You are a helpful assistant. Use a tool when it helps; otherwise answer directly.",
    "You are an assistant for a small software team. Tools are available for actions; plain questions get plain answers.",
    "You are a concise assistant. Call a function only when the request needs live data or an action.",
    "You help an operations engineer. Prefer a direct answer when no tool is needed.",
    "You are a friendly assistant. Ask a short question when a request is missing information you need.",
    "Assistant for a data team. Answer questions from your own knowledge unless a tool is clearly required.",
]

MIXED_NOCALL = [
    lambda r: f"Explain in a few sentences what a {pick(r, ['cold front', 'dew point', 'heat index', 'wind chill', 'barometric pressure drop'])} is and why it matters for {pick(r, ['hikers', 'cyclists', 'gardeners', 'pilots'])}. No need to look anything up.",
    lambda r: f"What is the difference between {pick(r, ['a LEFT JOIN and an INNER JOIN', 'WHERE and HAVING', 'a primary key and a unique index', 'DELETE and TRUNCATE', 'a view and a materialized view', 'UNION and UNION ALL'])} in SQL? Just explain, do not run anything.",
    lambda r: f"Write a {pick(r, ['python', 'javascript', 'go', 'bash'])} function that {pick(r, ['reverses the words in a sentence', 'checks whether a string is a palindrome', 'returns the n-th fibonacci number iteratively', 'counts vowels in a string', 'merges two sorted arrays', 'validates an ipv4 address', 'flattens a nested list'])}. Include a short example.",
    lambda r: f"Which is the better default for a new {pick(r, ['internal tool', 'rest api', 'cli', 'data pipeline'])}: {pick(r, ['postgres or sqlite', 'rest or grpc', 'yaml or toml config', 'pytest or unittest', 'cron or a queue worker', 'docker compose or plain systemd'])}? Give me your opinion and the reasoning.",
    lambda r: f"I got the error '{pick(r, ['ECONNREFUSED 127.0.0.1:5432', 'ModuleNotFoundError: No module named yaml', 'fatal: refusing to merge unrelated histories', 'CrashLoopBackOff', 'ERR_TOO_MANY_REDIRECTS', 'permission denied (publickey)', 'SSL: CERTIFICATE_VERIFY_FAILED', 'exit code 137'])}'. What does it usually mean and what are the common causes?",
    lambda r: f"Give me {num(r, 3, 6)} name ideas for a {pick(r, ['cli that syncs dotfiles', 'team retro app', 'log viewer', 'internal feature flag service', 'weather widget', 'note taking app for meetings'])}. Short names, no explanations.",
    lambda r: f"Rewrite this so it sounds friendlier: '{pick(r, ['Your request is denied. Provide the missing form.', 'The server is down again because someone deployed on Friday.', 'Stop emailing me about the invoice, it is paid.', 'This ticket is closed, the behavior is intended.'])}'",
    lambda r: f"In plain words, what does {pick(r, ['a cron expression like 0 3 * * *', 'the HTTP status 429', 'a kubernetes readiness probe', 'an s3 lifecycle rule', 'a git rebase', 'a database index', 'a dns ttl', 'a certificate chain'])} do?",
    lambda r: f"Which tools do you have available right now? List their names in one line, nothing else.",
    lambda r: f"Thanks, that fixed it. {pick(r, ['Have a good one.', 'Appreciate the quick turnaround.', 'I will close the ticket myself.', 'No further action needed.'])}",
    lambda r: f"Draft a two sentence status update for the {pick(r, ['launch', 'migration', 'audit', 'beta'])}: {pick(r, ['on track, one risk around the vendor api', 'delayed by a week, root cause known', 'done, monitoring for regressions', 'blocked on legal review'])}.",
    lambda r: f"Roughly how {pick(r, ['far is it from ' + pick(r, CITIES) + ' to ' + pick(r, CITIES), 'long does a flight from ' + pick(r, CITIES) + ' to ' + pick(r, CITIES) + ' take', 'many time zones are between ' + pick(r, CITIES) + ' and ' + pick(r, CITIES)])}? A ballpark from memory is fine, do not use any tool.",
    lambda r: f"Explain like I am new to ops: why do we {pick(r, ['rotate api keys', 'keep staging separate from prod', 'pin dependency versions', 'run canary deploys', 'set resource limits on pods', 'keep backups in another region'])}?",
    lambda r: f"Is it safe to {pick(r, ['delete the node_modules folder', 'run git gc on a shared repo', 'restart postgres during the day', 'scale a stateful set to zero', 'change the timezone of a server'])}? What should I think about before I do it? Do not run anything yet.",
    lambda r: f"What would you name the columns of a table that stores {pick(r, ['api keys', 'feature flags', 'audit events', 'file uploads', 'email bounces'])}? Just propose the schema in text.",
    lambda r: f"Summarize the trade-offs of {pick(r, ['speculative decoding', 'quantizing a model to 2 bits', 'a monorepo', 'server side rendering', 'event sourcing', 'feature flags'])} in five bullets.",
    lambda r: f"Here is a commit message: '{pick(r, ['fixed stuff', 'wip', 'Update file', 'changes from review', 'final final version'])}'. Write a better one for a change that {pick(r, ['adds retries to the exporter', 'fixes a null country in signup', 'bumps the base image', 'documents the release flow'])}.",
    lambda r: f"How do I convert {pick(r, ['celsius to fahrenheit', 'miles to kilometers', 'pounds to kilograms', 'knots to km/h'])} by hand? I want the formula, not a converted number.",
    lambda r: f"Tell me a {pick(r, ['limerick', 'haiku', 'two line poem'])} about {pick(r, ['a flaky test', 'a pager going off at 3 am', 'a merge conflict', 'a slow ci pipeline', 'a forgotten todo comment'])}.",
    lambda r: f"What questions should I ask a vendor before we buy {pick(r, ['a log management product', 'a feature flag service', 'an error tracker', 'a secrets manager', 'a status page'])}? Five questions, brief.",
    lambda r: f"Explain the difference between {pick(r, ['a container and a virtual machine', 'a process and a thread', 'tcp and udp', 'symmetric and asymmetric encryption', 'a hash and an encryption', 'latency and throughput'])} for a junior engineer.",
    lambda r: f"I am writing a runbook. What sections should it have for {pick(r, ['a database failover', 'a certificate renewal', 'a queue backlog', 'a disk full alert'])}? Outline only.",
    lambda r: f"What is {num(r, 12, 99)} times {num(r, 2, 12)}? Just answer, no need for a calculator.",
    lambda r: f"Why does {pick(r, ['git say detached HEAD', 'docker say no space left on device when df shows free space', 'python print a SyntaxWarning about invalid escape sequence', 'kubectl show a pod as Running but not Ready', 'nginx return 502 while the app logs look fine'])}? Explain the likely reason.",
    lambda r: f"Proofread this sentence and fix only the grammar: '{pick(r, ['The reports was sent to the customer yesterday and they has not replied.', 'Each of the servers have their own certificate.', 'Me and Tara deployed the fix on friday.', 'There is less errors since the patch.'])}'",
]

# --------------------------------------------------------------------------- mixed: clarifying question
MIXED_CLARIFY = [
    lambda r: f"Email the {pick(r, ['quarterly summary', 'updated contract', 'incident report', 'slide deck'])} to the team.",
    lambda r: f"Book a {pick(r, ['room', 'meeting', 'slot'])} for the {pick(r, ['sync', 'review', 'handover', 'kickoff'])} sometime next week.",
    lambda r: f"Delete the old {pick(r, ['logs', 'backups', 'exports', 'snapshots'])}.",
    lambda r: f"Scale the deployment {pick(r, ['up', 'down', 'a bit'])}.",
    lambda r: f"Convert {num(r, 10, 900)} for me.",
    lambda r: f"Run the usual query{pick(r, ['', ' again', ' for last week'])}.",
    lambda r: f"Translate this{pick(r, ['', ' for the customer', ' for the onboarding email'])}.",
    lambda r: f"Set a reminder for the {pick(r, ['dentist', 'call with legal', 'deploy', 'renewal'])}.",
    lambda r: f"Move the {pick(r, ['report', 'export', 'draft', 'archive'])} to the other folder.",
    lambda r: f"Commit my changes.",
    lambda r: f"What's the weather like {pick(r, ['there', 'at the venue', 'where the offsite is', 'at their office'])}?",
    lambda r: f"Install the {pick(r, ['library we discussed', 'package from the ticket', 'thing that fixes the ssl error'])}.",
    lambda r: f"Fetch that url again{pick(r, ['', ' and show me the body', ' with the auth header'])}.",
    lambda r: f"Post the payload to the webhook.",
    lambda r: f"Look up {pick(r, ['her', 'his', 'their'])} number.",
    lambda r: f"Copy the backup to the other bucket.",
    lambda r: f"Check the logs for the crashing pod.",
    lambda r: f"Open a ticket for the bug from this morning.",
    lambda r: f"Update the ticket and say it is fixed.",
    lambda r: f"Can you get me a quote for {pick(r, ['the stock we talked about', 'that ticker', 'the one on my watchlist'])}?",
    lambda r: f"How much is {num(r, 50, 5000)} in {pick(r, CURRENCIES)}?",
    lambda r: f"Read the config file and tell me the port.",
    lambda r: f"Schedule the cleanup job at the usual time.",
    lambda r: f"Send the invoice.",
    lambda r: f"List the bucket.",
    lambda r: f"Save this as a note.",
    lambda r: f"What time is it {pick(r, ['over there', 'for the customer', 'in their office', 'where Farid is'])}?",
    lambda r: f"Search the docs for the thing about {pick(r, ['retries', 'limits', 'the policy', 'that setting'])} - you know the one.",
    lambda r: f"Get the pods.",
    lambda r: f"Run it again but with the {pick(r, ['other flag', 'fix applied', 'new input'])}.",
    lambda r: f"Rename the file to something better.",
    lambda r: f"Show me the diff{pick(r, [' for the branch', ' from yesterday', ' of that change'])}.",
    lambda r: f"Calculate the total for the order{pick(r, ['', ' with the discount', ' including tax'])}.",
    lambda r: f"Make the PR{pick(r, ['', ' for my branch', ' like last time'])}.",
    lambda r: f"Resolve the domain.",
    lambda r: f"Tail the container logs.",
    lambda r: f"Pull the metric for the service that was slow.",
    lambda r: f"Add {pick(r, NAMES)} to the meeting.",
    lambda r: f"Generate a password for {pick(r, ['it', 'the new account', 'that box'])} - the same rules as before.",
    lambda r: f"Summarize the page I sent you.",
]

# tools that fit each clarify request (index-aligned with MIXED_CLARIFY); the record always includes one of these
CLARIFY_TOOLS = [
    ["send_email"], ["create_event", "list_events"], ["delete_file", "list_directory"], ["k8s_scale", "k8s_get_pods"],
    ["convert_units", "convert_currency"], ["run_sql"], ["translate_text"], ["set_reminder"], ["move_file"],
    ["git_commit", "git_status"], ["get_weather"], ["install_package"], ["http_get"], ["http_post"], ["lookup_contact"],
    ["s3_copy", "s3_list_objects"], ["k8s_logs", "k8s_get_pods"], ["create_ticket"], ["update_ticket"], ["get_stock_quote"],
    ["convert_currency"], ["cat_file"], ["schedule_job"], ["send_email"], ["s3_list_objects"], ["create_note"],
    ["get_timezone_time"], ["search_docs"], ["k8s_get_pods"], ["run_command"], ["move_file"], ["git_diff"],
    ["calculator"], ["create_pull_request"], ["dns_lookup"], ["docker_logs"], ["query_metrics"], ["create_event"],
    ["generate_password"], ["summarize_url"],
]
assert len(CLARIFY_TOOLS) == len(MIXED_CLARIFY)


def sample_tools(rng, must_include, k):
    """A tools list of k definitions that holds every name in must_include, in random order."""
    names = list(must_include)
    pool = [n for n in ALL_TOOL_NAMES if n not in names]
    rng.shuffle(pool)
    names += pool[: max(0, k - len(names))]
    rng.shuffle(names)
    return [TOOLS[n] for n in names]
