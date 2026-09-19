#!/usr/bin/env python3
"""agent_scenarios_ops.py - fake tool results and tasks for the ops, docs and data agents (ridge, quill, grain).

Same contract as agent_scenarios_code.py: gen(rng, n) -> {task, call, result, kind, tag}.
"""
import random

from agent_scenarios_code import pick, num, sample, sha, NAMES

# --------------------------------------------------------------------------- RIDGE (kubernetes)
K_SVCS = ["api", "gateway", "worker", "auth", "search", "billing", "notifier", "ingest", "report-builder", "image-resizer", "scheduler", "ledger"]
K_NS = ["prod", "staging", "batch", "monitoring"]
K_STATES = ["Running", "Running", "Running", "CrashLoopBackOff", "Pending", "OOMKilled", "ImagePullBackOff", "Error", "Terminating", "ContainerCreating"]
K_NODES = [f"ip-10-0-{a}-{b}.eu-west-1.compute.internal" for a, b in ((12, 41), (12, 77), (13, 15), (13, 98), (14, 22), (14, 63), (15, 9), (15, 130))]
K_REASONS = ["OOMKilled", "Error", "Completed", "ContainerCannotRun", "DeadlineExceeded"]
K_EVENTS = ["Back-off restarting failed container {c} in pod {p}", "Readiness probe failed: HTTP probe failed with statuscode: 503",
            "Liveness probe failed: Get \"http://10.0.{a}.{b}:8080/healthz\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)",
            "0/12 nodes are available: 2 Insufficient memory, 10 node(s) didn't match Pod's node affinity/selector.",
            "Failed to pull image \"registry.acme.internal/{c}:{v}\": rpc error: code = NotFound desc = manifest unknown",
            "Successfully assigned {ns}/{p} to {node}", "Created container {c}", "Started container {c}", "Pulled image \"registry.acme.internal/{c}:{v}\" in {ms}ms",
            "Memory cgroup out of memory: Killed process {pid} ({c}) total-vm:{vm}kB, anon-rss:{rss}kB", "Scaled up replica set {c}-{h} to {r}",
            "Warning  FailedMount  MountVolume.SetUp failed for volume \"{c}-config\" : configmap \"{c}-config\" not found"]


def podname(rng, svc):
    return f"{svc}-{sha(rng)}{sha(rng)[:2]}-{''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(5))}"


def ridge_get_pods(rng, n):
    ns = pick(rng, K_NS)
    svc_focus = pick(rng, K_SVCS)
    lines = [f"{'NAME':<44}{'READY':<8}{'STATUS':<20}{'RESTARTS':<16}{'AGE':<8}{'IP':<14}{'NODE'}"]
    bad_pods = []
    for i in range(max(6, n)):
        svc = svc_focus if i < 3 else pick(rng, K_SVCS)
        st = pick(rng, K_STATES) if (svc == svc_focus and i < 2) else pick(rng, ["Running"] * 6 + K_STATES)
        p = podname(rng, svc)
        ready = "1/1" if st == "Running" else pick(rng, ["0/1", "1/2", "0/2"])
        restarts = f"{num(rng, 3, 40)} ({num(rng, 1, 59)}m ago)" if st in ("CrashLoopBackOff", "Error", "OOMKilled") else str(num(rng, 0, 2))
        age = f"{num(rng, 1, 59)}{pick(rng, ['m', 'h', 'd'])}"
        ip = f"10.0.{num(rng, 12, 15)}.{num(rng, 2, 250)}" if st not in ("Pending",) else "<none>"
        node = pick(rng, K_NODES) if st != "Pending" else "<none>"
        if st != "Running":
            bad_pods.append((p, st))
        lines.append(f"{p:<44}{ready:<8}{st:<20}{restarts:<16}{age:<8}{ip:<14}{node}")
    v = rng.randrange(3)
    if v == 0:
        task = (f"Pods of {svc_focus} in {ns} restart. Find out why. Start with the pod list, then fetch the logs of the previous container of the "
                f"worst pod and continue from there. Report the cause and propose a change only if the evidence is clear.")
        kind = "call"
    elif v == 1:
        task = (f"Give the coordinator a health summary of the {ns} namespace: a table of service, pods ready, pods not ready and restarts in the last "
                f"hour, then the three pods to look at first. Use only the pod list.")
        kind = "report"
    else:
        task = (f"Alert {svc_focus.capitalize().replace('-', '')}PodsNotReady is firing for {ns}. Confirm it from the pod list, then describe the pod with "
                f"the most restarts to get the last state and the events.")
        kind = "call"
    call = {"name": "kubectl", "arguments": {"args": f"get pods -n {ns} -o wide" if rng.random() < 0.6 else f"get pods -n {ns} -l app={svc_focus} -o wide"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_get_pods"}


APP_LOG = {"INFO": ["http request method=GET path=/v1/{ep} status=200 duration_ms={ms} request_id={rid}", "worker {w} processed batch of {rows} in {ms} ms",
                    "health check ok db={ms}ms cache=ok", "config loaded env={ns} replicas={r}", "listening on :8080"],
           "WARN": ["upstream {svc} slow: {ms} ms (threshold 800)", "connection pool exhausted, waiting ({q} queued)", "retry {k}/3 for {svc} after 502",
                    "gc pause {ms} ms", "queue depth {q} above limit"],
           "ERROR": ["panic: runtime error: {panic}", "upstream {svc} returned 503 after 3 retries request_id={rid}", "database: connection refused (10.0.{a}.{b}:5432)",
                     "fatal: out of memory: cannot allocate {mb} MB for batch of {rows} rows", "tls: certificate for {svc}.prod.svc expired on 2026-09-{d}", "context deadline exceeded while waiting for {svc}"]}
PANICS = ["index out of range [3] with length 3", "invalid memory address or nil pointer dereference", "send on closed channel", "concurrent map writes"]


def _applog(rng, svc, ns, t, lvl=None):
    lvl = lvl or pick(rng, ["INFO"] * 5 + ["WARN"] * 2 + ["ERROR"])
    msg = pick(rng, APP_LOG[lvl]).format(ep=pick(rng, ["items", "search", "export", "me", "usage"]), ms=num(rng, 2, 4000), rid=sha(rng), w=num(rng, 1, 8), rows=num(rng, 50, 20000),
                                        ns=ns, r=num(rng, 2, 12), svc=pick(rng, K_SVCS), q=num(rng, 10, 900), k=num(rng, 1, 3), panic=pick(rng, PANICS), a=num(rng, 12, 15), b=num(rng, 2, 250),
                                        mb=num(rng, 200, 4000), d=num(rng, 1, 17))
    h, m, s = 8 + t // 3600, (t // 60) % 60, t % 60
    return f"2026-09-18T{h:02d}:{m:02d}:{s:02d}Z {lvl:<5} {svc} {msg}"


def ridge_logs(rng, n):
    ns = pick(rng, K_NS)
    svc = pick(rng, K_SVCS)
    p = podname(rng, svc)
    t = num(rng, 0, 3000)
    lines = []
    for i in range(max(8, n)):
        t += num(rng, 1, 30)
        lines.append(_applog(rng, svc, ns, t))
    # end with a crash sequence
    t += 3
    lines.append(_applog(rng, svc, ns, t, "ERROR"))
    if rng.random() < 0.6:
        lines += [f"goroutine 1 [running]:", f"main.({pick(rng, ['*Server', '*Worker', '*Batcher'])}).{pick(rng, ['handle', 'run', 'flush'])}(0xc000{sha(rng)[:6]})",
                  f"\t/src/{pick(rng, ['server', 'worker', 'batch'])}.go:{num(rng, 40, 400)} +0x{sha(rng)[:3]}", "exit status 2"]
    else:
        lines.append(f"2026-09-18T{8 + t // 3600:02d}:{(t // 60) % 60:02d}:{t % 60:02d}Z INFO  {svc} shutdown: signal terminated")
    v = rng.randrange(3)
    if v == 0:
        task = (f"The previous container of {p} in {ns} crashed. Read its log tail, name the failure, and say whether it is a code bug, a config "
                f"problem or a resource limit. Then describe the pod to confirm the last state and the limits.")
        kind = "call"
    elif v == 1:
        task = (f"Write the finding for the {svc} crash in {ns} from the container log: the last error, the time, what happened in the minute before, "
                f"and the user impact you can infer. Mark inferences as not verified. Under 250 words.")
        kind = "report"
    else:
        task = (f"From the {svc} log in {ns}, build a timeline table of the WARN and ERROR lines with UTC time and message, then say which upstream "
                f"or resource is the likely cause. No change proposal yet.")
        kind = "report"
    call = {"name": "container_logs", "arguments": {"pod": p, "namespace": ns, "tail": num(rng, 60, 200), "previous": True}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_logs"}


def ridge_describe(rng, n):
    ns = pick(rng, K_NS)
    svc = pick(rng, K_SVCS)
    p = podname(rng, svc)
    node = pick(rng, K_NODES)
    reason = pick(rng, ["OOMKilled", "Error", "CrashLoopBackOff", "Pending"])
    mem = pick(rng, ["256Mi", "512Mi", "1Gi", "2Gi"])
    lines = [f"Name:             {p}", f"Namespace:        {ns}", f"Priority:         0", f"Service Account:  {svc}", f"Node:             {node if reason != 'Pending' else '<none>'}",
             f"Start Time:       Thu, 18 Sep 2026 {num(rng, 6, 11):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d} +0000", f"Labels:           app={svc}", f"                  pod-template-hash={sha(rng)}{sha(rng)[:2]}",
             f"                  version=v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}", f"Status:           {'Running' if reason != 'Pending' else 'Pending'}", f"IP:               {'10.0.' + str(num(rng, 12, 15)) + '.' + str(num(rng, 2, 250)) if reason != 'Pending' else ''}",
             f"Controlled By:    ReplicaSet/{svc}-{sha(rng)}{sha(rng)[:2]}", "Containers:", f"  {svc}:", f"    Image:          registry.acme.internal/{svc}:v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}",
             f"    Port:           8080/TCP", f"    State:          {'Waiting' if reason in ('CrashLoopBackOff', 'Pending') else 'Running'}", f"      Reason:       {reason if reason != 'Pending' else 'ContainerCreating'}",
             f"    Last State:     Terminated", f"      Reason:       {reason if reason in ('OOMKilled', 'Error') else 'Error'}", f"      Exit Code:    {137 if reason == 'OOMKilled' else num(rng, 1, 2)}",
             f"      Started:      Thu, 18 Sep 2026 {num(rng, 6, 11):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d} +0000", f"      Finished:     Thu, 18 Sep 2026 {num(rng, 6, 11):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d} +0000",
             f"    Ready:          False", f"    Restart Count:  {num(rng, 3, 40)}", "    Limits:", f"      cpu:     {pick(rng, ['500m', '1', '2'])}", f"      memory:  {mem}", "    Requests:",
             f"      cpu:     {pick(rng, ['100m', '250m', '500m'])}", f"      memory:  {pick(rng, ['128Mi', '256Mi', '512Mi'])}", f"    Liveness:   http-get http://:8080/healthz delay=10s timeout=1s period=10s #success=1 #failure=3",
             f"    Readiness:  http-get http://:8080/ready delay=5s timeout=1s period=5s #success=1 #failure=3", "    Environment:", f"      LOG_LEVEL:     info", f"      DB_URL:        <set to the key 'db_url' in secret '{svc}-secrets'>  Optional: false",
             f"      BATCH_SIZE:    {num(rng, 100, 5000)}", "Conditions:", "  Type              Status", "  Initialized       True", f"  Ready             False", f"  ContainersReady   False", "  PodScheduled      True", "Events:",
             "  Type     Reason     Age                    From               Message", "  ----     ------     ----                   ----               -------"]
    for _ in range(max(3, n // 4)):
        ev = pick(rng, K_EVENTS).format(c=svc, p=p, a=num(rng, 12, 15), b=num(rng, 2, 250), v=f"v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}", ns=ns, node=node, ms=num(rng, 200, 9000),
                                        pid=num(rng, 100, 30000), vm=num(rng, 500000, 3000000), rss=num(rng, 200000, 2000000), h=sha(rng), r=num(rng, 1, 8))
        typ = "Warning" if any(k in ev for k in ("failed", "Failed", "Back-off", "out of memory", "Insufficient")) else "Normal"
        rsn = pick(rng, ["BackOff", "Unhealthy", "FailedScheduling", "Pulled", "Created", "Started", "Killing", "OOMKilling", "Scheduled"])
        lines.append(f"  {typ:<8} {rsn:<10} {num(rng, 1, 59)}m (x{num(rng, 1, 40)} over {num(rng, 1, 5)}h)  {pick(rng, ['kubelet', 'default-scheduler', 'kubelet'])}{'':<10} {ev}")
    v = rng.randrange(3)
    if v == 0:
        task = (f"{p} in {ns} keeps restarting. Use the describe output to name the cause, then propose a change with propose_change that fixes it "
                f"with the smallest risk. Include the exact kubectl or Helm commands.")
        kind = "call"
    elif v == 1:
        task = (f"Read the describe output for {p} and write the finding: cause, evidence lines, and whether a resource limit, an image or a probe is "
                f"at fault. Quote the exit code and the events. No change proposal.")
        kind = "report"
    else:
        task = (f"Check whether the memory limit of {svc} in {ns} explains the restarts. Use the describe output, then read the manifest under "
                f"deploy/{ns}/{svc}.yaml to compare the limits with what runs.")
        kind = "call"
    call = {"name": "kubectl", "arguments": {"args": f"describe pod {p} -n {ns}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_describe"}


def ridge_top(rng, n):
    ns = pick(rng, K_NS)
    lines = [f"{'POD':<44}{'CPU(cores)':<14}{'MEMORY(bytes)'}"]
    for _ in range(max(6, n)):
        svc = pick(rng, K_SVCS)
        lines.append(f"{podname(rng, svc):<44}{num(rng, 1, 1900)}m{'':<10}{num(rng, 40, 3900)}Mi")
    if n > 12:
        lines += ["", f"$ kubectl top nodes", f"{'NAME':<48}{'CPU(cores)':<12}{'CPU%':<8}{'MEMORY(bytes)':<16}{'MEMORY%'}"]
        for nd in sample(rng, K_NODES, num(rng, 3, 6)):
            lines.append(f"{nd:<48}{num(rng, 200, 3900)}m{'':<7}{num(rng, 5, 99)}%{'':<5}{num(rng, 2000, 15000)}Mi{'':<9}{num(rng, 10, 98)}%")
    v = rng.randrange(2)
    if v == 0:
        task = (f"Rank the {len(lines) - 1 if n <= 12 else sum(1 for l in lines if l.endswith('Mi') and not l.startswith('ip-'))} pods in {ns} by memory and name "
                f"the ones within {num(rng, 10, 25)} percent of their limit. For the top one, describe the pod to get the limit and the restart count.")
        kind = "call"
    else:
        task = (f"Write a capacity note for {ns} from the top output: the {num(rng, 3, 8)} pods with the highest memory, the total per service, and "
                f"whether any node is above {num(rng, 75, 90)} percent. One table and three sentences.")
        kind = "report"
    call = {"name": "kubectl", "arguments": {"args": f"top pods -n {ns} --sort-by=memory"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_top"}


def ridge_rollout(rng, n):
    ns = pick(rng, K_NS)
    svc = pick(rng, K_SVCS)
    lines = [f"deployment.apps/{svc}", "REVISION  CHANGE-CAUSE"]
    rev = num(rng, 20, 90)
    for i in range(max(4, n // 3)):
        lines.append(f"{rev - (max(4, n // 3) - 1 - i):<9} {pick(rng, ['kubectl set image ' + svc + '=registry.acme.internal/' + svc + ':v' + str(num(rng, 1, 4)) + '.' + str(num(rng, 0, 20)) + '.' + str(num(rng, 0, 9)), 'helm upgrade ' + svc + ' --set replicas=' + str(num(rng, 2, 12)), 'helm upgrade ' + svc + ' --set resources.limits.memory=' + pick(rng, ['512Mi', '1Gi', '2Gi']), '<none>'])}")
    lines += ["", f"$ kubectl get deploy {svc} -n {ns} -o wide", f"{'NAME':<18}{'READY':<8}{'UP-TO-DATE':<12}{'AVAILABLE':<11}{'AGE':<7}{'CONTAINERS':<14}{'IMAGES'}",
              f"{svc:<18}{num(rng, 0, 5)}/{num(rng, 4, 8)}{'':<4}{num(rng, 1, 8):<12}{num(rng, 0, 5):<11}{num(rng, 10, 300)}d{'':<4}{svc:<14}registry.acme.internal/{svc}:v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}"]
    if n > 10:
        lines += ["", f"$ kubectl rollout status deploy/{svc} -n {ns} --timeout=5s", f"Waiting for deployment \"{svc}\" rollout to finish: {num(rng, 1, 3)} of {num(rng, 4, 8)} updated replicas are available...",
                  "error: timed out waiting for the condition"]
        lines += ["", f"$ kubectl get events -n {ns} --field-selector involvedObject.name={svc} --sort-by=.lastTimestamp | tail -6"]
        for _ in range(num(rng, 3, 6)):
            lines.append(f"{num(rng, 1, 59)}m  {pick(rng, ['Normal', 'Warning'])}  {pick(rng, ['ScalingReplicaSet', 'ProgressDeadlineExceeded', 'FailedCreate', 'Injected'])}  deployment/{svc}  {pick(rng, ['Scaled up replica set ' + svc + '-' + sha(rng) + ' to ' + str(num(rng, 1, 8)), 'Scaled down replica set ' + svc + '-' + sha(rng) + ' to ' + str(num(rng, 0, 4)), 'ReplicaSet ' + svc + '-' + sha(rng) + ' has timed out progressing.'])}")
    v = rng.randrange(3)
    if v == 0:
        task = (f"The rollout of {svc} in {ns} is stuck. Find what changed in the last revisions, then read the manifest at deploy/{ns}/{svc}.yaml "
                f"and compare the image and the limits with the running deployment.")
        kind = "call"
    elif v == 1:
        task = (f"Write the change history of {svc} in {ns} as a table: revision, change cause, and whether it could explain a stuck rollout. "
                f"Then say which revision to roll back to, with the reason. No proposal yet.")
        kind = "report"
    else:
        task = (f"Confirm from the rollout output that {svc} in {ns} has fewer available replicas than desired, then propose a rollback to the "
                f"previous revision with propose_change, risk medium, with the exact command.")
        kind = "call"
    call = {"name": "kubectl", "arguments": {"args": f"rollout history deploy/{svc} -n {ns}"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_rollout"}


def ridge_probe(rng, n):
    ns = pick(rng, K_NS)
    svc = pick(rng, K_SVCS)
    lines = []
    for i in range(max(4, n // 3)):
        code = pick(rng, [200, 200, 503, 502, 504, 200, 429])
        lines += [f"$ curl -s -o /dev/null -w '%{{http_code}} %{{time_total}}\\n' http://{svc}.{ns}.svc.cluster.local:8080/{pick(rng, ['healthz', 'ready', 'v1/status', 'metrics'])}",
                  f"{code} {num(rng, 0, 9)}.{num(rng, 100, 999)}"]
    lines += ["", f"$ curl -s http://{svc}.{ns}.svc.cluster.local:8080/healthz | jq .",
              "{", f"  \"status\": \"{pick(rng, ['degraded', 'ok', 'unhealthy'])}\",", "  \"checks\": {", f"    \"db\": \"{pick(rng, ['ok', 'timeout after 2000ms', 'connection refused'])}\",",
              f"    \"cache\": \"{pick(rng, ['ok', 'ok', 'reconnecting'])}\",", f"    \"upstream_{pick(rng, K_SVCS)}\": \"{pick(rng, ['ok', '503', 'timeout'])}\",", f"    \"queue_depth\": {num(rng, 0, 4000)}", "  },",
              f"  \"version\": \"v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}\",", f"  \"uptime_s\": {num(rng, 30, 400000)}", "}"]
    if n > 10:
        lines += ["", f"$ dig +short {svc}.{ns}.svc.cluster.local"] + [f"10.0.{num(rng, 12, 15)}.{num(rng, 2, 250)}" for _ in range(num(rng, 1, 4))]
        lines += ["", f"$ curl -s http://{svc}.{ns}.svc.cluster.local:8080/metrics | grep -E 'http_requests_total|process_resident' | head -6"]
        for c in ["200", "429", "503"]:
            lines.append(f"http_requests_total{{code=\"{c}\",method=\"GET\"}} {num(rng, 100, 900000)}")
        lines.append(f"process_resident_memory_bytes {num(rng, 100000000, 1900000000)}")
    v = rng.randrange(2)
    if v == 0:
        task = (f"Alert {svc.capitalize().replace('-', '')}HighErrorRate fires for {ns}. Probe the service from the bastion, read the health payload, "
                f"and decide which dependency fails. Then get the pods of {svc} to see if all replicas show it.")
        kind = "call"
    else:
        task = (f"From the probe output for {svc} in {ns}, write the finding: which checks fail, the error share from the metrics, and whether "
                f"the problem is in {svc} or a dependency. Quote the numbers. Under 200 words.")
        kind = "report"
    call = {"name": "shell_ro", "arguments": {"command": f"for i in 1 2 3 4; do curl -s -o /dev/null -w '%{{http_code}} %{{time_total}}\\n' http://{svc}.{ns}.svc.cluster.local:8080/healthz; done; curl -s http://{svc}.{ns}.svc.cluster.local:8080/healthz | jq ."}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_probe"}


def ridge_manifest(rng, n):
    ns = pick(rng, K_NS)
    svc = pick(rng, K_SVCS)
    reps = num(rng, 2, 8)
    lines = ["apiVersion: apps/v1", "kind: Deployment", "metadata:", f"  name: {svc}", f"  namespace: {ns}", "  labels:", f"    app: {svc}", f"    team: {pick(rng, ['platform', 'payments', 'search', 'growth'])}",
             "spec:", f"  replicas: {reps}", "  strategy:", "    type: RollingUpdate", "    rollingUpdate:", "      maxSurge: 1", "      maxUnavailable: 0", "  selector:", "    matchLabels:", f"      app: {svc}",
             "  template:", "    metadata:", "      labels:", f"        app: {svc}", "    spec:", f"      serviceAccountName: {svc}", "      nodeSelector:", f"        pool: {pick(rng, ['general', 'memory', 'general'])}",
             "      containers:", f"        - name: {svc}", f"          image: registry.acme.internal/{svc}:v{num(rng, 1, 4)}.{num(rng, 0, 20)}.{num(rng, 0, 9)}", "          ports:", "            - containerPort: 8080",
             "          env:", "            - name: LOG_LEVEL", f"              value: {pick(rng, ['info', 'debug', 'warn'])}", f"            - name: BATCH_SIZE", f"              value: \"{num(rng, 100, 5000)}\"",
             "            - name: DB_URL", "              valueFrom:", "                secretKeyRef:", f"                  name: {svc}-secrets", "                  key: db_url",
             "          resources:", "            requests:", f"              cpu: {pick(rng, ['100m', '250m', '500m'])}", f"              memory: {pick(rng, ['128Mi', '256Mi', '512Mi'])}", "            limits:",
             f"              cpu: {pick(rng, ['500m', '1', '2'])}", f"              memory: {pick(rng, ['256Mi', '512Mi', '1Gi', '2Gi'])}", "          livenessProbe:", "            httpGet:", "              path: /healthz", "              port: 8080",
             f"            initialDelaySeconds: {num(rng, 5, 30)}", f"            timeoutSeconds: {num(rng, 1, 3)}", "            periodSeconds: 10", "          readinessProbe:", "            httpGet:", "              path: /ready", "              port: 8080",
             f"            initialDelaySeconds: {num(rng, 2, 10)}", "            periodSeconds: 5"]
    if n > 12:
        lines += ["---", "apiVersion: v1", "kind: ConfigMap", "metadata:", f"  name: {svc}-config", f"  namespace: {ns}", "data:", f"  MAX_CONNECTIONS: \"{num(rng, 10, 500)}\"", f"  QUEUE_LIMIT: \"{num(rng, 100, 5000)}\"",
                  f"  UPSTREAM_TIMEOUT_MS: \"{num(rng, 500, 8000)}\"", f"  FEATURE_{pick(rng, ['NEW_PARSER', 'BATCH_V2', 'ASYNC_EXPORT'])}: \"{pick(rng, ['true', 'false'])}\""]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Compare the manifest of {svc} in {ns} with the running pods: replicas, image tag, memory limit and probe timeouts. Get the pods "
                f"next and report every difference as a table.")
        kind = "call"
    else:
        task = (f"Review the manifest of {svc} in {ns} against the cluster rules: rolling update settings, requests versus limits, probe timing and "
                f"secrets handling. List each problem with the yaml path. Do not propose commands yet.")
        kind = "report"
    call = {"name": "read_manifest", "arguments": {"path": f"deploy/{ns}/{svc}.yaml"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "ridge_manifest"}


RIDGE = [ridge_get_pods, ridge_logs, ridge_describe, ridge_top, ridge_rollout, ridge_probe, ridge_manifest]

# --------------------------------------------------------------------------- QUILL (docs and release)
DOC_PAGES = ["docs/getting-started.md", "docs/install.md", "docs/sync.md", "docs/offline-mode.md", "docs/notifications.md", "docs/troubleshooting.md",
             "docs/privacy.md", "docs/account.md", "docs/widgets.md", "docs/shortcuts.md", "docs/faq.md", "docs/release-process.md", "docs/CHANGELOG.md",
             "docs/dev/build.md", "docs/dev/testing.md", "docs/dev/native-modules.md", "docs/dev/architecture.md", "docs/img/README.md"]
APP_COMMITS = ["Add the offline queue for drafts", "Fix the crash on rotate in the editor", "Show a banner when sync is paused", "Speed up the first launch by 40 percent",
               "Remove the legacy share sheet", "Add haptic feedback to the checklist", "Fix the wrong badge count after logout", "Support Android 15 edge to edge",
               "Add Dutch and Polish translations", "Fix the dark mode flash on start", "Bump react-native to 0.81", "Move the sync engine to a background task",
               "Add a setting to disable animations", "Fix the duplicate notification on resume", "Deprecate the v1 widget", "Add the export to PDF action",
               "Fix a memory leak in the image cache", "Update the privacy page for the new analytics opt-out", "Add tests for the offline queue", "Refactor the settings store"]
MDL_RULES = [("MD013", "line-length", "Line length [Expected: 100; Actual: {n}]"), ("MD022", "blanks-around-headings", "Headings should be surrounded by blank lines [Context: \"{h}\"]"),
             ("MD032", "blanks-around-lists", "Lists should be surrounded by blank lines"), ("MD040", "fenced-code-language", "Fenced code blocks should have a language specified"),
             ("MD041", "first-line-heading", "First line in a file should be a top-level heading"), ("MD034", "no-bare-urls", "Bare URL used"),
             ("MD024", "no-duplicate-heading", "Multiple headings with the same content [Context: \"{h}\"]"), ("MD009", "no-trailing-spaces", "Trailing spaces [Expected: 0 or 2; Actual: 1]"),
             ("MD029", "ol-prefix", "Ordered list item prefix [Expected: 1; Actual: 3; Style: 1/1/1]"), ("MD047", "single-trailing-newline", "Files should end with a single newline character")]
HEADINGS = ["## Install", "## Sync", "### Offline mode", "## Troubleshooting", "### Known issues", "## Shortcuts", "## Privacy", "### Export"]


def quill_git_log(rng, n):
    tag = f"v3.{num(rng, 0, 8)}.{num(rng, 0, 3)}"
    nxt = f"v3.{int(tag.split('.')[1]) + 1}.0"
    lines = []
    for _ in range(max(5, n // 2)):
        lines.append(f"{sha(rng)} {pick(rng, APP_COMMITS)}")
        for f in sample(rng, ["app/screens/Editor.tsx", "app/sync/queue.ts", "app/components/Banner.tsx", "ios/Podfile", "android/app/build.gradle",
                              "app/store/settings.ts", "app/i18n/nl.json", "app/i18n/pl.json", "docs/privacy.md", "app/notifications/handler.ts", "app/__tests__/queue.test.ts",
                              "app/widgets/v1/Widget.tsx", "app/export/pdf.ts", "app/cache/images.ts"], num(rng, 1, 3)):
            lines.append(f" {f:<36} | {num(rng, 2, 80):>3} {'+' * num(rng, 1, 8)}{'-' * num(rng, 0, 4)}")
        lines.append(f" {num(rng, 1, 3)} files changed, {num(rng, 3, 120)} insertions(+), {num(rng, 0, 50)} deletions(-)")
    v = rng.randrange(3)
    if v == 0:
        task = (f"Write the release notes for {nxt} from the git log since {tag}. Use the headings Added, Changed, Fixed and Removed. One bullet per user "
                f"visible commit, in the past tense, for a user of the app. Skip test only and refactor only commits.")
        kind = "report"
    elif v == 1:
        task = (f"Prepare the Unreleased section of docs/CHANGELOG.md from the commits since {tag}. Read the current changelog head first so the "
                f"format matches, then write the new section.")
        kind = "call"
    else:
        task = (f"From the log since {tag}, list the commits that touch a user facing screen or a translation and say which docs pages must change "
                f"for each. Table with commit, page and reason. Do not write files.")
        kind = "report"
    call = {"name": "git_cmd", "arguments": {"args": f"log --no-merges --stat --format='%h %s' {tag}..HEAD"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_git_log"}


def quill_markdownlint(rng, n):
    pages = sample(rng, DOC_PAGES, num(rng, 2, 5))
    lines = ["FAILED markdownlint: problems found"]
    for _ in range(max(5, n)):
        p = pick(rng, pages)
        code, name, msg = pick(rng, MDL_RULES)
        msg = msg.format(n=num(rng, 101, 180), h=pick(rng, HEADINGS))
        lines.append(f"{p}:{num(rng, 1, 200)}{':' + str(num(rng, 1, 80)) if rng.random() < 0.5 else ''} {code}/{name} {msg}")
    lines.sort()
    lines.append(f"Summary: {len(lines) - 1} problems in {len(set(l.split(':')[0] for l in lines[1:]))} files")
    if n > 12:
        p = pick(rng, pages)
        lines += ["", f"$ sed -n 1,16p {p}", f"{pick(rng, ['Getting started', 'Sync', 'Offline mode', 'Troubleshooting'])}", "=====", "",
                  f"This page explains {pick(rng, ['how the app syncs your notes', 'the offline queue', 'how to install the app', 'what to do when sync stops'])}.  ",
                  f"{pick(rng, HEADINGS)}", f"1. Open the app", f"3. Tap {pick(rng, ['Settings', 'Sync', 'Account'])}", f"3. Choose {pick(rng, ['Sync now', 'Sign in', 'Export'])}", "```",
                  f"adb logcat | grep {pick(rng, ['Sync', 'Queue', 'Widget'])}", "```", f"See https://brightpath.co/docs/{pick(rng, ['sync', 'install', 'faq'])} for details."]
    v = rng.randrange(3)
    top = max(set(l.split(":")[0] for l in lines[1:] if l.startswith("docs/")), key=lambda f: sum(1 for l in lines if l.startswith(f)))
    if v == 0:
        task = (f"markdownlint fails on the docs. Fix {top} first: read it, correct every finding without a change to the meaning, write it back, and run "
                f"markdownlint on that file again.")
        kind = "call"
    elif v == 1:
        task = (f"Summarize the markdownlint output: a table of rule, count and the fix for each rule in one sentence. Then name the file to fix "
                f"first and why. Do not edit anything.")
        kind = "report"
    else:
        task = (f"Open {top} and quote every line the linter flagged with its rule. Then decide whether each fix is mechanical or needs a rewrite. "
                f"Do not write yet.")
        kind = "call"
    call = {"name": "run_check", "arguments": {"name": "markdownlint", "args": "docs/"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_markdownlint"}


def quill_linkcheck(rng, n):
    lines = ["FAILED link-check: broken links found", ""]
    for _ in range(max(4, n // 2)):
        p = pick(rng, DOC_PAGES)
        k = rng.randrange(3)
        if k == 0:
            lines.append(f"{p}:{num(rng, 5, 200)}  [{pick(rng, ['sync guide', 'install steps', 'privacy policy', 'widget docs'])}]({pick(rng, ['sync.md', 'install.md', 'privacy.md', 'widgets-v1.md', '../dev/build.md', 'img/sync-flow.png'])})  -> file not found")
        elif k == 1:
            lines.append(f"{p}:{num(rng, 5, 200)}  https://brightpath.co/{pick(rng, ['docs/legacy', 'blog/2024/sync', 'help/widgets', 'status'])}  -> HTTP {pick(rng, [404, 410, 500])}")
        else:
            lines.append(f"{p}:{num(rng, 5, 200)}  [{pick(rng, ['Known issues', 'Shortcuts', 'Export'])}](#{pick(rng, ['known-issue', 'shortcut-list', 'export-pdf'])})  -> anchor not found")
    lines += ["", f"Checked {num(rng, 40, 300)} links in {num(rng, 10, 18)} files: {len(lines) - 3} broken, {num(rng, 0, 5)} skipped by the allow list"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Fix the {len(lines) - 4} broken links in the docs {pick(rng, ['before the release', 'for the v3.' + str(num(rng, 1, 9)) + ' docs build', 'that the nightly check found', 'flagged this morning'])}. "
                f"For each relative link, find the file that exists now with find_files, then correct the link. Mark external 404s as broken in the report "
                f"instead of changing them. Run link-check at the end.")
        kind = "call"
    else:
        task = (f"Write a table of the {len(lines) - 4} broken links {pick(rng, ['from the nightly check', 'from this run', 'the release manager asked about', 'in the docs tree'])}: "
                f"file, line, target, type (missing file, missing anchor, external) and the proposed fix. Say which ones need a decision from the release manager. Do not edit.")
        kind = "report"
    call = {"name": "run_check", "arguments": {"name": "link-check", "args": "docs/"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_linkcheck"}


def quill_find_files(rng, n):
    lines = sorted(set(sample(rng, DOC_PAGES, min(len(DOC_PAGES), max(6, n // 2))) + [f"docs/img/{pick(rng, ['sync-flow', 'install-ios', 'install-android', 'widget-v2', 'settings'])}.png" for _ in range(num(rng, 2, 6))]))
    if n > 10:
        lines += ["", "$ sed -n 1,30p mkdocs.yml", "site_name: Brightpath docs", "nav:", "  - Home: index.md", "  - Getting started: getting-started.md", "  - Install: install.md",
                  "  - Sync: sync.md", "  - Offline mode: offline-mode.md", "  - Troubleshooting: troubleshooting.md", "  - FAQ: faq.md", "  - Developers:", "      - Build: dev/build.md",
                  "      - Testing: dev/testing.md", "      - Native modules: dev/native-modules.md", "  - Changelog: CHANGELOG.md", "theme:", "  name: material", "markdown_extensions:", "  - admonition", "  - tables"]
    if n > 16:
        lines += ["", "$ wc -l docs/*.md docs/dev/*.md | sort -n | tail -12"]
        for p in sample(rng, DOC_PAGES, min(len(DOC_PAGES), num(rng, 8, 12))):
            lines.append(f"  {num(rng, 12, 400):>5} {p}")
        lines += ["", "$ sed -n 1,12p docs/index.md", "# Brightpath docs", "", "Welcome. Pick a page from the sidebar.", "", "- [Getting started](getting-started.md)", "- [Install](install.md)",
                  "- [Sync](sync.md)", f"- [{pick(rng, ['Widgets', 'Shortcuts', 'FAQ'])}]({pick(rng, ['widgets.md', 'shortcuts.md', 'faq.md'])})", "", f"Last updated: 2026-0{num(rng, 3, 9)}-{num(rng, 10, 28)}"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"The listing shows {sum(1 for l in lines if l.endswith('.md'))} markdown pages and {sum(1 for l in lines if l.endswith('.png'))} images. Which pages are missing from the nav in mkdocs.yml, and "
                f"which nav entries point to a file that does not exist? Answer with two lists from the listing. Do not edit.")
        kind = "report"
    else:
        task = (f"Add every doc page from the listing ({sum(1 for l in lines if l.endswith('.md'))} pages) that is missing from the nav to mkdocs.yml under "
                f"the right section. Read mkdocs.yml first and keep its order. Run mkdocs-build after the change.")
        kind = "call"
    call = {"name": "find_files", "arguments": {"glob": "**/*.md", "root": "docs"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_find_files"}


def quill_read_page(rng, n):
    page = pick(rng, [p for p in DOC_PAGES if "CHANGELOG" not in p and "img" not in p])
    title = page.split("/")[-1].replace(".md", "").replace("-", " ").capitalize()
    lines = [f"# {title}", "", f"This page is about {title.lower()}. It was written a while back and hasn't been touched since we were on 2.x so some of it might be out of date.", ""]
    for i in range(max(3, n // 5)):
        lines += [f"## {pick(rng, ['Overview', 'Setup', 'Usage', 'Known issues', 'Tips', 'Advanced', 'Limits', 'Permissions'])}", "",
                  pick(rng, ["You'll want to make sure you're syncing before doing anything else, otherwise things can get weird.",
                             "Basically the app is going to try and reconnect every 30s or so, and if it can't, it'll just keep queueing stuff up.",
                             "Note that we're deprecating the old widget so don't rely on it going forward.",
                             "If it doesn't work, try turning it off and on again (seriously, this fixes 90% of cases).",
                             "The export is a bit slow on older phones, we're looking into it.",
                             "Since 2.4 you can long-press the note to get the share menu, which is way more convenient.",
                             "Heads up: notifications won't fire when battery saver is on, that's an OS thing not us."]), "",
                  pick(rng, ["1. Open Settings\n2. Tap Sync\n3. Toggle 'Sync on cellular'", "- Go to Account\n- Tap Export\n- Pick PDF", "```\nadb shell am start -n co.brightpath/.Main\n```",
                             "| Setting | Default |\n|---|---|\n| Sync interval | 30s |\n| Retry | 5 |", "> Tip: you can also swipe left on a note to archive it."]), ""]
    v = rng.randrange(3)
    if v == 0:
        task = (f"Rewrite {page} in Simplified Technical English: active voice, short sentences, one instruction per step, no idioms, ASCII only. "
                f"Keep every fact and every heading. Return the full page in a markdown block. Do not write the file.")
        kind = "report"
    elif v == 1:
        task = (f"Bring {page} up to the house style and write it back with write_text. Keep the facts and the structure. Then run markdownlint "
                f"and ascii-check on the file.")
        kind = "call"
    else:
        task = (f"Review {page} against the writing rules: list every sentence that breaks a rule with the rule name and a corrected version. "
                f"Table with line, problem and fix. Do not edit the file.")
        kind = "report"
    call = {"name": "read_text", "arguments": {"path": page, "start": 1, "end": 120}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_read_page"}


def quill_ascii(rng, n):
    lines = ["FAILED ascii-check: non-ASCII characters found"]
    bad = ["em dash (U+2014)", "en dash (U+2013)", "right single quotation mark (U+2019)", "left double quotation mark (U+201C)", "right double quotation mark (U+201D)",
           "ellipsis (U+2026)", "non-breaking space (U+00A0)", "bullet (U+2022)", "arrow (U+2192)", "check mark (U+2713)", "multiplication sign (U+00D7)"]
    for _ in range(max(5, n)):
        lines.append(f"{pick(rng, DOC_PAGES)}:{num(rng, 1, 220)}:{num(rng, 1, 90)}  {pick(rng, bad)}")
    lines.sort()
    lines.append(f"Summary: {len(lines) - 1} characters in {len(set(l.split(':')[0] for l in lines[1:]))} files")
    v = rng.randrange(2)
    top = max(set(l.split(":")[0] for l in lines[1:-1]), key=lambda f: sum(1 for l in lines if l.startswith(f)))
    if v == 0:
        task = (f"The docs must be ASCII. Fix {top} first: read the flagged lines, replace each character with its ASCII form, write the file, and "
                f"run ascii-check on it again.")
        kind = "call"
    else:
        task = (f"Group the {len(lines) - 2} ascii-check findings by character with the ASCII replacement for each, then by file with a count. Two tables. "
                f"Do not edit any file.")
        kind = "report"
    call = {"name": "run_check", "arguments": {"name": "ascii-check", "args": "docs/"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_ascii"}


def quill_git_diff(rng, n):
    page = pick(rng, [p for p in DOC_PAGES if "img" not in p])
    lines = [f"diff --git a/{page} b/{page}", f"index {sha(rng)}..{sha(rng)} 100644", f"--- a/{page}", f"+++ b/{page}"]
    for _ in range(max(1, n // 7)):
        a = num(rng, 5, 150)
        lines += [f"@@ -{a},{num(rng, 5, 8)} +{a},{num(rng, 6, 10)} @@ {pick(rng, HEADINGS)}", " ",
                  f"-{pick(rng, ['Sync runs every 30 seconds while the app is open.', 'The widget shows the last 5 notes.', 'Export is available on iOS only.', 'Tap Settings, then Sync.'])}",
                  f"+{pick(rng, ['Sync runs every 15 seconds while the app is open and every 5 minutes in the background.', 'The widget shows the last 10 notes. The v1 widget is deprecated.', 'Export to PDF is available on iOS and Android.', 'Open Settings. Tap Sync. Tap Sync now.'])}",
                  f"+", f"+{pick(rng, ['> Note: background sync needs the battery saver off.', '| Platform | Interval |', '![export](img/export.png)', 'See [troubleshooting](troubleshooting.md) when sync stops.'])}", " "]
    if n > 10:
        lines += [f"diff --git a/docs/CHANGELOG.md b/docs/CHANGELOG.md", f"index {sha(rng)}..{sha(rng)} 100644", "--- a/docs/CHANGELOG.md", "+++ b/docs/CHANGELOG.md", "@@ -1,6 +1,12 @@", " # Changelog", " ", " ## Unreleased", " ",
                  "+### Added", "+", f"+- {pick(rng, APP_COMMITS)}", f"+- {pick(rng, APP_COMMITS)}", "+", "+### Fixed", "+", f"+- {pick(rng, APP_COMMITS)}", f" ## [v3.{num(rng, 0, 8)}.0] - 2026-0{num(rng, 6, 9)}-{num(rng, 10, 28)}"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Review the pending docs diff. Check every changed sentence against the writing rules and against the code facts you can verify "
                f"from git. List the problems with line numbers and a corrected sentence. Do not edit.")
        kind = "report"
    else:
        task = (f"The diff for {page} claims a new sync interval. Verify it in the code: search the sync module in app/sync/ with git grep and "
                f"quote the constant. Then say whether the doc is right.")
        kind = "call"
    call = {"name": "git_cmd", "arguments": {"args": "diff -- docs/"}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "quill_git_diff"}


QUILL = [quill_git_log, quill_markdownlint, quill_linkcheck, quill_find_files, quill_read_page, quill_ascii, quill_git_diff]

# --------------------------------------------------------------------------- GRAIN (data)
JOBS = ["ingest.orders.daily", "ingest.events.hourly", "ingest.customers.daily", "transform.staging.orders", "transform.mart.revenue", "ingest.shipments.daily",
        "transform.staging.dedupe", "export.invoices.monthly", "ingest.pageviews.hourly", "transform.mart.churn"]
TABLES = ["raw.orders", "raw.events", "raw.customers", "staging.orders", "staging.events", "mart.revenue_daily", "mart.churn_monthly", "raw.shipments", "staging.customers", "mart.job_row_counts"]
SOURCES = ["orders", "events", "customers", "shipments", "pageviews", "invoices"]


def runid(rng):
    return f"run_202609{num(rng, 10, 18)}_{num(rng, 0, 23):02d}{pick(rng, ['00', '15', '30'])}"


def grain_job_status(rng, n):
    job = pick(rng, JOBS)
    rid = runid(rng)
    state = pick(rng, ["failed", "failed", "failed", "succeeded_with_warnings", "running"])
    lines = [f"job_id: {job}", f"run_id: {rid}", f"state: {state}", f"started_at: 2026-09-18T{num(rng, 0, 6):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d}Z",
             f"finished_at: {'2026-09-18T' + str(num(rng, 3, 8)).zfill(2) + ':' + str(num(rng, 0, 59)).zfill(2) + ':' + str(num(rng, 0, 59)).zfill(2) + 'Z' if state != 'running' else 'null'}",
             f"duration_s: {num(rng, 30, 5400)}", f"attempt: {num(rng, 1, 3)}", "inputs:"]
    src = job.split(".")[1] if job.split(".")[1] in SOURCES else pick(rng, SOURCES)
    for _ in range(num(rng, 1, 3)):
        lines.append(f"  - s3://lumen-landing/{src}/2026/09/{num(rng, 10, 18)}/{src}-{num(rng, 0, 23):02d}.csv.gz ({num(rng, 1, 900)} MB)")
    lines += ["row_counts:"]
    for t in sample(rng, TABLES, num(rng, 2, 4)):
        lines.append(f"  {t}: {num(rng, 0, 9000000):,}")
    lines += ["last_log_lines:"]
    t0 = num(rng, 0, 3000)
    msgs = ["INFO  loaded {n:,} rows from {f}", "INFO  transform {t} took {ms} ms", "WARN  {n:,} rows rejected: {reason}", "WARN  schema drift: column {col} is {ty} in source, expected {ty2}",
            "ERROR {exc}", "INFO  checkpoint written at offset {n}", "WARN  late data: {n:,} rows older than the watermark", "ERROR duplicate key on {t}: ({k}) appears {n} times",
            "INFO  freshness check: {t} max(_updated_at) = 2026-09-{d}T{h}:00Z"]
    for i in range(max(6, n)):
        m = pick(rng, msgs).format(n=num(rng, 1, 3000000), f=f"{src}-{num(rng, 0, 23):02d}.csv.gz", t=pick(rng, TABLES), ms=num(rng, 100, 90000), reason=pick(rng, ["null order_id", "bad timestamp", "unknown currency", "negative amount"]),
                                   col=pick(rng, ["amount", "created_at", "customer_id", "country"]), ty=pick(rng, ["text", "float", "int"]), ty2=pick(rng, ["numeric(12,2)", "timestamp", "bigint"]),
                                   exc=pick(rng, ["psycopg.errors.NumericValueOutOfRange: value overflows numeric format", "FileNotFoundError: s3://lumen-landing/" + src + "/2026/09/18/" + src + "-03.csv.gz", "MemoryError: unable to allocate 4.2 GiB", "psycopg.errors.UniqueViolation: duplicate key value violates unique constraint", "TimeoutError: warehouse did not respond within 600 s", "ValueError: could not convert string to float: 'N/A'"]),
                                   k=f"order_id={num(rng, 100000, 999999)}", d=num(rng, 10, 18), h=num(rng, 0, 23))
        t0 += num(rng, 1, 90)
        lines.append(f"  2026-09-18T{(t0 // 3600) % 24:02d}:{(t0 // 60) % 60:02d}:{t0 % 60:02d}Z {m}")
    v = rng.randrange(3)
    if v == 0:
        task = (f"Run {rid} of {job} failed. Find the cause from the job status, then run one query that confirms it in the warehouse, and say "
                f"whether a retry is safe. Do not retry yet.")
        kind = "call"
    elif v == 1:
        task = (f"Write the summary for the orchestrator on {job} run {rid}: the state, the cause with the log line that shows it, the rows affected, "
                f"and whether the mart tables are stale because of it. Under 250 words.")
        kind = "report"
    else:
        task = (f"Check whether the warnings in {job} run {rid} explain a drop in the row count. Compare the counts with mart.job_row_counts for the "
                f"last 7 runs with one query.")
        kind = "call"
    call = {"name": "job_status", "arguments": {"job_id": job, "run_id": rid}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_job_status"}


def grain_query(rng, n):
    t = pick(rng, TABLES)
    kind_q = rng.randrange(3)
    lines = []
    if kind_q == 0:
        lines = [f" day        | rows     | distinct_keys | null_customer | max_amount", "------------+----------+---------------+---------------+-----------"]
        base = num(rng, 20000, 900000)
        for i in range(max(5, n // 2)):
            rows = int(base * (1 + rng.uniform(-0.15, 0.15)))
            if i == max(5, n // 2) - 2:
                rows = int(rows * pick(rng, [0.1, 2.4, 0.0]))
            lines.append(f" 2026-09-{18 - (max(5, n // 2) - 1 - i):02d} | {rows:>8,} | {int(rows * rng.uniform(0.8, 1.0)):>13,} | {num(rng, 0, 3000):>13,} | {num(rng, 100, 99999)}.{num(rng, 10, 99)}")
        sql = f"SELECT date_trunc('day', created_at)::date AS day, count(*) AS rows, count(DISTINCT id) AS distinct_keys, count(*) FILTER (WHERE customer_id IS NULL) AS null_customer, max(amount) AS max_amount FROM {t} WHERE created_at >= now() - interval '{max(5, n // 2)} days' GROUP BY 1 ORDER BY 1 LIMIT 200"
    elif kind_q == 1:
        lines = [f" job_id                      | run_id               | table_name        | rows      | ran_at", "-----------------------------+----------------------+-------------------+-----------+---------------------"]
        j = pick(rng, JOBS)
        for i in range(max(5, n // 2)):
            lines.append(f" {j:<27} | {runid(rng):<20} | {pick(rng, TABLES):<17} | {num(rng, 0, 9000000):>9,} | 2026-09-{num(rng, 10, 18):02d} {num(rng, 0, 23):02d}:{num(rng, 0, 59):02d}:00")
        sql = f"SELECT job_id, run_id, table_name, rows, ran_at FROM mart.job_row_counts WHERE job_id = '{j}' ORDER BY ran_at DESC LIMIT 200"
    else:
        lines = [f" id      | customer_id | amount    | currency | created_at          | _source_file", "---------+-------------+-----------+----------+---------------------+----------------------"]
        for i in range(max(5, n // 2)):
            cid = num(rng, 1000, 99999)
            lines.append(f" {num(rng, 100000, 999999):<7} | {cid if rng.random() > 0.2 else '':<11} | {pick(rng, ['', '-']) if rng.random() < 0.15 else ''}{num(rng, 1, 99999)}.{num(rng, 10, 99):<7} | {pick(rng, ['EUR', 'USD', 'EU', 'usd', 'GBP']):<8} | 2026-09-{num(rng, 10, 18):02d} {num(rng, 0, 23):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d} | orders-{num(rng, 0, 23):02d}.csv.gz")
        sql = f"SELECT id, customer_id, amount, currency, created_at, _source_file FROM {t} WHERE amount < 0 OR customer_id IS NULL OR currency NOT IN ('EUR','USD','GBP') ORDER BY created_at DESC LIMIT 200"
    lines.append(f"({len(lines) - 2} rows)")
    v = rng.randrange(3)
    if v == 0:
        task = (f"The revenue dashboard shows a jump for {t.split('.')[-1]}. Use the query result to find the day that differs, then run one more query "
                f"that splits that day by _source_file to see where the rows came from.")
        kind = "call"
    elif v == 1:
        task = (f"Explain the anomaly in the query result for `{t}` in two sentences, quote the exact numbers, and say whether it is a duplicate load, "
                f"a missing file or a real change. Propose the check that would settle it. No more queries.")
        kind = "report"
    else:
        task = (f"Validate `{t}`: from the rows returned, list every data quality problem you see (nulls, negative amounts, bad currency codes, "
                f"duplicates), count each, and propose the SQL fix for the staging transform. Do not run it.")
        kind = "report"
    call = {"name": "run_query", "arguments": {"sql": sql, "limit": 200}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_query"}


def grain_list_bucket(rng, n):
    src = pick(rng, SOURCES)
    day = num(rng, 10, 18)
    lines = [f"{'key':<58}{'size':>12}  last_modified"]
    hours = sorted(sample(rng, list(range(24)), min(24, max(6, n))))
    for h in hours:
        size = num(rng, 20, 900) * 1024 * 1024
        if rng.random() < 0.12:
            size = 0
        lines.append(f"{'s3://lumen-landing/' + src + '/2026/09/' + str(day) + '/' + src + '-' + str(h).zfill(2) + '.csv.gz':<58}{size:>12,}  2026-09-{day}T{h:02d}:{num(rng, 2, 59):02d}:{num(rng, 0, 59):02d}Z")
    if rng.random() < 0.5:
        lines.append(f"{'s3://lumen-landing/' + src + '/2026/09/' + str(day) + '/_SUCCESS':<58}{0:>12,}  2026-09-{day}T23:{num(rng, 50, 59)}:00Z")
    lines.append(f"({len(lines) - 1} objects, {sum(int(l.split()[1].replace(',', '')) for l in lines[1:]) // (1024 * 1024):,} MB)")
    v = rng.randrange(2)
    if v == 0:
        task = (f"The {src} load for 2026-09-{day} looks short. From the bucket listing, name the hours that are missing or empty, then check the "
                f"latest run of ingest.{src}.{'hourly' if src in ('events', 'pageviews') else 'daily'} to see which files it read.")
        kind = "call"
    else:
        task = (f"Audit the landing files for {src} on 2026-09-{day}: list the missing hours, the zero size files and the total size, and say "
                f"whether the source or the ingest job is at fault. Table plus three sentences.")
        kind = "report"
    call = {"name": "list_bucket", "arguments": {"prefix": f"{src}/2026/09/{day}/", "max_keys": 100}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_list_bucket"}


def grain_read_csv(rng, n):
    src = pick(rng, SOURCES)
    path = f"/lake/landing/{src}/2026/09/{num(rng, 10, 18)}/{src}-{num(rng, 0, 23):02d}.csv"
    cols = {"orders": "id,customer_id,amount,currency,status,created_at", "events": "id,user_id,name,payload,ts", "customers": "id,name,email,country,created_at",
            "shipments": "id,order_id,carrier,shipped_at,delivered_at", "pageviews": "id,path,user_id,ts,referrer", "invoices": "id,customer_id,amount,due_date,paid_at"}[src]
    lines = [cols]
    for i in range(max(8, n)):
        k = rng.random()
        if src == "orders":
            row = f"{num(rng, 100000, 999999)},{num(rng, 1000, 99999) if k > 0.1 else ''},{num(rng, 1, 9999)}.{num(rng, 10, 99)},{pick(rng, ['EUR', 'USD', 'GBP']) if k > 0.15 else pick(rng, ['EU', 'usd', 'N/A'])},{pick(rng, ['paid', 'pending', 'refunded'])},2026-09-{num(rng, 10, 18):02d}T{num(rng, 0, 23):02d}:{num(rng, 0, 59):02d}:00{pick(rng, ['Z', '+02:00', '', '-05:00']) if k > 0.2 else ''}"
        elif src == "events":
            row = f"{num(rng, 10**8, 10**9)},{num(rng, 1000, 99999)},{pick(rng, ['click', 'view', 'signup', 'purchase'])},\"{{\"\"page\"\": \"\"/{pick(rng, ['home', 'pricing', 'docs'])}\"\"}}\",2026-09-{num(rng, 10, 18):02d} {num(rng, 0, 23):02d}:{num(rng, 0, 59):02d}:{num(rng, 0, 59):02d}"
        elif src == "customers":
            row = f"{num(rng, 1000, 99999)},{pick(rng, NAMES).capitalize()} {pick(rng, ['A.', 'B.', 'K.', 'M.'])},{pick(rng, NAMES)}{num(rng, 1, 99)}@example.{pick(rng, ['org', 'net'])},{pick(rng, ['DE', 'FR', 'US', 'GB', '', 'de'])},2026-0{num(rng, 1, 9)}-{num(rng, 10, 28)}"
        elif src == "shipments":
            row = f"{num(rng, 10000, 99999)},{num(rng, 100000, 999999)},{pick(rng, ['dhl', 'ups', 'DHL', 'gls'])},2026-09-{num(rng, 10, 18):02d},{('2026-09-' + str(num(rng, 10, 18)).zfill(2)) if k > 0.3 else ''}"
        elif src == "pageviews":
            row = f"{num(rng, 10**8, 10**9)},/{pick(rng, ['home', 'pricing', 'docs/sync', 'blog'])},{num(rng, 1000, 99999) if k > 0.4 else ''},2026-09-{num(rng, 10, 18):02d}T{num(rng, 0, 23):02d}:{num(rng, 0, 59):02d}:00Z,{pick(rng, ['google', 'direct', '', 'newsletter'])}"
        else:
            row = f"{num(rng, 10000, 99999)},{num(rng, 1000, 99999)},{num(rng, 1, 99999)}.{num(rng, 10, 99)},2026-{num(rng, 9, 12):02d}-{num(rng, 1, 28):02d},{('2026-09-' + str(num(rng, 10, 18)).zfill(2)) if k > 0.5 else ''}"
        if rng.random() < 0.06:
            row += ","
        lines.append(row)
    v = rng.randrange(2)
    if v == 0:
        task = (f"The {src} load rejects rows. Look at the head of {path}, name every format problem you see with an example row, and then run "
                f"a short pandas snippet that counts each problem over the whole file.")
        kind = "call"
    else:
        task = (f"From the sample rows of {path}, list the data quality problems with one example each, say which ones the staging transform "
                f"already handles, and propose the rule for the rest as SQL. Do not run anything.")
        kind = "report"
    call = {"name": "read_data_file", "arguments": {"path": path, "max_lines": num(rng, 20, 60)}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_read_csv"}


def grain_python(rng, n):
    src = pick(rng, SOURCES)
    cols = ["id", "customer_id", "amount", "currency", "created_at"] if src in ("orders", "invoices") else ["id", "user_id", "name", "ts"]
    lines = [f"shape: ({num(rng, 10000, 3000000):,}, {len(cols) + 2})", "", "dtypes:"]
    for c in cols + ["_loaded_at", "_source_file"]:
        lines.append(f"  {c:<14} {pick(rng, ['int64', 'object', 'float64', 'datetime64[ns, UTC]', 'object'])}")
    lines += ["", "null counts:"]
    for c in cols:
        lines.append(f"  {c:<14} {num(rng, 0, 40000):,}")
    lines += ["", "describe(amount):" if "amount" in cols else "value_counts(name):"]
    if "amount" in cols:
        for k, val in (("count", f"{num(rng, 10000, 3000000):,}"), ("mean", f"{num(rng, 10, 900)}.{num(rng, 10, 99)}"), ("std", f"{num(rng, 10, 9000)}.{num(rng, 10, 99)}"), ("min", f"-{num(rng, 1, 99999)}.{num(rng, 10, 99)}"),
                       ("25%", f"{num(rng, 5, 50)}.{num(rng, 10, 99)}"), ("50%", f"{num(rng, 50, 200)}.{num(rng, 10, 99)}"), ("75%", f"{num(rng, 200, 900)}.{num(rng, 10, 99)}"), ("max", f"{num(rng, 100000, 99999999)}.{num(rng, 10, 99)}")):
            lines.append(f"  {k:<8} {val:>16}")
    else:
        for nm in ["view", "click", "signup", "purchase", "VIEW", "null"]:
            lines.append(f"  {nm:<10} {num(rng, 0, 900000):>10,}")
    lines += ["", "duplicates by id:", f"  {num(rng, 0, 50000):,} rows share an id with another row", f"  top: id={num(rng, 100000, 999999)} x{num(rng, 2, 40)}", "",
              "created_at range:" if "created_at" in cols else "ts range:", f"  min 2026-0{num(rng, 1, 8)}-{num(rng, 10, 28)} {num(rng, 0, 23):02d}:00:00+00:00", f"  max 2026-09-{num(rng, 17, 19)} {num(rng, 0, 23):02d}:59:00+00:00",
              f"  rows in the future: {num(rng, 0, 900)}"]
    for _ in range(max(0, n // 8)):
        lines.append(f"  rows with offset {pick(rng, ['+02:00', '-05:00', '+05:30'])}: {num(rng, 0, 50000):,}")
    if rng.random() < 0.4:
        lines += ["", "/usr/lib/python3/dist-packages/pandas/core/dtypes/cast.py:1234: FutureWarning: Setting an item of incompatible dtype is deprecated"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Interpret the profile of raw.{src} ({lines[0].split('(')[1].split(',')[0]} rows): which columns have problems, how many rows each problem "
                f"touches, and whether the dedupe transform covers the duplicates. Then propose the staging rule as SQL. No more tool calls.")
        kind = "report"
    else:
        task = (f"The profile of raw.{src} shows negative or extreme amounts and future timestamps. Run one query that returns the {num(rng, 10, 50)} worst "
                f"rows by amount with their _source_file, so we can trace the file.")
        kind = "call"
    call = {"name": "run_python", "arguments": {"code": f"import pandas as pd\ndf = pd.read_parquet('/lake/raw/{src}/2026-09-18.parquet')\nprint('shape:', df.shape)\nprint(df.dtypes)\nprint(df.isna().sum())\nprint(df['{'amount' if 'amount' in cols else 'name'}'].{'describe()' if 'amount' in cols else 'value_counts(dropna=False)'})\nprint(df.duplicated('id').sum())", "timeout_s": 60}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_python"}


def grain_transform_sql(rng, n):
    src = pick(rng, ["orders", "events", "customers", "shipments"])
    path = f"transforms/staging/{src}.sql"
    lines = [f"-- {path}: build staging.{src} from raw.{src}", f"-- owner: data-platform, last change: 2026-0{num(rng, 4, 9)}-{num(rng, 10, 28)}", "",
             f"CREATE OR REPLACE TABLE staging.{src} AS", "WITH src AS (", f"    SELECT", "        id,",
             f"        {pick(rng, ['customer_id', 'user_id'])},", f"        {pick(rng, ['CAST(amount AS NUMERIC(12,2)) AS amount,', 'amount,', 'ROUND(amount::numeric, 2) AS amount,'])}",
             f"        {pick(rng, ['UPPER(currency) AS currency,', 'currency,', 'COALESCE(currency, ' + chr(39) + 'EUR' + chr(39) + ') AS currency,'])}",
             f"        {pick(rng, ['created_at::timestamp AS created_at,', 'created_at AT TIME ZONE ' + chr(39) + 'UTC' + chr(39) + ' AS created_at,', 'created_at,'])}",
             "        _loaded_at,", "        _source_file", f"    FROM raw.{src}", f"    WHERE _loaded_at >= now() - interval '{num(rng, 1, 7)} days'", "),",
             "ranked AS (", "    SELECT *,", f"        ROW_NUMBER() OVER (PARTITION BY id ORDER BY _loaded_at {pick(rng, ['DESC', 'ASC'])}) AS rn", "    FROM src", ")",
             "SELECT", "    id,", f"    {pick(rng, ['customer_id', 'user_id'])},", "    amount,", "    currency,", "    created_at,", "    now() AS _updated_at", "FROM ranked",
             f"WHERE rn = 1{pick(rng, ['', ' AND amount >= 0', ' AND currency IN (' + chr(39) + 'EUR' + chr(39) + ',' + chr(39) + 'USD' + chr(39) + ',' + chr(39) + 'GBP' + chr(39) + ')'])};"]
    for _ in range(max(0, n // 10)):
        lines += ["", f"-- index for the dashboard join", f"CREATE INDEX IF NOT EXISTS staging_{src}_{pick(rng, ['created_at', 'customer_id'])}_idx ON staging.{src} ({pick(rng, ['created_at', 'customer_id'])});"]
    v = rng.randrange(2)
    if v == 0:
        task = (f"Review {path} ({len(lines)} lines). Say whether the dedupe keeps the newest or the oldest row, whether the currency and amount rules "
                f"reject bad rows or silently change them, and whether the time zone handling is right. Propose the corrected SQL in a block. Do not run it.")
        kind = "report"
    else:
        task = (f"The mart shows {num(rng, 12, 9000):,} duplicate ids for {src}. Read the staging transform, decide whether the window in it can produce "
                f"duplicates, then run one query on staging.{src} that counts ids with more than one row.")
        kind = "call"
    call = {"name": "read_data_file", "arguments": {"path": path, "max_lines": 80}}
    return {"task": task, "call": call, "result": "\n".join(lines), "kind": kind, "tag": "grain_transform_sql"}


GRAIN = [grain_job_status, grain_query, grain_list_bucket, grain_read_csv, grain_python, grain_transform_sql]
