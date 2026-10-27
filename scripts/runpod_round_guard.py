"""Cost guard for this round's explicitly registered disposable Runpod resources.

Uses Runpod's official v2 API. Never discovers resources by name for deletion.
Credentials are read in memory, excluded from the registry and all logs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def credential(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == "RUNPOD_API_KEY":
                return value.strip().strip("\"'")
    raise RuntimeError("Runpod credential unavailable")


def cli(args, env_file):
    env = dict(os.environ, RUNPOD_API_KEY=credential(env_file))
    result = subprocess.run(["runpodctl", *args], env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        # Return only structured error fields, redacting the credential defensively.
        message = "provider output suppressed"
        try:
            error = json.loads(result.stderr.split("\nUsage:")[0])
            message = str(error.get("error", error.get("message", "unspecified")))
            message = message.replace(env["RUNPOD_API_KEY"], "[REDACTED]")[:600]
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"Runpod command failed: {args[0]}, exit {result.returncode}: {message}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def request(method, path, token):
    req = urllib.request.Request("https://api.runpod.io" + path, method=method,
                                 headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return 404, None
        raise RuntimeError(f"Runpod HTTP {exc.code}") from None


def estimated_cost(state, now):
    return sum(max(0, (r.get("ended_at") or now) - r["created_at"]) * r["hourly_rate"] / 3600
               for r in state["resources"])


def expire_reason(state, now, balance=None):
    if now >= state["deadline_unix"]:
        return "round_deadline"
    if estimated_cost(state, now) >= state["kill_at_usd"]:
        return "estimated_spend_limit"
    if balance is not None and state["opening_balance_usd"] - balance >= state["kill_at_usd"]:
        return "account_balance_limit"
    return None


def terminate_registered_pods(state, token, log):
    for resource in state["resources"]:
        if resource["kind"] != "pod" or resource.get("ended_at"):
            continue
        resource_id = resource["id"]
        if not resource_id.isalnum():
            raise ValueError("Invalid registered resource id")
        status, _ = request("DELETE", "/v2/pods/" + resource_id, token)
        check, _ = request("GET", "/v2/pods/" + resource_id, token)
        log({"event": "pod_termination", "id": resource_id, "status": status, "absent": check == 404})
        if check != 404:
            raise RuntimeError("Termination read-back has not confirmed absence")
    # Network volume deliberately survives emergency compute termination so checkpoints
    # can be retrieved. Its cost stays counted by the local guard until archived/deleted.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--heartbeat", required=True)
    args = parser.parse_args()
    token = credential(args.env_file)
    heartbeat = Path(args.heartbeat)
    def log(event):
        print(json.dumps({"utc_epoch": time.time(), **event}), flush=True)
    log({"event": "watchdog_started", "pid": os.getpid()})
    while True:
        state = json.loads(Path(args.registry).read_text())
        now = time.time()
        balance = None
        try:
            balance = float(cli(["user"], args.env_file)["clientBalance"])
        except Exception as exc:
            log({"event": "balance_read_failed", "error_type": type(exc).__name__})
        reason = expire_reason(state, now, balance)
        record = {"pid": os.getpid(), "time": now, "balance_usd": balance,
                  "estimated_cost_usd": estimated_cost(state, now), "reason": reason}
        heartbeat.write_text(json.dumps(record))
        if reason:
            log({"event": "limit_reached", **record})
            try:
                terminate_registered_pods(state, token, log)
            except Exception as exc:
                log({"event": "termination_retry", "error_type": type(exc).__name__})
        if state.get("closed"):
            log({"event": "watchdog_closed"})
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
