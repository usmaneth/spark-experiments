#!/usr/bin/env python3
"""
12-Hour Automated Watcher for Claude Code Session: 'Mafia terminal panel testing'
Target Session ID: 46b8af19-f68e-45fa-a0f6-bf11e06ce650
Process PID: 1407421 (TTY: /dev/pts/7)

Monitors session transcript and terminal for Claude usage/rate limits.
When a limit is detected:
1. Extracts the exact reset time / duration.
2. Sleeps until the limit expires (+ 30s safety buffer).
3. Injects 'continue\\n' directly into /dev/pts/7 via kernel TIOCSTI ioctl.
4. Verifies resumption and continues watching for 12 hours.
"""

import os
import sys
import time
import json
import re
import datetime
import subprocess
import glob

SESSION_ID = "46b8af19-f68e-45fa-a0f6-bf11e06ce650"
TARGET_TITLE = "Mafia terminal panel testing"
INITIAL_PID = 1407421
INITIAL_TTY = "/dev/pts/7"

TRANSCRIPT_PATH = f"/home/REDACTED/.claude/projects/-home-REDACTED/{SESSION_ID}.jsonl"
SESSION_JSON_PATH = f"/home/REDACTED/.claude/sessions/{INITIAL_PID}.json"
LOG_FILE = "/home/REDACTED/Bonsai-demo/dflash-training/claude_watcher.log"

WATCH_DURATION_SEC = 12 * 3600 # 12 hours
CHECK_INTERVAL_SEC = 3.0       # Polling interval

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def find_active_target():
    # 1. Check initial PID
    if os.path.exists(f"/proc/{INITIAL_PID}"):
        tty = INITIAL_TTY
        try:
            tty = os.readlink(f"/proc/{INITIAL_PID}/fd/0")
        except Exception:
            pass
        return INITIAL_PID, tty, TRANSCRIPT_PATH

    # 2. Check all active Claude sessions in ~/.claude/sessions/
    session_files = glob.glob("/home/REDACTED/.claude/sessions/*.json")
    for sf in session_files:
        try:
            with open(sf, "r") as f:
                data = json.load(f)
            pid = data.get("pid")
            sid = data.get("sessionId")
            if pid and os.path.exists(f"/proc/{pid}"):
                # Check transcript for title
                trans = f"/home/REDACTED/.claude/projects/-home-REDACTED/{sid}.jsonl"
                if os.path.exists(trans):
                    with open(trans, "r") as tf:
                        for l in tf:
                            if TARGET_TITLE in l:
                                tty = os.readlink(f"/proc/{pid}/fd/0")
                                return pid, tty, trans
        except Exception:
            pass
    return None, None, None

def inject_continue_keystrokes(tty_path):
    log(f"Injecting 'continue\\n' into {tty_path} via root TIOCSTI...")
    cmd = [
        "sudo", "/usr/bin/python3", "-c",
        f"import os, fcntl, termios; "
        f"fd = os.open('{tty_path}', os.O_WRONLY); "
        f"[fcntl.ioctl(fd, termios.TIOCSTI, bytes([b])) for b in b'continue\\n']; "
        f"os.close(fd)"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            log(f"Successfully injected 'continue\\n' into {tty_path}!")
            return True
        else:
            log(f"TIOCSTI injection failed: {res.stderr}")
    except Exception as e:
        log(f"Error injecting keystrokes: {e}")
    return False

def parse_reset_duration(text):
    """
    Parses wait time from rate limit strings:
    - 'try again in 4h 12m'
    - 'in 45m' / 'in 30 minutes'
    - 'until 3:00 PM' / 'until 15:30'
    - 'resets at 4:00 PM'
    """
    now = datetime.datetime.now()

    # Pattern 1: 'in Xh Ym' or 'in X hours Y minutes'
    m_hm = re.search(r'in\s+(\d+)\s*h(?:ours?)?(?:\s+(\d+)\s*m(?:in(?:utes?)?)?)?', text, re.IGNORECASE)
    if m_hm:
        hours = int(m_hm.group(1))
        mins = int(m_hm.group(2)) if m_hm.group(2) else 0
        return hours * 3600 + mins * 60 + 20

    # Pattern 2: 'in Xm' or 'in X minutes'
    m_m = re.search(r'in\s+(\d+)\s*m(?:in(?:utes?)?)?', text, re.IGNORECASE)
    if m_m:
        mins = int(m_m.group(1))
        return mins * 60 + 20

    # Pattern 3: 'until HH:MM AM/PM' or 'resets at HH:MM AM/PM'
    m_clock = re.search(r'(?:until|resets? at)\s+(\d{1,2}):(\d{2})(?:\s*(AM|PM))?', text, re.IGNORECASE)
    if m_clock:
        hr = int(m_clock.group(1))
        minute = int(m_clock.group(2))
        ampm = m_clock.group(3)
        if ampm:
            ampm = ampm.upper()
            if ampm == "PM" and hr < 12: hr += 12
            if ampm == "AM" and hr == 12: hr = 0
        
        target = now.replace(hour=hr, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        diff_sec = (target - now).total_seconds()
        return int(diff_sec) + 30 # 30s safety buffer

    return None

def check_for_rate_limits(transcript_path, last_checked_pos):
    """
    Scans new lines added to the transcript since last_checked_pos.
    """
    if not os.path.exists(transcript_path):
        return last_checked_pos, None

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_checked_pos)
            new_lines = f.readlines()
            new_pos = f.tell()

        for line in new_lines:
            if not line.strip(): continue
            # Look for limit indicators
            low = line.lower()
            if any(term in low for term in ["usage limit", "rate limit", "limit reached", "wait until", "resets at", "too many requests", "429"]):
                # Found potential limit
                duration = parse_reset_duration(line)
                return new_pos, (line.strip(), duration)

        return new_pos, None
    except Exception as e:
        log(f"Error reading transcript: {e}")
        return last_checked_pos, None

def main():
    log("==================================================================")
    log(f"Starting 12-Hour Watcher for Claude Session: '{TARGET_TITLE}'")
    log(f"Target PID: {INITIAL_PID} | Target TTY: {INITIAL_TTY}")
    log(f"Duration: {WATCH_DURATION_SEC // 3600} hours | Log: {LOG_FILE}")
    log("==================================================================")

    start_time = time.time()
    last_heartbeat = 0
    last_file_pos = 0

    # Initialize file position to current end of transcript
    if os.path.exists(TRANSCRIPT_PATH):
        last_file_pos = os.path.getsize(TRANSCRIPT_PATH)

    while time.time() - start_time < WATCH_DURATION_SEC:
        pid, tty, transcript = find_active_target()
        now = time.time()
        elapsed_min = (now - start_time) / 60.0
        remaining_min = (WATCH_DURATION_SEC - (now - start_time)) / 60.0

        if now - last_heartbeat >= 60.0:
            last_heartbeat = now
            alive_str = f"PID {pid} Alive on {tty}" if pid else "PID Not Found"
            log(f"[HEARTBEAT] {alive_str} | Elapsed: {elapsed_min:.1f}m / 720m (Remaining: {remaining_min:.1f}m)")

        if not pid or not transcript:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        # Check transcript for new limit events
        last_file_pos, limit_event = check_for_rate_limits(transcript, last_file_pos)

        if limit_event:
            raw_text, duration_sec = limit_event
            log("******************************************************************")
            log(f"[LIMIT DETECTED] {raw_text[:200]}")
            
            if duration_sec:
                wait_min = duration_sec / 60.0
                reset_time = datetime.datetime.now() + datetime.timedelta(seconds=duration_sec)
                log(f"[ACTION] Parsed wait duration: {wait_min:.1f} min ({duration_sec}s)")
                log(f"[ACTION] Target reset time: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                log(f"[ACTION] Pausing watcher. Will inject 'continue\\n' at exact reset time...")
                
                # Sleep in increments so process monitoring stays intact
                sleep_start = time.time()
                while time.time() - sleep_start < duration_sec:
                    left = int(duration_sec - (time.time() - sleep_start))
                    if left % 300 == 0 or left < 60:
                        log(f"[COUNTDOWN] {left // 60}m {left % 60}s remaining until reset...")
                    time.sleep(min(10, left))

                log("[TIMER EXPIRED] Reset window elapsed! Preparing injection...")
                time.sleep(5) # Extra 5s safety margin
                
                # Re-verify PID and TTY
                cur_pid, cur_tty, _ = find_active_target()
                target_tty = cur_tty or tty or INITIAL_TTY
                
                # Send 'continue\n'
                success = inject_continue_keystrokes(target_tty)
                if success:
                    log("[SUCCESS] Session resumed automatically without manual intervention!")
                    time.sleep(15)
                    # Reset transcript pointer to current end to prevent re-trigger
                    if os.path.exists(transcript):
                        last_file_pos = os.path.getsize(transcript)
            else:
                log("[WARNING] Could not parse exact duration from limit text. Retrying in 60s...")
                time.sleep(60)

        time.sleep(CHECK_INTERVAL_SEC)

    log("==================================================================")
    log(f"12-Hour Watcher Completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("==================================================================")

if __name__ == "__main__":
    main()
