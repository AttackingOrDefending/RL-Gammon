# Playing against GNU Backgammon (gnubg)

This package lets a trainable agent play full games against
[GNU Backgammon](https://www.gnu.org/software/gnubg/) (`gnubg`). It works by running a
small HTTP server **inside** gnubg's embedded Python interpreter
([`bridge.py`](bridge.py)), and talking to it over HTTP from normal Python
([`gnubg_backgammon.py`](gnubg_backgammon.py), [`gnu_testing.py`](../../trainer/testing/gnu_testing.py)).

```
                HTTP POST  {command: "..."}
  your python  ───────────────────────────────►  gnubg (embedded python 3.11)
  (requests)   ◄───────────────────────────────  bridge.py  →  import gnubg
                JSON  {board, last_move, info}
```

## Requirements

* `gnubg` installed on the machine that will run the bridge. This repo was verified on
  **WSL Debian** with `gnubg 1.07.001` at `/usr/games/gnubg`.
* gnubg must be built **with Python support** (the `-p`/`--python` flag). Verify with the
  probe below.
* Normal `python3` with the project deps (`torch`, `requests`, the `rlgammon` package).
  The client side does **not** need gnubg.

> All commands below are written for **WSL Debian**. From Windows you can prefix any of
> them with `wsl -d Debian -e bash -lc "cd /mnt/c/.../RL-Gammon && <CMD>"`.

## 0. (Once) Verify gnubg has Python scripting

```bash
/usr/games/gnubg --help | grep -- --python          # should show "-p, --python=FILE"
/usr/games/gnubg -t -q -p rlgammon/environment/gnubg/probe.py 2>&1 | grep PROBE
```

Expected (Python version may differ):

```
PROBE: python version = 3.11.2 ...
PROBE: gnubg python OK
PROBE: new session OK
PROBE: board type = tuple len = 2
```

If you instead see `gnubg import FAILED`, your gnubg build has **no Python support** and
the bridge cannot run — you would need a gnubg built `--with-python`.

## 1. Start the bridge (must be running first!)

The bridge serves forever on **`localhost:8001`**. Start it in the background and leave it
running while you test:

```bash
# from the repo root, on the machine with gnubg:
/usr/games/gnubg -t -q -p rlgammon/environment/gnubg/bridge.py > /tmp/gnubg_bridge.log 2>&1 &
BRIDGE_PID=$!
# wait until it is listening, then check the log:
sleep 5 && grep "Starting httpd" /tmp/gnubg_bridge.log
# -> [bridge] Starting httpd localhost:8001 (python 3.11.2)
```

Quick smoke test of the round-trip (separate normal `python3`):

```bash
PYTHONPATH=. python3 rlgammon/environment/gnubg/smoke_client.py
# -> SMOKE: board len = 2 ... SMOKE: OK
```

Stop the bridge when you are done (**kill by PID**, see caveats):

```bash
kill $BRIDGE_PID
```

## 2. Run a GNU test from Python

With the bridge running, run a test from normal `python3`:

```python
from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.testing.gnu_testing import GNUTesting

agent = TDAgent()                       # fresh untrained net is fine for a plumbing test
results = GNUTesting(episodes_in_test=1).test(agent)
print(results)
# e.g. {'win_rate': 0.0, 'draws': 0.0, 'losses': 1.0, 'points_white': 0.0, 'points_black': 2.0}
```

A ready-made version of the snippet above is in
[`example_gnu_test.py`](example_gnu_test.py). The agent always plays **WHITE**
(gnubg's `O` / "pantidis" seat); gnubg plays **BLACK** (`X`). A freshly initialised network
is expected to lose — the point of this test is that a **legal, complete** game runs and a
results dict comes back without exceptions.

## One-shot helper (start bridge → run client → stop bridge)

[`run_with_bridge.sh`](run_with_bridge.sh) does the whole dance in a single shell session
(starts the bridge, waits for the port, runs your script with a timeout, then kills the
bridge by PID):

```bash
# bash run_with_bridge.sh <python-script> [timeout-seconds]
bash rlgammon/environment/gnubg/run_with_bridge.sh rlgammon/environment/gnubg/example_gnu_test.py 70
```

Copy-pasteable from Windows:

```powershell
wsl -d Debian -e bash -lc "cd /mnt/c/Users/panti/PycharmProjects/RL-Gammon && bash rlgammon/environment/gnubg/run_with_bridge.sh rlgammon/environment/gnubg/example_gnu_test.py 70"
```

## Caveats

* **The bridge must be started before** any client / `GNUTesting` call, and it must stay
  running for the whole test. The client just does HTTP `POST`s to `localhost:8001`.
* **Port 8001 is hard-coded** on both sides (`bridge.py` and
  `GnubgInterface("localhost", 8001)` in `gnu_testing.py`). Change both if you need a
  different port.
* **gnubg embeds Python 3.11** in this build, so `bridge.py` runs under Python 3 (it keeps
  Python-2 import fallbacks for older builds, but they are not used here).
* `gnubg.board()` and the move/dice fields come back as **tuples**; the bridge converts them
  to JSON lists before sending, and the client parses lists.
* **Do NOT clean up with `pkill -f bridge.py`.** When you launch the bridge from a shell, the
  string `bridge.py` is part of the launching command line, so `pkill -f bridge.py` matches
  (and kills) the launching shell itself. Always kill the bridge by the **PID** you captured
  (`kill $BRIDGE_PID`), as `run_with_bridge.sh` does.
* `PYTHONPATH` must include the repo root when running a script directly with
  `python3 path/to/script.py` (running from the repo root with `PYTHONPATH=.` is enough);
  `run_with_bridge.sh` sets this for you.
* On WSL, run the bridge and the client in the **same WSL session** (or keep one WSL session
  alive), otherwise the distro may tear down the background gnubg process.

## Files

| file | purpose |
|------|---------|
| [`bridge.py`](bridge.py) | HTTP server run **inside** gnubg (`gnubg -t -p bridge.py`). Exposes `gnubg.command/board/match` over POST. |
| [`gnubg_backgammon.py`](gnubg_backgammon.py) | Client: `GnubgInterface` (HTTP + parsing), `GnubgEnv`, `evaluate_vs_gnubg`. |
| [`probe.py`](probe.py) | One-off check that gnubg's embedded Python and the `gnubg` module work. |
| [`smoke_client.py`](smoke_client.py) | Minimal HTTP round-trip test against a running bridge. |
| [`example_gnu_test.py`](example_gnu_test.py) | Runs `GNUTesting(...).test(TDAgent())` end to end. |
| [`run_with_bridge.sh`](run_with_bridge.sh) | Convenience: start bridge → run a client script → stop bridge. |
