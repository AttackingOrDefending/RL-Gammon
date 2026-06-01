#type: ignore
"""Standalone smoke test for the gnubg bridge: POST a few commands and print the JSON."""
import json
import sys

import requests

URL = "http://localhost:8001"


def post(command):
    resp = requests.post(url=URL, data={"command": command}, timeout=30)
    return resp.json()


def main():
    r = post("new session")
    print("SMOKE: keys =", list(r.keys()))
    print("SMOKE: error =", r.get("error"))
    board = r.get("board")
    print("SMOKE: board len =", len(board) if board else None)
    print("SMOKE: board =", json.dumps(board))
    print("SMOKE: last_move =", json.dumps(r.get("last_move"))[:400])
    print("SMOKE: info =", json.dumps(r.get("info")))

    # turn off automatic play so the agent drives the turns
    for cmd in ("set automatic roll off", "set automatic game off"):
        rr = post(cmd)
        print("SMOKE:", cmd, "-> error =", rr.get("error"))

    if not board or len(board) != 2:
        print("SMOKE: FAIL - no valid board returned")
        sys.exit(1)
    print("SMOKE: OK")


if __name__ == "__main__":
    main()
