#type: ignore
import sys

print("PROBE: python version =", sys.version)
print("PROBE: python executable =", getattr(sys, "executable", "n/a"))

try:
    import gnubg
    print("PROBE: gnubg python OK")
    print("PROBE: gnubg dir =", [a for a in dir(gnubg) if not a.startswith("__")])
    try:
        gnubg.command("new session")
        print("PROBE: new session OK")
        b = gnubg.board()
        print("PROBE: board type =", type(b).__name__, "len =", len(b) if b is not None else None)
        print("PROBE: board =", b)
        m = gnubg.match(0)
        print("PROBE: match(0) keys =", list(m.keys()) if isinstance(m, dict) else type(m).__name__)
    except Exception as e:
        print("PROBE: command/board error:", repr(e))
except Exception as e:
    print("PROBE: gnubg import FAILED:", repr(e))
