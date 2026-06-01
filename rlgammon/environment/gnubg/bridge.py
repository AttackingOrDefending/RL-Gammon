#type: ignore

# THIS FILE MUST BE RUN INSIDE gnubg (it does `import gnubg`):
#     gnubg -t -q -p rlgammon/environment/gnubg/bridge.py
# gnubg 1.07 embeds Python 3.11, so the Python 3 branches below are used.
# (The Python 2.7 fallbacks are kept in case an older gnubg build is used.)
import json
import sys
import traceback

import gnubg

try:
    # Python 3
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
except ImportError:
    # Python 2.7
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from urlparse import parse_qs, urlparse


def _log(*args):
    print("[bridge]", *args)
    sys.stdout.flush()


def _to_jsonable(obj):
    # gnubg returns tuples (board, dice, move); JSON only knows lists. Recurse so the
    # client always receives plain lists/dicts/scalars it can parse with response.json().
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):

    def _set_headers(self, response=200):
        self.send_response(response)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Quiet the default noisy per-request logging; we log commands explicitly.
        return

    def _build_response(self, command):
        response = {"board": [], "last_move": [], "info": [], "error": None}

        prev_game = gnubg.match(0)["games"][-1]["game"] if gnubg.match(0) else []

        gnubg.command(command)

        # check if the game is started/exists (handle the case the command executed is set at the beginning)
        if gnubg.match(0):
            # get the board after the execution of a move
            response["board"] = _to_jsonable(gnubg.board())

            # get the last games
            games = gnubg.match(0)["games"][-1]

            # get the last game entry
            game = games["game"][-1] if games["game"] else None

            # save the state of the game before and after having executed a command
            response["last_move"] = [_to_jsonable(prev_game), _to_jsonable(game)]

            # save the info of all games played so far
            for g in gnubg.match(0)["games"]:
                info = g["info"]

                response["info"].append(
                    {
                        "winner": info["winner"],
                        "n_moves": len(g["game"]),
                        "resigned": info["resigned"] if "resigned" in info else None,
                    },
                )

        return response

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length).decode("utf-8")
        data = parse_qs(post_data)

        command = data.get("command", [""])[0]
        _log("command:", repr(command))

        try:
            response = self._build_response(command)
        except Exception as exc:  # keep the server alive on any gnubg error
            _log("ERROR running command", repr(command), ":", repr(exc))
            traceback.print_exc()
            sys.stdout.flush()
            response = {"board": [], "last_move": [], "info": [], "error": str(exc)}

        self._set_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        _ = parsed.path
        self._set_headers()
        self.wfile.write(b"Hello! Welcome to Backgammon WebGUI")


def run(host, server_class=HTTPServer, handler_class=Handler, port=8001):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
    _log("Starting httpd {}:{} (python {})".format(host, port, sys.version.split()[0]))
    httpd.serve_forever()


if __name__ == "__main__":
    HOST = "localhost"  # <-- YOUR HOST HERE
    PORT = 8001  # <-- YOUR PORT HERE
    run(host=HOST, port=PORT)
