"""
Serve the evaluation report at http://localhost/evals

Serves the docs/ folder (docs/index.html + docs/data.json) under /evals with
Python's built-in web server. Starts instantly: no models, database or API
key are loaded. Listens on this computer only (127.0.0.1).

    python eval/serve_report.py              # http://localhost/evals   (port 80)
    python eval/serve_report.py --port 8080  # http://localhost:8080/evals
"""

import argparse
import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
PREFIX = "/evals"


class ReportHandler(SimpleHTTPRequestHandler):
    """Maps /evals/<file> to docs/<file>; everything else redirects to /evals/."""

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == PREFIX or not path.startswith(PREFIX + "/"):
            self.send_response(301)
            self.send_header("Location", PREFIX + "/")
            self.end_headers()
            return
        self.path = self.path[len(PREFIX):]
        super().do_GET()

    def do_HEAD(self):
        self.do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(DOCS_DIR, "data.json")):
        sys.exit("docs/data.json not found. Build it first: python eval/build_report_data.py")

    handler = functools.partial(ReportHandler, directory=DOCS_DIR)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as e:
        sys.exit(f"Could not use port {args.port} ({e}). It may be taken; try --port 8080.")

    suffix = "" if args.port == 80 else f":{args.port}"
    print(f"Evaluation report: http://localhost{suffix}/evals   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
