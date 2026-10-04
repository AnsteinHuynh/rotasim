"""
Console logging that never loses a crash.

Every entry script routes through here so that:
  * everything printed to stdout/stderr also lands in <project>/debug/<ts>-<name>.jsonl,
    one JSON object per line: {"ts": ..., "stream": "stdout"|"stderr", "msg": "<line>"}
  * the file is flushed per line, so a hard crash cannot swallow the last output
  * an unhandled exception writes the FULL traceback as the final jsonl line and the
    process exits nonzero -- the "console closed before I could read it" failure mode

usage:
    from dreamsim_oft.logutil import run
    if __name__ == "__main__":
        raise SystemExit(run("probe-backbones", main))
    # or, with args:
        raise SystemExit(run("train", main, argv_style_callable))

Writes: <project>/debug/<yyyymmdd-hhmmss>-<name>.jsonl   (project root = 2 levels up from this file)
"""

from __future__ import annotations

import datetime
import io
import json
import os
import sys
import traceback
from pathlib import Path

# <project>/src/dreamsim_oft/logutil.py -> <project>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "debug"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class _Tee(io.TextIOBase):
    """Mirror a text stream into a per-line JSONL file, flushing as we go."""

    def __init__(self, original, fh, stream_name: str):
        self._original = original
        self._fh = fh
        self._stream_name = stream_name
        self._buf = ""

    # -- io.TextIOBase surface ------------------------------------------------
    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            self._original.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        if self._buf:
            self._emit(self._buf)
            self._buf = ""
        try:
            self._fh.flush()
        except Exception:
            pass

    def isatty(self) -> bool:  # keep tqdm/rich from misdetecting a terminal
        return False

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._original, "encoding", "utf-8")

    def _emit(self, line: str) -> None:
        try:
            self._fh.write(json.dumps({"ts": _now(), "stream": self._stream_name, "msg": line}) + "\n")
            self._fh.flush()
        except Exception:
            pass


class _Session:
    def __init__(self, path: Path, fh):
        self.path = path
        self.fh = fh
        self._old_out = None
        self._old_err = None

    def install(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_out, self.fh, "stdout")
        sys.stderr = _Tee(self._old_err, self.fh, "stderr")

    def restore(self):
        for tee, old in ((sys.stdout, self._old_out), (sys.stderr, self._old_err)):
            if isinstance(tee, _Tee):
                tee.flush()
        if self._old_out is not None:
            sys.stdout = self._old_out
        if self._old_err is not None:
            sys.stderr = self._old_err


def start(name: str, debug_dir: os.PathLike | str | None = None) -> _Session:
    d = Path(debug_dir) if debug_dir else DEFAULT_DEBUG_DIR
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = d / f"{stamp}-{name}.jsonl"
    sess = _Session(path, open(path, "a", encoding="utf-8"))
    sess.install()
    print(f"[logutil] console log -> {path}")
    return sess


def _raw_emit(sess: _Session, stream: str, msg: str) -> None:
    """Write straight to the jsonl, bypassing the tee (used for the final traceback)."""
    try:
        sess.fh.write(json.dumps({"ts": _now(), "stream": stream, "msg": msg}) + "\n")
        sess.fh.flush()
    except Exception:
        pass


def run(name: str, impl, *args, debug_dir=None, **kwargs) -> int:
    """Run impl(*args, **kwargs) under console+jsonl capture. Returns a process exit code."""
    sess = start(name, debug_dir=debug_dir)
    code = 0
    try:
        impl(*args, **kwargs)
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        tb = traceback.format_exc()
        sess.restore()  # get the tee out of the way so the traceback is readable
        sys.stderr.write(tb)
        sys.stderr.flush()
        _raw_emit(sess, "stderr", tb)
        code = 1
    finally:
        sess.restore()
        try:
            sess.fh.close()
        except Exception:
            pass
    return code
