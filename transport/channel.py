import time
import threading

class Channel:
    # single-slot, latest-wins. Consumers block; slow ones skip frames rather
    # than queue them, which is the correct policy for a live sky.
    def __init__(self, name):
        self.name = name
        self._cond = threading.Condition()
        self._payload = None
        self._capture_time = None
        self._seq = 0

    def publish(self, payload, capture_time):
        if payload is None:
            raise RuntimeError(f"Channel[{self.name}]: publish(None)")
        with self._cond:
            self._payload = payload
            self._capture_time = capture_time
            self._seq += 1
            self._cond.notify_all()

    def wait(self, last_seq, timeout=5.0):
        # returns None on timeout so callers can re-check their stop event
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._payload, self._capture_time, self._seq

    def peek(self):
        # non-blocking snapshot for one-shot HTTP readers (stats, overlay,
        # solved_frame). Returns the current value without waiting for a new one.
        with self._cond:
            if self._seq == 0:
                return None
            return self._payload, self._capture_time, self._seq