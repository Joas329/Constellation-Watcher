import threading

from .channel import Channel

class Stage:
    # one thread, one input channel, one output channel. Subclasses only
    # implement process(); lifecycle and dedup live here once.
    def __init__(self, name, source: Channel | None, sink: Channel | None):
        self.name = name
        self.source = source
        self.sink = sink
        self._thread = None
        self._stop = threading.Event()
        self.dropped = 0 # frames superseded while we were busy

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"{self.name} already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=False)
        self._thread.start()

    def stop(self, timeout=10.0):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError(f"{self.name} did not exit within {timeout}s")
        self._thread = None

    def _loop(self):
        last_seq = 0
        while not self._stop.is_set():
            item = self.source.wait(last_seq)
            if item is None:
                continue # timeout, re-check stop

            payload, capture_time, seq = item
            self.dropped += seq - last_seq - 1
            last_seq = seq

            result = self.process(payload, capture_time)

            if result is not None and self.sink is not None:
                self.sink.publish(result, capture_time)

    def process(self, payload, capture_time):
        raise NotImplementedError