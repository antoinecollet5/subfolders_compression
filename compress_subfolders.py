"""
Script to compress folders to .tar.br (tar + brotli) archives.
Lossless, max compression ratio (quality=11, window=24).
Shows one progress bar per active folder + an overall bar.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from pathlib import Path
import tarfile
import argparse
import threading
import os

import brotli
from tqdm import tqdm


BROTLI_QUALITY = 0  # 0–11  │ 11 = best ratio; drop to 6 for ~3× speed
BROTLI_LGWIN = 24  # 10–24 │ 24 = largest window → best ratio on large inputs
_UPDATE_EVERY = 1 * 1024 * 1024  # send a progress tick every 1 MB
_queue = None


def _init_worker(q):
    global _queue
    _queue = q


class _BrotliWriter:
    def __init__(self, f, compressor, name: str = ""):
        self._f = f
        self._compressor = compressor
        self._name = name
        self._pending = 0  # bytes accumulated since last tick

    def write(self, data: bytes) -> int:
        self._f.write(self._compressor.process(data))
        self._pending += len(data)
        if _queue is not None and self._pending >= _UPDATE_EVERY:
            _queue.put(("progress", self._name, self._pending))
            self._pending = 0
        return len(data)

    def flush(self):
        """Flush remaining accumulated bytes after tar streaming is done."""
        if _queue is not None and self._pending > 0:
            _queue.put(("progress", self._name, self._pending))
            self._pending = 0


def make_archive(simu_path: Path, target_dir_path: Path) -> str:
    if simu_path.is_file():
        return f"skip (file)  {simu_path.name}"

    target_path = target_dir_path / (simu_path.stem + ".tar.br")
    if target_path.exists():
        if _queue:
            _queue.put(("done", simu_path.name, 0))
        return f"already done {simu_path.name}"

    # Total uncompressed size → drives the per-file progress bar.
    total_bytes = sum(f.stat().st_size for f in simu_path.rglob("*") if f.is_file())
    if _queue:
        _queue.put(("start", simu_path.name, total_bytes))

    tmp_path = target_path.with_suffix(".tmp")
    try:
        compressor = brotli.Compressor(quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
        raw_file = tmp_path.open("wb")
        writer = _BrotliWriter(raw_file, compressor, simu_path.name)

        with tarfile.open(fileobj=writer, mode="w|") as tar:  # ty:ignore[no-matching-overload]
            tar.add(simu_path, arcname=simu_path.name)

        writer.flush()  # report the last partial chunk
        raw_file.write(compressor.finish())
        raw_file.close()
        tmp_path.rename(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if _queue:
            _queue.put(("done", simu_path.name, 0))

    return f"done {simu_path.name}"


def _listen(queue, n_slots: int, overall: tqdm):
    """
    Manages per-folder tqdm bars.

    Protocol messages from workers:
      ("start",    name, total_bytes)  — folder started, open a bar
      ("progress", name, n_bytes)      — n input bytes processed
      ("done",     name, 0)            — folder finished, close bar
      None                             — sentinel: shut down
    """
    bars: dict[str, tqdm] = {}
    free_slots: list[int] = list(range(1, n_slots + 1))
    finished_early: set[str] = set()  # "done" arrived before "start"

    while True:
        msg = queue.get()
        if msg is None:
            break

        kind, name, value = msg

        if kind == "start":
            if name in finished_early:  # already done, skip opening a bar
                finished_early.discard(name)
                continue
            pos = free_slots.pop(0) if free_slots else len(bars) + 1
            bars[name] = tqdm(
                total=value,
                desc=name[:35],
                unit="B",
                unit_scale=True,
                position=pos,
                leave=False,
                dynamic_ncols=True,
            )
            bars[name]._slot = pos  # ty:ignore[unresolved-attribute]

        elif kind == "progress":
            if name in bars:
                bars[name].update(value)

        elif kind == "done":
            if name in bars:
                slot = getattr(bars[name], "_slot", None)
                bars[name].close()
                del bars[name]
                if slot is not None:
                    free_slots.append(slot)
                    free_slots.sort()
            else:
                finished_early.add(name)  # "start" may still be in flight
            overall.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress sub-folders to .tar.br with brotli."
    )
    parser.add_argument("folder", help="Source directory whose sub-folders to compress")
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Parallel compression workers (default: -1, i.e., all CPU cores)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=BROTLI_QUALITY,
        choices=range(0, 12),
        metavar="[0-11]",
        help="Brotli quality level (default: 0 = fastest)",
    )
    args = parser.parse_args()
    BROTLI_QUALITY = args.quality

    n_workers = os.cpu_count() if args.workers == -1 else args.workers

    source_path = Path(args.folder)
    if not source_path.is_dir():
        raise FileNotFoundError(f"{args.folder} not found")

    target_dir_path = source_path.parent.joinpath(f"{source_path.stem}_COMPRESSED")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    simu_paths = sorted(p for p in source_path.glob("*") if p.is_dir())
    if not simu_paths:
        print("No sub-directories found — nothing to do.")
        raise SystemExit(0)

    print(
        f"Compressing {len(simu_paths)} folder(s) → {target_dir_path}  "
        f"[brotli q{BROTLI_QUALITY}, w{BROTLI_LGWIN}, {n_workers} workers]"
    )

    queue = Queue()

    overall = tqdm(
        total=len(simu_paths),
        desc="Overall",
        unit="folder",
        position=0,
        dynamic_ncols=True,
    )
    listener = threading.Thread(
        target=_listen, args=(queue, n_workers, overall), daemon=True
    )
    listener.start()

    # Queue is injected once at worker startup — no per-call pickling
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(queue,),
    ) as executor:
        futures = {
            executor.submit(make_archive, p, target_dir_path): p for p in simu_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                overall.write(f"ERROR — {futures[future].name}: {exc}")

    queue.put(None)  # shut down the listener
    listener.join()
    overall.close()
    print("All done.")
