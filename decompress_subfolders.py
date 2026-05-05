"""
Script to decompress .tar.br archives (tar + brotli).
Shows one progress bar per active archive + an overall bar.

The one difference vs compression: the bar tracks compressed bytes read from the
.tar.br file (since that's what _BrotliReader sees chunk by chunk), not uncompressed
bytes. The total is just archive_path.stat().st_size, which is instant to get.
The bar will still be smooth and accurate — it just reflects how far through the
compressed file you are rather than how much data has been inflated.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
import tarfile
import argparse
import threading
import shutil

import brotli
from tqdm import tqdm


class _BrotliReader:
    def __init__(self, f, queue=None, name: str = ""):
        self._f = f
        self._decompressor = brotli.Decompressor()
        self._buf = b""
        self._queue = queue
        self._name = name

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buf) < size:
            chunk = self._f.read(65536)
            if not chunk:
                break
            if self._queue is not None:
                self._queue.put(("progress", self._name, len(chunk)))
            self._buf += self._decompressor.process(chunk)

        if size < 0:
            data, self._buf = self._buf, b""
        else:
            data, self._buf = self._buf[:size], self._buf[size:]
        return data

    def readable(self) -> bool:
        return True


def decompress_archive(archive_path: Path, target_dir_path: Path, queue=None) -> str:
    target_path = target_dir_path / archive_path.stem  # strips .tar.br → folder name

    if target_path.exists():
        if queue:
            queue.put(("done", archive_path.name, 0))
        return f"already done {archive_path.name}"

    # Compressed file size drives the progress bar.
    total_bytes = archive_path.stat().st_size
    if queue:
        queue.put(("start", archive_path.name, total_bytes))

    tmp_path = target_path.with_suffix(".tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        raw_file = archive_path.open("rb")
        reader = _BrotliReader(raw_file, queue, archive_path.name)

        with tarfile.open(fileobj=reader, mode="r|") as tar:  # ty:ignore[no-matching-overload]
            tar.extractall(path=tmp_path)

        raw_file.close()
        tmp_path.rename(target_path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    finally:
        if queue:
            queue.put(("done", archive_path.name, 0))

    return f"done {archive_path.name}"


def _listen(queue, n_slots: int, overall: tqdm):
    """
    Manages per-archive tqdm bars.

    Protocol messages from workers:
      ("start",    name, total_bytes)  — archive started, open a bar
      ("progress", name, n_bytes)      — n compressed bytes read
      ("done",     name, 0)            — archive finished, close bar
      None                             — sentinel: shut down
    """
    bars: dict[str, tqdm] = {}
    free_slots: list[int] = list(range(1, n_slots + 1))

    while True:
        msg = queue.get()
        if msg is None:
            break

        kind, name, value = msg

        if kind == "start":
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
            overall.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompress .tar.br archives produced by the companion compress script."
    )
    parser.add_argument("folder", help="Directory containing .tar.br archives")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel decompression workers (default: 2)",
    )
    args = parser.parse_args()

    source_path = Path(args.folder)
    if not source_path.is_dir():
        raise FileNotFoundError(f"{args.folder} not found")

    target_dir_path = source_path.parent / (
        source_path.stem.replace("_COMPRESSED", "") + "_DECOMPRESSED"
    )
    target_dir_path.mkdir(parents=True, exist_ok=True)

    archive_paths = sorted(source_path.glob("*.tar.br"))
    if not archive_paths:
        print("No .tar.br archives found — nothing to do.")
        raise SystemExit(0)

    print(
        f"Decompressing {len(archive_paths)} archive(s) → {target_dir_path}  "
        f"[{args.workers} workers]"
    )

    manager = Manager()
    queue = manager.Queue()

    overall = tqdm(
        total=len(archive_paths),
        desc="Overall",
        unit="archive",
        position=0,
        dynamic_ncols=True,
    )
    listener = threading.Thread(
        target=_listen, args=(queue, args.workers, overall), daemon=True
    )
    listener.start()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(decompress_archive, p, target_dir_path, queue): p
            for p in archive_paths
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
