"""
Script to decompress .tar.zst archives (tar + zstd).
Shows one progress bar per active archive + an overall bar.

The bar tracks compressed bytes read from the .tar.zst file (since that's what
the progress reader sees chunk by chunk), not uncompressed bytes. The total is
just archive_path.stat().st_size, which is instant to get. The bar will still
be smooth and accurate — it just reflects how far through the compressed file
you are rather than how much data has been inflated.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from pathlib import Path
import tarfile
import argparse
import threading
import shutil
import os

import zstandard as zstd
from tqdm import tqdm

# Permit the larger windows produced by long-range mode (LDM) at compression
# time. zstd's default cap of 128 MiB rejects them; 2 GiB is well above any
# realistic LDM window and still prevents pathological memory bombs.
ZSTD_MAX_WINDOW = 2**31

_queue = None


def _init_worker(q):
    global _queue
    _queue = q


class _ProgressReader:
    """File-like wrapper that counts compressed bytes pulled from the raw
    file and reports them to the progress queue, then forwards the data
    to whoever called .read() (the zstd stream_reader)."""

    def __init__(self, inner, name: str = ""):
        self._inner = inner
        self._name = name

    def read(self, size: int = -1) -> bytes:
        chunk = self._inner.read(size)
        if chunk and _queue is not None:
            _queue.put(("progress", self._name, len(chunk)))
        return chunk

    def readable(self) -> bool:
        return True

    def close(self):
        self._inner.close()


def decompress_archive(archive_path: Path, target_dir_path: Path) -> str:
    # Strip both ".tar" and ".zst" so "foo.tar.zst" → folder "foo"
    folder_name = archive_path.name.removesuffix(".tar.zst")
    target_path = target_dir_path / folder_name

    if target_path.exists():
        if _queue:
            _queue.put(("done", archive_path.name, 0))
        return f"already done {archive_path.name}"

    # Compressed file size drives the progress bar.
    total_bytes = archive_path.stat().st_size
    if _queue:
        _queue.put(("start", archive_path.name, total_bytes))

    tmp_path = target_dir_path / (folder_name + ".tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        with archive_path.open("rb") as raw_file:
            progress = _ProgressReader(raw_file, archive_path.name)
            with (
                dctx.stream_reader(progress) as zstd_reader, 
                tarfile.open(fileobj=zstd_reader, mode="r|") as tar,
            ):
                tar.extractall(path=tmp_path)
        tmp_path.rename(target_path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    finally:
        if _queue:
            _queue.put(("done", archive_path.name, 0))

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
    finished_early: set[str] = set()  # "done" arrived before "start"

    while True:
        msg = queue.get()
        if msg is None:
            break

        kind, name, value = msg

        if kind == "start":
            if name in finished_early:
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
                finished_early.add(name)
            overall.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompress .tar.zst archives produced by the companion compress script."
    )
    parser.add_argument("folder", help="Directory containing .tar.zst archives")
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Parallel decompression workers (default: -1, i.e., all cpu cores)",
    )
    args = parser.parse_args()

    n_workers = os.cpu_count() if args.workers == -1 else args.workers

    source_path = Path(args.folder)
    if not source_path.is_dir():
        raise FileNotFoundError(f"{args.folder} not found")

    target_dir_path = source_path.parent / (
        source_path.stem.replace("_COMPRESSED", "") + "_DECOMPRESSED"
    )
    target_dir_path.mkdir(parents=True, exist_ok=True)

    archive_paths = sorted(source_path.glob("*.tar.zst"))
    if not archive_paths:
        print("No .tar.zst archives found — nothing to do.")
        raise SystemExit(0)

    print(
        f"Decompressing {len(archive_paths)} archive(s) → {target_dir_path}  "
        f"[{n_workers} workers]"
    )

    queue = Queue()

    overall = tqdm(
        total=len(archive_paths),
        desc="Overall",
        unit="archive",
        position=0,
        dynamic_ncols=True,
    )
    listener = threading.Thread(
        target=_listen, args=(queue, n_workers, overall), daemon=True
    )
    listener.start()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(queue,),
    ) as executor:
        futures = {
            executor.submit(decompress_archive, p, target_dir_path): p
            for p in archive_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                overall.write(f"ERROR — {futures[future].name}: {exc}")

    queue.put(None)
    listener.join()
    overall.close()
    print("All done.")
