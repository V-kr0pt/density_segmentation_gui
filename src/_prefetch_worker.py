"""
Subprocess worker for background slice prefetching.

Runs in a separate Python process so pydicom decompression (which can hold
the GIL on multi-frame DBT reads) cannot block Streamlit's UI thread.
The result is pickled to a temp file; the parent process reads it back.

This module deliberately avoids importing streamlit so that spawning the
worker on Windows doesn't pay the cost of re-running the streamlit setup.
"""
import os
import pickle
import sys


def run(file_path: str, result_path: str) -> None:
    """Load a slice and pickle (slice, affine, shape, index, img_type) to result_path."""
    try:
        # Make src/ importable in the spawned process (Windows doesn't
        # inherit sys.path entries reliably).
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        from ImageLoader import UnifiedImageLoader

        result = UnifiedImageLoader.load_slice(file_path)
        with open(result_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        # Write a sentinel so the parent knows the job failed
        try:
            with open(result_path + ".err", "w") as f:
                f.write(f"{type(e).__name__}: {e}")
        except Exception:
            pass


if __name__ == "__main__":
    # CLI: python _prefetch_worker.py <file_path> <result_path>
    if len(sys.argv) != 3:
        sys.exit(2)
    run(sys.argv[1], sys.argv[2])
