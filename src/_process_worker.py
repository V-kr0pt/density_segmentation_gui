"""
Subprocess worker for background file processing.

Runs in a separate Python process so pydicom decompression and the per-slice
numpy threshold loop can't hold the GIL against Streamlit's UI thread.
With N workers the parent launches N of these concurrently; each worker owns
its own GIL so they don't contend with the main process either.

CLI: python _process_worker.py <info_json_path> <result_json_path>
  info_json:   {file, file_name, input_folder, final_thresholds}
  result_json: {success, file_name, file, error?}
"""
import os
import sys
import json
import traceback


def _write_result(path: str, payload: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    info_path, result_path = sys.argv[1], sys.argv[2]

    # Ensure src/ is importable in the spawned interpreter (Windows doesn't
    # inherit sys.path the way POSIX fork-exec does).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        with open(info_path) as f:
            file_info = json.load(f)
    except Exception as e:
        _write_result(result_path, {
            "success": False,
            "file_name": None,
            "file": None,
            "error": f"failed to read worker info: {e}",
        })
        return 1

    try:
        # Deferred import: pulls in streamlit (~1s cold) but that's fine
        # amortized over a multi-minute processing job, and keeps the logic
        # in exactly one place so parent and worker stay in sync.
        from batch_process_step import BatchProcessingManager
        manager = BatchProcessingManager(max_workers=1)
        result = manager.process_single_file(file_info)
    except Exception as e:
        _write_result(result_path, {
            "success": False,
            "file_name": file_info.get("file_name"),
            "file": file_info.get("file"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        })
        return 1

    out = {
        "success":   bool(result.get("success", False)),
        "file_name": result.get("file_name"),
        "file":      result.get("file"),
    }
    if not out["success"]:
        out["error"] = result.get("error", "unknown error")
    _write_result(result_path, out)
    return 0 if out["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
