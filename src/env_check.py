import os
import sys
import ssl
import tempfile
from importlib import import_module

MIN_PYTHON = (3, 10)

# Minimal versions are optional; keep only what matters for your project.
MIN_VERSIONS = {
    "streamlit": "1.30.0",
    "numpy": "1.23.0",
    "pydicom": "2.3.0",
    "nibabel": "5.0.0",
}

# (import_name, pip_name) pairs
REQUIRED_IMPORTS = [
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "pillow"),
    ("pydicom", "pydicom"),
    ("nibabel", "nibabel"),
]


def _version_tuple(v: str) -> tuple:
    # Very small helper: compares "1.30.0" style versions
    parts = []
    for x in v.split("."):
        try:
            parts.append(int(x))
        except ValueError:
            # e.g. "1.0rc1" -> keep only numeric prefix
            num = "".join(ch for ch in x if ch.isdigit())
            parts.append(int(num) if num else 0)
    return tuple(parts)


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            "\nThis project requires Python >= 3.10.\n"
            f"Detected version: {sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}\n"
        )


def check_required_imports() -> None:
    missing = []
    for import_name, pip_name in REQUIRED_IMPORTS:
        try:
            import_module(import_name)
        except Exception:
            missing.append((import_name, pip_name))

    if missing:
        msg = "\n".join([f"- import '{i}' (pip install {p})" for i, p in missing])
        raise RuntimeError(
            "\nMissing required dependencies.\n"
            "Please install them and try again:\n"
            f"{msg}\n"
        )


def check_min_versions() -> None:
    try:
        from importlib.metadata import version as pkg_version
    except Exception:
        # Very old Python only; not relevant here
        return

    problems = []
    for pkg, min_v in MIN_VERSIONS.items():
        try:
            installed = pkg_version(pkg)
            if _version_tuple(installed) < _version_tuple(min_v):
                problems.append((pkg, installed, min_v))
        except Exception:
            # Package not installed: handled by import checks
            pass

    if problems:
        lines = "\n".join([f"- {p}: installed {i}, required >= {m}" for p, i, m in problems])
        raise RuntimeError(
            "\nSome dependencies are too old.\n"
            "Please upgrade and try again:\n"
            f"{lines}\n"
        )


def check_writable_tempdir() -> None:
    # Quick write test (common failure on locked-down environments)
    try:
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(b"ok")
            f.flush()
    except Exception as e:
        raise RuntimeError(
            "\nCannot write to temporary directory.\n"
            f"Error: {e}\n"
        )


def warn_if_libressl() -> None:
    # urllib3 v2 warns when ssl is built with LibreSSL; not necessarily fatal
    openssl_ver = getattr(ssl, "OPENSSL_VERSION", "")
    if "LibreSSL" in openssl_ver:
        # Do not raise; just print a friendly warning.
        print(
            "\n[Warning] Your Python 'ssl' module is linked against LibreSSL.\n"
            "Some HTTPS-related features may not work with urllib3 v2.\n"
            f"Detected: {openssl_ver}\n"
        )


def run_all_checks() -> None:
    check_python_version()
    warn_if_libressl()
    check_required_imports()
    check_min_versions()
    check_writable_tempdir()
