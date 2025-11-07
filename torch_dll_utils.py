from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, List

__all__ = ["ensure_torch_dlls"]


_BOOTSTRAPPED = False
_DLL_HANDLES: List[object] = []
_PRELOADED: List[object] = []


def _iter_torch_dirs(spec_origin: str) -> Iterable[Path]:
    """Yield directories that contain the torch native libraries."""

    torch_root = Path(spec_origin).resolve().parent
    lib_dir = torch_root / "lib"
    if lib_dir.is_dir():
        yield lib_dir
    yield torch_root


def ensure_torch_dlls() -> None:
    """Expose the torch DLL directory to the Windows loader.

    On Windows the search path can be reset by third-party code (Qt, malware
    scanners, vendor audio drivers, ...).  Registering the directory with
    ``AddDllDirectory`` *and* preloading the relevant DLLs ensures that later
    changes do not affect ``import torch``.
    """

    global _BOOTSTRAPPED

    if _BOOTSTRAPPED or sys.platform != "win32":
        return

    spec = importlib.util.find_spec("torch")
    if not spec or not spec.origin:
        return

    torch_dirs = list(_iter_torch_dirs(spec.origin))
    if not torch_dirs:
        return

    current_path_entries = os.environ.get("PATH", "").split(os.pathsep)

    for folder in torch_dirs:
        # 1) Register via AddDllDirectory if available (Py >= 3.8).
        if hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(str(folder))
            except (FileNotFoundError, OSError):
                handle = None
            if handle:
                _DLL_HANDLES.append(handle)

        # 2) Prepend to PATH for APIs still honouring the environment.
        if str(folder) not in current_path_entries:
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            current_path_entries.insert(0, str(folder))

    # 3) Preload the core DLLs so later search-path changes cannot break them.
    try:
        import ctypes
    except ImportError:
        ctypes = None

    if ctypes is not None:
        for folder in torch_dirs:
            for dll_name in ("c10.dll", "torch_cpu.dll", "torch.dll"):
                dll_path = folder / dll_name
                if dll_path.exists():
                    try:
                        _PRELOADED.append(ctypes.WinDLL(str(dll_path)))
                    except OSError:
                        # leave the actual import error to surface later with
                        # a more descriptive message if another dependency is
                        # missing.
                        pass

    _BOOTSTRAPPED = True
