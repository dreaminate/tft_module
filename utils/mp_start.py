"""Utilities for multiprocessing start method configuration."""
from __future__ import annotations

import torch.multiprocessing as mp
from typing import Literal


def ensure_start_method(method: Literal["spawn", "fork", "forkserver"] = "spawn", *, force: bool = False) -> None:
    """Ensure the multiprocessing start method is set.

    On Windows, DataLoader with num_workers>0 requires the ``spawn`` start method and
    that the configuration happens under ``if __name__ == "__main__"`` guard.  Calling
    this helper prevents ``RuntimeError: An attempt has been made to start a new process
    before the current process has finished its bootstrapping phase`` when scripts are
    executed as entry points.

    Parameters
    ----------
    method:
        Desired start method.  Defaults to ``"spawn"`` which is the only option on
        Windows.
    force:
        Forwarded to :func:`multiprocessing.set_start_method`.  Set to ``True`` if we
        must override a previously configured start method.
    """
    try:  # ``torch.multiprocessing`` follows the stdlib API.
        mp.set_start_method(method, force=force)
    except RuntimeError:
        # The context has already been set – nothing to do.
        pass
    except ValueError:
        # Unsupported start method on current platform – ignore silently.
        pass
