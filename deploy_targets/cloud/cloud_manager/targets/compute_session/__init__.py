"""
Backend.AI Compute Session integration for TANGO Cloud Deploy.

This package provides deployment target implementation for Backend.AI compute sessions,
supporting heterogeneous accelerators (GPU, NPU, TPU, CPU).
"""

from cloud_manager.targets.compute_session.compute_session import ComputeSession

__all__ = ["ComputeSession"]
