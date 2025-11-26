"""
Accelerator abstraction layer for TANGO Cloud Deploy.

This module provides a unified interface for specifying various hardware accelerators
including GPUs, NPUs, TPUs, and CPUs from different vendors.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AcceleratorType(str, Enum):
    """Types of hardware accelerators supported."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"


class AcceleratorVendor(str, Enum):
    """Hardware accelerator vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    REBELLION = "rebellion"
    GOOGLE = "google"
    INTEL = "intel"


class AcceleratorSpec(BaseModel):
    """
    Specification for hardware accelerator requirements.

    This abstraction allows TANGO to work with various accelerator types
    across different deployment targets.
    """

    type: AcceleratorType = Field(
        default=AcceleratorType.CPU, description="Type of accelerator"
    )
    vendor: Optional[AcceleratorVendor] = Field(
        default=None, description="Vendor of the accelerator hardware"
    )
    model: Optional[str] = Field(
        default=None, description="Specific model name (e.g., 'v100', 'a100', 'atom')"
    )
    count: int = Field(default=0, description="Number of accelerator units required")
    shares: Optional[float] = Field(
        default=None, description="Fractional allocation (e.g., 0.5 for half GPU)"
    )

    def to_backend_ai_resource(self) -> Dict[str, Any]:
        """
        Convert accelerator spec to Backend.AI resource specification.

        Returns:
            Dictionary of resource specifications for Backend.AI API
        """
        resources = {}

        if (
            self.type == AcceleratorType.NPU
            and self.vendor == AcceleratorVendor.REBELLION
        ):
            # Rebellion Atom NPU
            model_name = self.model or "device"
            resources[f"atom.{model_name}"] = str(self.count)

        elif self.type == AcceleratorType.GPU:
            # GPU (NVIDIA/AMD)
            if self.shares is not None:
                resources["cuda.shares"] = str(self.shares)
            elif self.count > 0:
                resources["cuda.shares"] = str(float(self.count))

        elif self.type == AcceleratorType.TPU:
            # TPU support (future extension)
            if self.count > 0:
                resources["tpu.device"] = str(self.count)

        return resources

    def to_gcp_accelerator(self) -> Optional[Dict[str, Any]]:
        """
        Convert to GCP Cloud Run accelerator specification.

        Returns:
            Dictionary for GCP accelerator config or None if not applicable
        """
        if self.type != AcceleratorType.GPU:
            return None

        # GCP supports NVIDIA GPUs
        accelerator_type = "nvidia-tesla-t4"  # Default

        if self.model:
            model_lower = self.model.lower()
            if "a100" in model_lower:
                accelerator_type = "nvidia-a100-80gb"
            elif "v100" in model_lower:
                accelerator_type = "nvidia-tesla-v100"
            elif "t4" in model_lower:
                accelerator_type = "nvidia-tesla-t4"

        return {
            "accelerator_type": accelerator_type,
            "accelerator_count": self.count or 1,
        }

    def to_aws_accelerator(self) -> Optional[str]:
        """
        Convert to AWS ECS instance type that supports the accelerator.

        Returns:
            AWS instance type string or None
        """
        if self.type != AcceleratorType.GPU:
            return None

        # Map to AWS GPU instance types
        if self.model:
            model_lower = self.model.lower()
            if "a100" in model_lower:
                return "p4d.24xlarge"
            elif "v100" in model_lower:
                return "p3.2xlarge"
            elif "t4" in model_lower:
                return "g4dn.xlarge"

        return "g4dn.xlarge"  # Default GPU instance

    @classmethod
    def from_string(cls, spec_str: str) -> "AcceleratorSpec":
        """
        Create AcceleratorSpec from simple string specification.

        Args:
            spec_str: String like "gpu", "gpu:nvidia", "npu:rebellion:atom:1"

        Returns:
            AcceleratorSpec instance
        """
        parts = spec_str.split(":")

        if len(parts) == 1:
            return cls(type=AcceleratorType(parts[0]))

        spec_dict = {"type": parts[0]}

        if len(parts) >= 2:
            try:
                spec_dict["vendor"] = AcceleratorVendor(parts[1])
            except ValueError:
                pass

        if len(parts) >= 3:
            spec_dict["model"] = parts[2]

        if len(parts) >= 4:
            try:
                spec_dict["count"] = int(parts[3])
            except ValueError:
                pass

        return cls(**spec_dict)

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [self.type.value]
        if self.vendor:
            parts.append(self.vendor.value)
        if self.model:
            parts.append(self.model)
        if self.count > 0:
            parts.append(f"x{self.count}")
        elif self.shares:
            parts.append(f"{self.shares}shares")
        return ":".join(parts)
