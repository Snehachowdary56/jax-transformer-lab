import os
import sys
from pathlib import Path

# 8 CPU devices so the data-parallel tests run anywhere; must be set before JAX starts.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
