# utils.py
# Small helper functions shared by all scripts:
#   - fixing the SSL certificate problem on macOS python.org installs
#   - setting random seeds and determinism flags
#   - choosing the device (CUDA, Apple MPS, or CPU)
#   - recording library versions and hardware
#   - loading YAML configs, saving JSON, simple logging and timing

import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import yaml


def fix_ssl_certificates():
    """Point Python to the `certifi` certificate bundle.

    On macOS, Python from python.org sometimes cannot verify HTTPS
    certificates, so downloading the dataset or the pre-trained weights fails
    with CERTIFICATE_VERIFY_FAILED. Using certifi's bundle fixes this.
    This only affects the current Python process.
    """
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


# Run the fix as soon as any script imports utils.
fix_ssl_certificates()

import torch  # noqa: E402  (imported after the SSL fix on purpose)


def set_seed(seed, deterministic=True):
    """Set random seeds for Python, NumPy and PyTorch (CPU + CUDA + MPS).

    If `deterministic` is True we also ask PyTorch to use deterministic
    algorithms where they exist. `warn_only=True` means PyTorch prints a
    warning (instead of crashing) when an operation has no deterministic
    version, for example some kernels on Apple MPS.

    Returns a dictionary describing the settings, so it can be saved.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed by cuBLAS for deterministic matrix multiplies on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded_cpu_cuda_mps": True,
        "torch.use_deterministic_algorithms": deterministic,
        "deterministic_warn_only": deterministic,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
    }


def seed_worker(worker_id):
    """Give every DataLoader worker its own fixed seed.

    PyTorch already sets a per-worker torch seed derived from the
    DataLoader's generator; here we copy it to NumPy and Python `random`
    so random augmentations are reproducible too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Pick the best available device: CUDA GPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cpu_name():
    """Return a readable CPU name for the current machine."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def get_environment_info():
    """Collect versions of the main libraries and a hardware description."""
    import sklearn
    import torchvision

    device = get_device()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        gpu_name = "Apple MPS (integrated GPU of " + get_cpu_name() + ")"
    else:
        gpu_name = "none"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": get_cpu_name(),
        "gpu": gpu_name,
        "device_used": device.type,
    }


def load_config(path):
    """Read a YAML config file and return it as a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    """Save a Python object as nicely formatted JSON (creates folders)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class Logger:
    """Print messages to the screen and also append them to a log file."""

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        # Start a fresh log file each run.
        with open(log_path, "w") as f:
            f.write("")

    def log(self, message=""):
        print(message, flush=True)
        with open(self.log_path, "a") as f:
            f.write(str(message) + "\n")


class Timer:
    """Very small stopwatch: `t = Timer(); ...; t.elapsed()` gives seconds."""

    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start


def sync_device(device):
    """Wait for the GPU to finish its work, so timings are correct."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
