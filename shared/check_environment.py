from __future__ import annotations

import sys


def main() -> None:
    print(f"python_executable={sys.executable}")
    print(f"python_version={sys.version}")

    try:
        import torch
    except ModuleNotFoundError:
        print("torch=NOT_INSTALLED")
        raise

    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
