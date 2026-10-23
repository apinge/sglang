"""Raw file-to-GPU throughput for one safetensors file, to compare with
fastsafetensors docs/amd-perf.md. Usage: fst_raw_throughput.py <file> {fst,mmap} [threads] [bbuf_kb]
"""

import sys
import time

import torch
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from safetensors import safe_open


def main() -> None:
    path, mode = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    start = time.perf_counter()
    nbytes = 0
    if mode == "fst":
        threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        bbuf_kb = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
        loader = SafeTensorsFileLoader(
            SingleGroup(),
            "cuda:0",
            nogds=True,
            max_threads=threads,
            bbuf_size_kb=bbuf_kb,
        )
        loader.add_filenames({0: [path]})
        buf = loader.copy_files_to_device()
        torch.cuda.synchronize()
        for key in buf.key_to_rank_lidx:
            t = buf.get_tensor(key)
            nbytes += t.numel() * t.element_size()
        buf.close()
        loader.close()
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).to("cuda", non_blocking=False)
                nbytes += t.numel() * t.element_size()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"{mode} {sys.argv[3:]} {nbytes / 1e9:.1f} GB in {elapsed:.1f} s = {nbytes / 1e9 / elapsed:.2f} GB/s"
    )


if __name__ == "__main__":
    main()
