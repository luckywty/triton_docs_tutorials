"""Check imports and run a Triton vector addition on the GPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and GPU access.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # An uneven length also exercises masking in the last block.
    n = 98432
    x = torch.rand(n, device="cuda")
    y = torch.rand_like(x)
    output = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, output, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y)
    print("PASS: all imports and the Triton GPU vector-add kernel.")


if __name__ == "__main__":
    main()
