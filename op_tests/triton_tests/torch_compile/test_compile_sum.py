# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from . import _get_compiled


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
@pytest.mark.parametrize("shape", [(64, 128), (32, 16, 64), (4, 8, 16, 32)])
def test_compile_sum(shape, dim, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    device = "cuda"
    x = torch.randn(shape, device=device, dtype=dtype)

    if dim is None:
        out_eager = torch.sum(x)

        def fn(x):
            return torch.sum(x)

    else:
        out_eager = torch.sum(x, dim=dim)

        def fn(x):
            return torch.sum(x, dim=dim)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 128), (32, 16, 64)])
def test_compile_sum_keepdim(shape, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    device = "cuda"
    x = torch.randn(shape, device=device, dtype=dtype)

    out_eager = torch.sum(x, dim=-1, keepdim=True)

    def fn(x):
        return torch.sum(x, dim=-1, keepdim=True)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
