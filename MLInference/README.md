# Operator Shapes for Benchmarking
The datasets are at: https://github.com/HicrestLaboratory/PyTorch_Trace_Parser/blob/main/datasets/

The only columns that matter: **`Operation`** and **`Input Dims`**.

---

## Which rows to pick

Filter on these two operations:

- `aten::cudnn_convolution` — 2D convolution
- `aten::mm` — matrix multiply

Discard everything else.

---

## Reading `Input Dims`

The field is a list-of-lists. Each sub-list is the shape of one positional argument to the op. Trailing `[]` entries are scalar config args (stride, padding, etc.) — ignore them.

### `aten::mm`

```
[[201, 768], [768, 768]]
```

Two 2D tensors: `A (M×K)` and `B (K×N)`. Directly maps to `cblas_sgemm`.

### `aten::cudnn_convolution`

```
[[1, C_in, H, W], [C_out, C_in/groups, kH, kW], [], ...] depthwise convolution
```

First tensor is the input `[N, C_in, H, W]`, second is the weight `[C_out, C_in/groups, kH, kW]`. Everything after is config, ignore.

---

## Unique shapes

### ConvNeXt-B — `aten::cudnn_convolution` (depthwise conv) only

| Input `[N, C_in, H, W]` | Weight `[C_out, C_in/groups, kH, kW]` |
|--------------------------|---------------------------------------|
| `[1, 3, 224, 224]`       | `[128, 3, 4, 4]`                      |
| `[1, 128, 56, 56]`       | `[128, 1, 7, 7]`                      |
| `[1, 128, 56, 56]`       | `[256, 128, 2, 2]`                    |
| `[1, 256, 28, 28]`       | `[256, 1, 7, 7]`                      |
| `[1, 256, 28, 28]`       | `[512, 256, 2, 2]`                    |
| `[1, 512, 14, 14]`       | `[512, 1, 7, 7]`                      |
| `[1, 512, 14, 14]`       | `[1024, 512, 2, 2]`                   |
| `[1, 1024, 7, 7]`        | `[1024, 1, 7, 7]`                     |

### DINOv3 ViT-B — `aten::mm` + one `aten::cudnn_convolution`

| Op | A | B |
|----|---|---|
| `aten::cudnn_convolution` | `[1, 3, 224, 224]` | `[768, 3, 16, 16]` |
| `aten::mm` | `[201, 768]` | `[768, 768]` |

The `aten::mm` shape repeats identically 12 times (once per transformer block).