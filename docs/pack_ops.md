# Pack-ops

To produce and deal with non-matrix & packed data, and efficiently perform per-pack or pack-wise operations with Pytorch-CUDA exentions.

## Basic concepts

### Packed tensors and `pack_infos`

`Pytorch` deals with matrix tensors, i.e. batched data with the same data size on each batch. 

However, there are situations where we have to deal with non-matrix data with different data size on each batch (or, as named by [kaolin](https://github.com/NVIDIAGameWorks/kaolin) "nuggets"). This is especially the case for efficient 3D neural rendering in which each batch or "nuggets" represents buffer data of each ray.

Hence, we develop a toolbox for such needs, trying our best to deal with pack production, pack reduction, per-pack operations and so on. 

We acknowledge that the current repository does not cover all required use cases, and our implementations are still far from complete. Therefore, **any collaboration or pull-requests are warmly welcomed !** :hugs:


| batched data                                          | packed data ("nuggets" in [kaolin](https://github.com/NVIDIAGameWorks/kaolin)) |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| <img alt="data_batched" src="../media/data_batched2.png" width="400">             | <img alt="data_packed" src="../media/data_packed2.png" width="400">                    |
| batch infos:<br />batch_size=4<br />batch_data_size=5 | `pack_infos`:<br />`torch.tensor([[0,4],[4,1],[5,5],[10,3]])`<br />which means:<br />`first_inds = pack_infos[:,0] = [0,4,5,10]`<br />`lengths = pack_infos[:,1] = [4,1,5,3]`<br /> |

## Supported ops

### interleave ops

interleave_arange_simple

<img alt="interleave_arange_simple" src="../media/pack_ops/interleave_arange_simple.png" width="600">

interleave_arange

<img alt="interleave_arange" src="../media/pack_ops/interleave_arange.png" width="600">

interleave_linstep

<img alt="interleave_linstep" src="../media/pack_ops/interleave_linstep.png" width="600">

### packed_add/sub/mul/div [differentiable]

<img alt="packed_add" src="../media/pack_ops/packed_add.png" width="600">

### packed_sum / mean [differentiable]

implementations are borrowed from [kaolin](https://github.com/NVIDIAGameWorks/kaolin)

packed_sum

<img alt="packed_sum" src="../media/pack_ops/packed_sum.png" width="600">

packed_mean

<img alt="packed_mean" src="../media/pack_ops/packed_mean.png" width="600">

### packed_cumsum / cumprod [differentiable]

implementations are borrowed from [kaolin](https://github.com/NVIDIAGameWorks/kaolin)

packed_cumsum

<img alt="packed_cumsum" src="../media/pack_ops/packed_cumsum.png" width="600">

packed_cumprod

<img alt="packed_cumprod" src="../media/pack_ops/packed_cumprod.png" width="600">

### packed_diff [differentiable]

`packed_diff`

forward difference:

<img alt="packed_diff" src="../media/pack_ops/packed_diff.png" width="600">

`packed_diff(feats, appends=...)`

forward difference with appends:

<img alt="packed_diff2" src="../media/pack_ops/packed_diff_appends.png" width="600">

`packed_backward_diff(feats, prepends=...)`

backward difference with prepends:

<img alt="packed_backward_diff_prepends" src="../media/pack_ops/packed_backward_diff_prepends.png" width="600">

### packed_sort

<img alt="packed_sort" src="../media/pack_ops/packed_sort.png" width="600">

### packed_searchsorted / search_sorted_packed

packed_searchsorted

<img alt="packed_searchsorted" src="../media/pack_ops/packed_searchsorted.png" width="600">

packed_searchsorted_packed_vals

<img alt="packed_searchsorted_packed" src="../media/pack_ops/packed_searchsorted_packed.png" width="600">

### packed_invert_cdf

<img alt="packed_invert_cdf" src="../media/pack_ops/packed_invert_cdf.png" width="600">

### merging two sorted packs

You achieve multiple packs merging via `packed_sort`. 

However, If you are only merging two packs with one of them already sorted, it's faster to use APIs here since they are based on `searchsorted` which is faster than general sorting.

`merge_two_packs_sorted_aligned`

Merge two packs with aligned nuggets

<img alt="merge_two_packs_sorted_aligned" src="../media/pack_ops/merge_two_packs_sorted_aligned.png" width="600">

`merge_two_packs_sorted_a_includes_b`

Merge two packs with not aligned nuggets, but a's nuggets includes b's nuggets.

<img alt="merge_two_packs_sorted_a_includes_b" src="../media/pack_ops/merge_two_packs_sorted_a_includes_b.png" width="600">

`merge_two_packs_sorted`

Merge two packs with intersecting nuggets. 

NOTE: The unique nuggets of `b` must be sorted in advance since their orders will be untouched.

<img alt="merge_two_packs_sorted" src="../media/pack_ops/merge_two_packs_sorted.png" width="600">

