/** @file   pack_ops.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Pack ops Pytorch bindings (declaration).
 */

#pragma once

#include <stdint.h>
#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor> interleave_arange(at::Tensor stop, bool return_idx);
std::tuple<at::Tensor, at::Tensor> interleave_linstep(at::Tensor start, at::Tensor num_steps, at::Tensor step_size, bool return_idx);
std::tuple<at::Tensor, at::Tensor> interleave_linstep(at::Tensor start, at::Tensor num_steps, int32_t step_size, bool return_idx);
std::tuple<at::Tensor, at::Tensor> interleave_linstep(at::Tensor start, at::Tensor num_steps, double step_size, bool return_idx);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_clamp_deprecated(at::Tensor near, at::Tensor far, int32_t max_steps, double dt_gamma, double min_step_size, double max_step_size);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_clamped(at::Tensor near, at::Tensor far, int32_t max_steps, double dt_gamma, double min_step_size, double max_step_size);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_in_packed_segments(at::Tensor near, at::Tensor far, at::Tensor entry, at::Tensor exit, at::Tensor seg_pack_infos, int32_t max_steps, double dt_gamma, double min_step_size, double max_step_size);

/* Pack reduction */
at::Tensor packed_sum(at::Tensor feats, at::Tensor pack_infos);
at::Tensor packed_diff(at::Tensor feats, at::Tensor pack_infos, at::optional<at::Tensor> pack_appends_, at::optional<at::Tensor> pack_last_fill_);
at::Tensor packed_backward_diff(at::Tensor feats, at::Tensor pack_infos, at::optional<at::Tensor> pack_prepends_, at::optional<at::Tensor> pack_first_fill_);
at::Tensor packed_cumprod(at::Tensor feats, at::Tensor pack_infos, bool exclusive, bool reverse);
at::Tensor packed_cumsum(at::Tensor feats, at::Tensor pack_infos, bool exclusive, bool reverse);

/* Per pack arithmetic operation */
enum class PackBinaryOpType {
    Add, 
    Subtract, 
    Multiply,
    Division, 
    Matmul, 
    Gt, // >
    Geq, // >=
    Lt, // <
    Leq, // <=
    Eq, // ==
    Neq, // !=
};
at::Tensor packed_add(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_sub(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_mul(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_div(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_matmul(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_gt(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_geq(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_lt(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_leq(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_eq(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_neq(at::Tensor feats, at::Tensor other, at::Tensor pack_infos);
at::Tensor packed_binary_ops(at::Tensor feats, at::Tensor other, at::Tensor pack_infos, PackBinaryOpType op);

at::Tensor packed_sort_qsort(at::Tensor vals, at::Tensor pack_infos, bool return_idx);
at::Tensor packed_sort_thrust(at::Tensor vals, at::Tensor pack_infos, bool return_idx);
at::Tensor packed_searchsorted(at::Tensor bins, at::Tensor vals, at::Tensor pack_infos);
at::Tensor packed_searchsorted_packed_vals(at::Tensor bins, at::Tensor pack_infos, at::Tensor vals, at::Tensor val_pack_infos);
std::tuple<at::Tensor, at::Tensor, at::Tensor> try_merge_two_packs_sorted_aligned(at::Tensor vals_a, at::Tensor pack_infos_a, at::Tensor vals_b, at::Tensor pack_infos_b, bool b_sorted);
std::tuple<at::Tensor, at::Tensor> packed_invert_cdf(at::Tensor bins, at::Tensor cdfs, at::Tensor u, at::Tensor pack_infos);

std::tuple<at::Tensor, at::Tensor, at::Tensor> packed_alpha_to_vw_forward(at::Tensor alphas, at::Tensor pack_infos, float early_stop_eps, float alpha_thre, bool compression);
at::Tensor packed_alpha_to_vw_backward(at::Tensor weights, at::Tensor grad_weights, at::Tensor alphas, at::Tensor pack_infos, float early_stop_eps, float alpha_thre);

at::Tensor mark_pack_boundaries_cuda(at::Tensor pack_ids);
std::tuple<at::Tensor, at::Tensor> octree_mark_consecutive_segments(at::Tensor pidx, at::Tensor pack_infos, at::Tensor point_hierarchies);
