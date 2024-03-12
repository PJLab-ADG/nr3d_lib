/** @file   dense_grid.cuh
 *  @author Nianchen Deng, Shanghai AI Lab
 *  @brief  Dense grid class.
 */
#pragma once
#include "common.cuh"

class DenseGrid {
public:
    DenseGrid(glm::ivec3 res, bool *grid_occ) : _res(res), _grid_occ(grid_occ) {}

    /**
     * @brief Get the flattened index of voxel.
     *
     * @param voxel grid coordinate of a voxel
     * @return voxel index or -1 if the coordinate is out of the grid boundary [0, _res)
     */
    HOST_DEVICE int get_voxel_idx(glm::ivec3 voxel) const {
        if (voxel.x < 0 || voxel.x >= _res.x || //
            voxel.y < 0 || voxel.y >= _res.y || //
            voxel.z < 0 || voxel.z >= _res.z)
            return -1;
        return voxel.x * _res.y * _res.z + voxel.y * _res.z + voxel.z;
    }

    /**
     * @brief Advance to next voxel from `pos` along the direction `dir`.
     *
     * @param pos position in the grid space (i.e. normalized between [0, _res))
     * @param voxel grid coordinate of current voxel
     * @param dir direction to advance, converted to the grid space
     * @param inv_dir reciprocal of direction for faster calculation
     * @param dir_sign sign of direction for faster calculation
     * @param o_voxel (output) grid coordinate of next voxel
     * @param o_dt (output, optional) distance advanced
     * @return position after advanced
     */
    HOST_DEVICE glm::vec3 advance_to_next_voxel(glm::vec3 pos, glm::ivec3 voxel, glm::vec3 dir,
                                                glm::vec3 inv_dir, glm::ivec3 dir_sign,
                                                glm::ivec3 *__restrict__ o_voxel,
                                                float *__restrict__ o_dt = nullptr) const {
        // Equivalent to: next_grid[ax] = dir_sign[ax] > 0 ? voxel[ax] + 1 : voxel[ax]
        glm::ivec3 next_grid = voxel + dir_sign;
        glm::vec3 txyz = (glm::vec3(next_grid) - pos) * inv_dir;

        // Get axis of min value in txyz
        int axis = txyz.x < txyz.y ? 0 : 1;
        axis = txyz[axis] < txyz.z ? axis : 2;
        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;

        // Get the grid coordinate of next voxel
        glm::ivec3 next_voxel = voxel;
        next_voxel[axis] += dir_sign[axis] * 2 - 1;

        glm::vec3 new_pos;
        new_pos[axis] = next_grid[axis];
        new_pos[axis1] = pos[axis1] + txyz[axis] * dir[axis1];
        new_pos[axis2] = pos[axis2] + txyz[axis] * dir[axis2];

        // Output
        if (o_dt)
            *o_dt = txyz[axis];
        *o_voxel = next_voxel;

        return new_pos;
    }

    /**
     * @brief Advance to next occupied voxel from `pos` along the direction `dir`.
     *
     * @param pos position in the grid space (i.e. normalized between [0, _res))
     * @param dir direction to advance, converted to the grid space
     * @param o_voxel_idx (output) index of next occupied voxel, -1 if advancing out of the grid
     *                    boundary
     * @param o_dt (output, optional) distance advanced
     * @param advance_to_empty_instead (optional) if true, advance to next empty voxel instead of
     *                                 occupied voxel. Default: false
     * @return position after advanced
     */
    HOST_DEVICE glm::vec3 advance_to_next_occ_voxel(glm::vec3 pos, glm::vec3 dir,
                                                    int *__restrict__ o_voxel_idx,
                                                    float *__restrict__ o_dt = nullptr,
                                                    bool advance_to_empty_instead = false) const {
        int voxel_idx;
        float dt;
        float tot_dt = 0.0f;

        glm::vec3 inv_dir = 1.0f / (dir + 1e-10f);
        glm::ivec3 dir_sign =
            1 - glm::ivec3(std::signbit(dir.x), std::signbit(dir.y), std::signbit(dir.z));
        glm::ivec3 voxel = pos;

        do {
            pos = advance_to_next_voxel(pos, voxel, dir, inv_dir, dir_sign, &voxel, &dt);
            voxel_idx = get_voxel_idx(voxel);
            tot_dt += dt;
        } while (voxel_idx >= 0 && _grid_occ[voxel_idx] == advance_to_empty_instead);
        *o_voxel_idx = voxel_idx;
        if (o_dt)
            *o_dt = tot_dt;
        return pos;
    }

    /**
     * @brief Ray march through the dense grid.
     *
     * @param origin origin of the ray in the grid space (i.e. normalized between [0, _res))
     * @param dir direction of the ray, converted to the grid space
     * @param near the nearest distance to start ray marching
     * @param far the farthest distance to stop ray marching
     * @param o_segs (output) segmentations of the ray, each segment is a pair of (t_enter, t_exit)
     * @param callback_fn called when a new segmentation is created, should receive a glm::vec2
     *                    parameter, which is the new segmentation
     * @return number of segments
     */
    template <bool ENABLE_DEBUG, typename CallbackFn>
    HOST_DEVICE int32_t ray_march(glm::vec3 origin, glm::vec3 dir, float near, float far,
                                  glm::vec2 *__restrict__ o_segs, CallbackFn callback_fn,
                                  int32_t debug_maxsegs = 0, int *o_debug_flag = nullptr,
                                  glm::vec3 *o_debug_poses = nullptr,
                                  glm::ivec3 *o_debug_voxels = nullptr) const {
        int32_t n_segs = 0;
        int vidx;
        float t_enter, t_exit;
        int debug_i = 0;

        glm::vec3 inv_dir = 1.0f / (dir + 1e-10f);
        glm::ivec3 dir_sign =
            1 - glm::ivec3(std::signbit(dir.x), std::signbit(dir.y), std::signbit(dir.z));
        glm::vec3 abs_dir = glm::abs(dir);
        int dir_max_axis = abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z   ? 0
                           : abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z ? 1
                                                                            : 2;
        float inv_k = inv_dir[dir_max_axis];
        glm::vec3 pos = origin + dir * near;
        glm::ivec3 voxel = pos;
        vidx = get_voxel_idx(voxel);

        if (ENABLE_DEBUG)
            *o_debug_flag = 0;

        // Advance to the first occupied voxel
        if (vidx < 0 || !_grid_occ[vidx]) {
            do {
                if (ENABLE_DEBUG) {
                    if (debug_i >= debug_maxsegs) {
                        *o_debug_flag = 1;
                        return n_segs;
                    }
                    o_debug_poses[debug_i] = pos;
                    o_debug_voxels[debug_i] = voxel;
                    ++debug_i;
                }
                pos = advance_to_next_voxel(pos, voxel, dir, inv_dir, dir_sign, &voxel);
                vidx = get_voxel_idx(voxel);
            } while (vidx >= 0 && !_grid_occ[vidx]);
        }
        t_enter = (pos[dir_max_axis] - origin[dir_max_axis]) * inv_k;

        while (vidx >= 0 && t_enter < far) {
            do {
                if (ENABLE_DEBUG) {
                    if (debug_i >= debug_maxsegs) {
                        *o_debug_flag = 1;
                        return n_segs;
                    }
                    o_debug_poses[debug_i] = pos;
                    o_debug_voxels[debug_i] = voxel;
                    ++debug_i;
                }
                pos = advance_to_next_voxel(pos, voxel, dir, inv_dir, dir_sign, &voxel);
                vidx = get_voxel_idx(voxel);
            } while (vidx >= 0 && _grid_occ[vidx]);
            t_exit = (pos[dir_max_axis] - origin[dir_max_axis]) * inv_k;
            if (o_segs) {
                *o_segs = glm::vec2(t_enter, std::min(t_exit, far));
                callback_fn(*o_segs);
                ++o_segs;
            }
            ++n_segs;
            if (vidx < 0 || t_exit >= far) // out of bound
                break;
            do {
                if (ENABLE_DEBUG) {
                    if (debug_i >= debug_maxsegs) {
                        *o_debug_flag = 1;
                        return n_segs;
                    }
                    o_debug_poses[debug_i] = pos;
                    o_debug_voxels[debug_i] = voxel;
                    ++debug_i;
                }
                pos = advance_to_next_voxel(pos, voxel, dir, inv_dir, dir_sign, &voxel);
                vidx = get_voxel_idx(voxel);
            } while (vidx >= 0 && !_grid_occ[vidx]);
            t_enter = (pos[dir_max_axis] - origin[dir_max_axis]) * inv_k;
        }
        return n_segs;
    }

    HOST_DEVICE glm::ivec3 res() const { return _res; }
    HOST_DEVICE bool *grid_occ() const { return _grid_occ; }
    void set_grid_occ(bool *grid_occ) { _grid_occ = grid_occ; }

private:
    glm::ivec3 _res;
    bool *_grid_occ;
};