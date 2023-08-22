#pragma once
#include "common.cuh"

class DenseGrid {
  public:
    DenseGrid(glm::ivec3 res, bool *grid_occ)
        : _res(res), _voxel_size(2.f / glm::vec3(res)), _grid_occ(grid_occ) {}

    /**
     * @brief Convert the grid coordinate to voxel index.
     *
     * @param voxel grid coordinate of a voxel
     * @return voxel index or -1 if the coordinate is out of the grid boundary [0, _res)
     */
    HOST_DEVICE int grid_coord_to_idx(glm::ivec3 voxel) const {
        if (voxel.x < 0 || voxel.x >= _res.x || //
            voxel.y < 0 || voxel.y >= _res.y || //
            voxel.z < 0 || voxel.z >= _res.z)
            return -1;
        return voxel.x * _res.y * _res.z + voxel.y * _res.z + voxel.z;
    }

    /**
     * @brief Get the voxel index at `pos`.
     *
     * @param pos position normalized between [-1, 1)
     * @return voxel index or -1 if the position is out of the normalized grid boundary [-1, 1)
     */
    HOST_DEVICE int get_voxel_idx(glm::vec3 pos) const {
        return grid_coord_to_idx((pos + 1.f) * glm::vec3(_res) * .5f);
    }

    /**
     * @brief Check whether `pos` is in occupied voxel.
     *
     * @param pos position normalized between [0, 1)
     * @return occupied or not
     */
    HOST_DEVICE bool is_occupied(glm::vec3 pos) const {
        int voxel_idx = get_voxel_idx(pos);
        return voxel_idx >= 0 && _grid_occ[voxel_idx];
    }

    /**
     * @brief Advance to next voxel from `pos` along the direction `dir`.
     *
     * @param pos position normalized between [-1, 1)
     * @param dir direction to advance (also converted to the normalized space)
     * @param inv_dir reciprocal of direction for faster calculation
     * @param o_voxel_idx (output) index of next voxel, -1 if advancing out of the grid boundary
     * @param o_dt (output) distance advanced
     * @return position after advanced
     */
    HOST_DEVICE glm::vec3
    advance_to_next_voxel(glm::vec3 pos, glm::vec3 dir, glm::vec3 inv_dir, glm::ivec3 dir_sign,
                          int *__restrict__ o_voxel_idx, float *__restrict__ o_dt,
                          glm::vec3 *o_debug_voxelf = nullptr,
                          glm::vec3 *o_debug_dir_sign = nullptr, glm::vec3 *o_debug_txyz = nullptr,
                          glm::ivec3 *o_debug_next_voxel = nullptr) const {
        glm::vec3 voxelf = (pos + 1.f) / _voxel_size;
        glm::ivec3 voxel = voxelf;
        // Equivalent to: next_grid[ax] = dir_sign[ax] > 0 ? voxel[ax] + 1 : voxel[ax]
        glm::ivec3 next_grid = voxel + (dir_sign + 1) / 2;
        glm::vec3 txyz = (glm::vec3(next_grid) - voxelf) * _voxel_size * inv_dir;

        // Get axis of min value in txyz
        int axis = txyz.x < txyz.y ? 0 : 1;
        axis = txyz[axis] < txyz.z ? axis : 2;

        // Get the grid coordinate of next voxel
        glm::ivec3 next_voxel = voxel;
        next_voxel[axis] += dir_sign[axis];

        glm::vec3 new_pos;
        new_pos[axis] = next_grid[axis] * _voxel_size[axis] - 1.f + 1e-6f * dir_sign[axis];
        float dt = (new_pos[axis] - pos[axis]) * inv_dir[axis];
        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;
        new_pos[axis1] = pos[axis1] + dt * dir[axis1];
        new_pos[axis2] = pos[axis2] + dt * dir[axis2];

        // Output
        *o_dt = dt;
        *o_voxel_idx = grid_coord_to_idx(next_voxel);

        // Debug
        if (o_debug_voxelf)
            *o_debug_voxelf = voxelf;
        if (o_debug_dir_sign)
            *o_debug_dir_sign = dir_sign;
        if (o_debug_next_voxel)
            *o_debug_next_voxel = next_voxel;
        if (o_debug_txyz)
            *o_debug_txyz = txyz;

        return new_pos;
    }

    /**
     * @brief Advance to next occupied voxel from `pos` along the direction `dir`.
     *
     * @param pos position normalized between [-1, 1)
     * @param dir direction to advance (also converted to the normalized space)
     * @param inv_dir reciprocal of direction for faster calculation
     * @param o_voxel_idx (output) index of next occupied voxel, -1 if advancing out of the grid
     *                    boundary
     * @param o_dt (output) distance advanced
     * @return position after advanced
     */
    HOST_DEVICE glm::vec3
    advance_to_next_occ_voxel(glm::vec3 pos, glm::vec3 dir, glm::vec3 inv_dir,
                              int *__restrict__ o_voxel_idx, float *__restrict__ o_dt,
                              int *o_debug_steps = nullptr, glm::vec3 *o_debug_voxelf = nullptr,
                              glm::ivec3 *o_debug_next_voxel = nullptr) const {
        int voxel_idx;
        float tot_dt = 0.0f;
        int i = 0; // TODO: i is only for debug

        glm::ivec3 dir_sign = glm::ivec3(signbit(dir.x), signbit(dir.y), signbit(dir.z)) * -2 + 1;

        do {
            float dt;
            pos = advance_to_next_voxel(pos, dir, inv_dir, dir_sign, &voxel_idx, &dt,
                                        o_debug_voxelf, nullptr, nullptr, o_debug_next_voxel);
            tot_dt += dt;
            ++i;
        } while (voxel_idx >= 0 && !_grid_occ[voxel_idx] && i < 100000);
        *o_voxel_idx = voxel_idx;
        *o_dt = tot_dt;

        // Debug
        if (o_debug_steps)
            *o_debug_steps = i;

        return pos;
    }

    HOST_DEVICE glm::ivec3 res() const { return _res; }
    HOST_DEVICE bool *grid_occ() const { return _grid_occ; }
    void set_grid_occ(bool *grid_occ) { _grid_occ = grid_occ; }

  private:
    glm::ivec3 _res;
    glm::vec3 _voxel_size;
    bool *_grid_occ;
};