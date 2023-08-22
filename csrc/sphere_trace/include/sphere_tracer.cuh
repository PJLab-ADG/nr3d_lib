#pragma once
#include "common.cuh"
#include "dense_grid.cuh"

enum RayStatus : uint8_t { ALIVE, HIT, OUT };

struct RaysPayload {
    glm::vec3 pos;
    glm::vec3 dir;
    float max_t;
    float t;
    float last_t;
    int32_t idx;
    int32_t voxel_idx;
    uint16_t n_steps;
    RayStatus status;
    uint8_t unused; // To make structure align with 32bit
};

using distance_fun_t = std::function<at::Tensor(at::Tensor)>;

class SphereTracer {
  public:
    SphereTracer(bool debug = false);
    ~SphereTracer();

    std::tuple<at::Tensor, at::Tensor, at::Tensor> init_rays(at::Tensor rays_o, at::Tensor rays_d,
                                                             at::Tensor rays_idx, at::Tensor near,
                                                             at::Tensor far, const DenseGrid &grid);

    uint32_t compact_rays();

    void advance_rays(at::Tensor distances, float zero_offset, float distance_scale,
                      float hit_threshold, bool hit_at_neg, const DenseGrid &grid);

    std::map<std::string, at::Tensor> get_rays(RayStatus status) const;

    at::Tensor get_trace_positions() const;

    void trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor rays_idx, at::Tensor near,
               at::Tensor far, const distance_fun_t &distance_function, float zero_offset,
               float distance_scale, float hit_threshold, bool hit_at_neg,
               uint32_t max_steps_between_compact, uint32_t max_march_iters, const DenseGrid &grid);

    uint32_t n_rays(RayStatus status) const {
        return status == OUT     ? _n_total_rays - n_rays(ALIVE) - n_rays(HIT)
               : status == ALIVE ? _n_rays_alive
                                 : _get_count(status);
    }

  private:
    bool _debug;
    RaysPayload* _rays_payload[2];
    RaysPayload* _rays_payload_hit;
    uint32_t* _counters;
    uint32_t _n_total_rays;
    uint32_t _n_rays_alive;
    uint32_t _buffer_index;

    uint32_t _get_count(RayStatus status) const;
    void _malloc_payload_buffers(uint32_t n_rays);
    void _free_payload_buffers();
};