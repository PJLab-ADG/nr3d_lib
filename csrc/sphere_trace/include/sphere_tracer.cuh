#pragma once
#include "common.cuh"

enum RayStatus : uint8_t { ALIVE, HIT, OUT };

struct TracePayload {
    int32_t idx;
    int32_t seg_idx;
    int32_t seg_end_idx;
    uint16_t n_steps;
    RayStatus status;
    uint8_t unused;
};

struct HitPayload {
  int32_t idx;
  float t;
  int32_t n_steps;
};

class TraceBuffer {
  public:
    float *trace_depths;
    glm::vec4 *trace_hit_regions; // t0, t1, d0, d1
    TracePayload *trace_payloads;

    TraceBuffer(uint32_t n_rays) {
        cudaMalloc(&trace_depths, n_rays * sizeof(float));
        cudaMalloc(&trace_hit_regions, n_rays * sizeof(glm::vec4));
        cudaMalloc(&trace_payloads, n_rays * sizeof(TracePayload));
    }
    ~TraceBuffer() {
        cudaFree(trace_depths);
        cudaFree(trace_hit_regions);
        cudaFree(trace_payloads);
    }
};

using distance_fun_t = std::function<at::Tensor(at::Tensor)>;

class SphereTracer {
  public:
    SphereTracer(bool debug = false);
    ~SphereTracer();

    void init_rays(at::Tensor rays_o, at::Tensor rays_d, at::Tensor valid_rays_idx,
                   at::Tensor segs_pack_info, at::Tensor segs);

    uint32_t compact_rays();

    void advance_rays(at::Tensor distances, float zero_offset, float distance_scale,
                      float min_step);

    std::map<std::string, at::Tensor> get_rays(RayStatus status) const;

    at::Tensor get_trace_positions() const;

    void trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor near, at::Tensor far,
               const distance_fun_t &distance_function, float zero_offset, float distance_scale,
               float min_step, uint32_t max_steps_between_compact, uint32_t max_march_iters,
               at::Tensor valid_rays_idx, at::Tensor segs_pack_info, at::Tensor segs);

    uint32_t n_rays(RayStatus status) const {
        return status == OUT     ? _n_total_rays - n_rays(ALIVE) - n_rays(HIT)
               : status == ALIVE ? _n_rays_alive
                                 : _get_count(status);
    }

  private:
    bool _debug;
    TraceBuffer *_rays_payload[2];
    HitPayload *_rays_payload_hit;
    uint32_t *_counters;
    uint32_t _n_total_rays;
    uint32_t _n_rays_alive;
    uint32_t _buffer_index;
    at::Tensor _rays_o;
    at::Tensor _rays_d;
    at::Tensor _segs;

    uint32_t _get_count(RayStatus status) const;
    void _malloc_payload_buffers(uint32_t n_rays);
    void _free_payload_buffers();
};