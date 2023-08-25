#include <sphere_tracer.cuh>
#include <ATen/cuda/CUDAContext.h>

using namespace at::indexing;

__global__ void init_rays_kernel(uint32_t n_elements, const int64_t *__restrict__ valid_rays_idx,
                                 const glm::ivec2 *__restrict__ segs_pack_info,
                                 const glm::vec2 *__restrict__ segs,
                                 TracePayload *__restrict__ trace_payloads,
                                 float *__restrict__ trace_depths,
                                 glm::vec4 *__restrict__ trace_hit_regions) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    int32_t ray_idx = valid_rays_idx[i];
    int32_t seg_start_idx = segs_pack_info[i].x;
    int32_t n_segs = segs_pack_info[i].y;
    float t = segs[seg_start_idx].x;

    // Initialize the TracePayload structure
    trace_payloads[i] = {
        ray_idx,                // idx
        seg_start_idx,          // seg_idx
        seg_start_idx + n_segs, // seg_end_idx
        0,                      // n_steps
        ALIVE,                  // status
        0                       // unused
    };

    trace_depths[i] = t;
    trace_hit_regions[i] = {-1.0f, 10000.0f, -1.0f, -1.0f};
}

__global__ void compact_rays_kernel(const uint32_t n_elements, uint32_t *counter,
                                    // Input buffer
                                    const TracePayload *__restrict__ trace_payloads,
                                    const float *__restrict__ trace_depths,
                                    const glm::vec4 *__restrict__ trace_hit_regions,
                                    // Output compact buffer
                                    TracePayload *__restrict__ o_trace_payloads,
                                    float *__restrict__ o_trace_depths,
                                    glm::vec4 *__restrict__ o_trace_hit_regions,
                                    // Output hit buffer
                                    HitPayload *__restrict__ o_hit_payloads) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    RayStatus status = trace_payloads[i].status;
    if (status == OUT)
        return;
    uint32_t idx = atomicAdd(counter + status, 1);
    if (status == HIT) {
        o_hit_payloads[idx] = {
            trace_payloads[i].idx,    // idx
            trace_depths[i],          // t
            trace_payloads[i].n_steps // n_steps
        };
    } else {
        o_trace_payloads[idx] = trace_payloads[i];
        o_trace_depths[idx] = trace_depths[i];
        o_trace_hit_regions[idx] = trace_hit_regions[i];
    }
}

__global__ void advance_rays_kernel(const uint32_t n_elements, const float *__restrict__ distances,
                                    TracePayload *__restrict__ trace_payloads,
                                    float *__restrict__ trace_depths,
                                    glm::vec4 *__restrict__ trace_hit_regions,
                                    const float zero_offset, float distance_scale, float min_step,
                                    const glm::vec2 *__restrict__ segs) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    // As compact may not be performed every time after advance,
    // we need check the payload's status here
    if (trace_payloads[i].status != ALIVE)
        return;

    TracePayload trace_payload = trace_payloads[i];
    float t = trace_depths[i];
    glm::vec4 hit_region_info = trace_hit_regions[i];
    float distance = distances[i] * distance_scale - zero_offset;

    float adv_t;
    RayStatus adv_status = ALIVE;

    int32_t adv_seg_idx = trace_payload.seg_idx;
    glm::vec2 seg = segs[adv_seg_idx];
    int adv = -1; // -1: no advance, 0: retreat, 1: advance

    // Update hit region
    if (distance > 0.0f) {
        hit_region_info.x = t;
        hit_region_info.z = distance;
    } else if (hit_region_info.x >= 0.0f) {
        hit_region_info.y = t;
        hit_region_info.w = distance;
    }

    if (abs(distance) <= 5.0e-3f) {
        adv_t = t + distance;
        adv_status = HIT;
    } else if (hit_region_info.y - hit_region_info.x <= 1.1f * min_step) {
        float k = hit_region_info.z / (hit_region_info.z - hit_region_info.w);
        adv_t = hit_region_info.x + k * (hit_region_info.y - hit_region_info.x);
        adv_status = HIT;
    } else {
        adv = !signbit(distance) || hit_region_info.x < 0.0f;
        distance = abs(distance);
    }

    if (adv > 0) {
        adv_t = t + max(min(distance, (hit_region_info.y - hit_region_info.x) * 0.8f), min_step);
        while (adv_t > seg.y && adv_seg_idx < trace_payload.seg_end_idx)
            seg = segs[++adv_seg_idx];
        if (adv_t <= seg.y)
            adv_t = max(adv_t, seg.x);
        else {
            adv_t = seg.y;
            adv_status = OUT;
        }
    } else if (adv == 0) {
        adv_t = t - min((hit_region_info.y - hit_region_info.x) / 2.0f, distance);
        while (adv_t < seg.x)
            seg = segs[--adv_seg_idx];
        adv_t = min(adv_t, seg.y);
    }

    // Update payload
    trace_payloads[i] = {
        trace_payload.idx,         // idx
        adv_seg_idx,               // seg_idx
        trace_payload.seg_end_idx, // seg_end_idx
        trace_payload.n_steps++,   // n_steps
        adv_status,                // status
        (uint8_t)adv               // unused
    };
    trace_depths[i] = adv_t;
    trace_hit_regions[i] = hit_region_info;
}

__global__ void get_positions_kernel(const uint32_t n_elements, //
                                     const glm::vec3 *__restrict__ rays_o,
                                     const glm::vec3 *__restrict__ rays_d,
                                     const TracePayload *__restrict__ trace_payloads,
                                     const float *__restrict__ trace_depths,
                                     glm::vec3 *__restrict__ o_positions) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    int64_t ray_idx = trace_payloads[i].idx;
    float t = trace_depths[i];
    glm::vec3 ray_o = rays_o[ray_idx];
    glm::vec3 ray_d = rays_d[ray_idx];
    o_positions[i] = ray_o + ray_d * t;
}

__global__ void get_rays_kernel(const uint32_t n_elements, //
                                const glm::vec3 *__restrict__ rays_o,
                                const glm::vec3 *__restrict__ rays_d,
                                const TracePayload *__restrict__ trace_payloads,
                                const float *__restrict__ trace_depths,
                                glm::vec3 *__restrict__ o_rays_pos,          //
                                glm::vec3 *__restrict__ o_rays_dir,          //
                                int64_t *__restrict__ o_rays_idx,            //
                                float *__restrict__ o_rays_t,                //
                                int32_t *__restrict__ o_n_steps,             //
                                RayStatus *__restrict__ o_debug_rays_status, //
                                uint8_t *__restrict__ o_debug_rays_flag) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    int64_t ray_idx = trace_payloads[i].idx;
    float t = trace_depths[i];
    glm::vec3 ray_o = rays_o[ray_idx];
    glm::vec3 ray_d = rays_d[ray_idx];
    o_rays_pos[i] = ray_o + ray_d * t;
    o_rays_idx[i] = ray_idx;
    o_rays_dir[i] = ray_d;
    o_rays_t[i] = t;
    o_n_steps[i] = trace_payloads[i].n_steps;
    o_debug_rays_status[i] = trace_payloads[i].status;
    o_debug_rays_flag[i] = trace_payloads[i].unused;
}

__global__ void get_hit_rays_kernel(const uint32_t n_elements, //
                                    const glm::vec3 *__restrict__ rays_o,
                                    const glm::vec3 *__restrict__ rays_d,
                                    const HitPayload *__restrict__ hit_payloads,
                                    glm::vec3 *__restrict__ o_rays_pos, //
                                    glm::vec3 *__restrict__ o_rays_dir, //
                                    int64_t *__restrict__ o_rays_idx,   //
                                    float *__restrict__ o_rays_t,       //
                                    int32_t *__restrict__ o_n_steps) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    int64_t ray_idx = hit_payloads[i].idx;
    float t = hit_payloads[i].t;
    glm::vec3 ray_o = rays_o[ray_idx];
    glm::vec3 ray_d = rays_d[ray_idx];
    o_rays_pos[i] = ray_o + ray_d * t;
    o_rays_idx[i] = ray_idx;
    o_rays_dir[i] = ray_d;
    o_rays_t[i] = t;
    o_n_steps[i] = hit_payloads[i].n_steps;
}

SphereTracer::SphereTracer(bool debug) : _debug(debug) {
    _rays_payload[0] = _rays_payload[1] = nullptr;
    _rays_payload_hit = nullptr;
    cudaMalloc(&_counters, sizeof(uint32_t) * 2);
}

SphereTracer::~SphereTracer() {
    _free_payload_buffers();
    cudaFree(_counters);
    _counters = nullptr;
}

void SphereTracer::init_rays(at::Tensor rays_o, at::Tensor rays_d, at::Tensor valid_rays_idx,
                             at::Tensor segs_pack_info, at::Tensor segs) {
    _rays_o = rays_o;
    _rays_d = rays_d;
    _segs = segs;
    _n_rays_alive = _n_total_rays = valid_rays_idx.size(0);

    _buffer_index = 0;
    cudaMemset(_counters, 0, 2 * sizeof(uint32_t));
    _free_payload_buffers();

    if (!_n_rays_alive)
        return;
    
    _malloc_payload_buffers(_n_rays_alive);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(init_rays_kernel, 0, stream, _n_rays_alive, (int64_t *)valid_rays_idx.data_ptr(),
                  (glm::ivec2 *)segs_pack_info.data_ptr(), (glm::vec2 *)segs.data_ptr(),
                  _rays_payload[_buffer_index]->trace_payloads,
                  _rays_payload[_buffer_index]->trace_depths,
                  _rays_payload[_buffer_index]->trace_hit_regions);
}

uint32_t SphereTracer::compact_rays() {
    cudaMemset(&_counters[ALIVE], 0, sizeof(uint32_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(compact_rays_kernel, 0, stream, _n_rays_alive, _counters,
                  _rays_payload[_buffer_index]->trace_payloads,
                  _rays_payload[_buffer_index]->trace_depths,
                  _rays_payload[_buffer_index]->trace_hit_regions, //
                  _rays_payload[1 - _buffer_index]->trace_payloads,
                  _rays_payload[1 - _buffer_index]->trace_depths,
                  _rays_payload[1 - _buffer_index]->trace_hit_regions, //
                  _rays_payload_hit);
    _buffer_index = 1 - _buffer_index;
    return _n_rays_alive = _get_count(ALIVE);
}

void SphereTracer::advance_rays(at::Tensor distances, float zero_offset, float distance_scale,
                                float min_step) {
    if (_n_rays_alive == 0)
        return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(advance_rays_kernel, 0, stream, _n_rays_alive,   //
                  distances.data_ptr<float>(),                     //
                  _rays_payload[_buffer_index]->trace_payloads,    //
                  _rays_payload[_buffer_index]->trace_depths,      //
                  _rays_payload[_buffer_index]->trace_hit_regions, //
                  zero_offset, distance_scale, min_step, (glm::vec2 *)_segs.data_ptr());
}

std::map<std::string, at::Tensor> SphereTracer::get_rays(RayStatus status) const {
    if (status == OUT)
        throw std::invalid_argument("Cannot get rays of status OUT");
    uint32_t n_rays = this->n_rays(status);
    at::Tensor rays_pos = at::empty({n_rays, 3}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_dir = at::empty({n_rays, 3}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_idx = at::empty({n_rays}, at::dtype(at::kLong).device(at::kCUDA));
    at::Tensor rays_t = at::empty({n_rays}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_n_steps = at::empty({n_rays}, at::dtype(at::kInt).device(at::kCUDA));
    at::Tensor debug_rays_status = at::empty({n_rays}, at::dtype(at::kByte).device(at::kCUDA));
    at::Tensor debug_rays_flag = at::empty({n_rays}, at::dtype(at::kByte).device(at::kCUDA));
    if (n_rays > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if (status == HIT)
            linear_kernel(get_hit_rays_kernel, 0, stream, n_rays, (glm::vec3 *)_rays_o.data_ptr(),
                          (glm::vec3 *)_rays_d.data_ptr(), _rays_payload_hit,
                          (glm::vec3 *)rays_pos.data_ptr(), (glm::vec3 *)rays_dir.data_ptr(),
                          rays_idx.data_ptr<int64_t>(), rays_t.data_ptr<float>(),
                          rays_n_steps.data_ptr<int32_t>());
        else
            linear_kernel(
                get_rays_kernel, 0, stream, n_rays, (glm::vec3 *)_rays_o.data_ptr(),
                (glm::vec3 *)_rays_d.data_ptr(), _rays_payload[_buffer_index]->trace_payloads,
                _rays_payload[_buffer_index]->trace_depths, (glm::vec3 *)rays_pos.data_ptr(),
                (glm::vec3 *)rays_dir.data_ptr(), rays_idx.data_ptr<int64_t>(),
                rays_t.data_ptr<float>(), rays_n_steps.data_ptr<int32_t>(),
                (RayStatus *)debug_rays_status.data_ptr(), debug_rays_flag.data_ptr<uint8_t>());
    }
    return {{"n_rays", at::scalar_tensor((int)n_rays)},
            {"pos", rays_pos},
            {"dir", rays_dir},
            {"idx", rays_idx},
            {"t", rays_t},
            {"n_steps", rays_n_steps},
            {"debug_status", debug_rays_status},
            {"debug_flag", debug_rays_flag}};
}

at::Tensor SphereTracer::get_trace_positions() const {
    at::Tensor positions = at::empty({_n_rays_alive, 3}, at::dtype(at::kFloat).device(at::kCUDA));
    if (_n_rays_alive > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        linear_kernel(
            get_positions_kernel, 0, stream, _n_rays_alive, (glm::vec3 *)_rays_o.data_ptr(),
            (glm::vec3 *)_rays_d.data_ptr(), _rays_payload[_buffer_index]->trace_payloads,
            _rays_payload[_buffer_index]->trace_depths, (glm::vec3 *)positions.data_ptr());
    }
    return positions;
}

void SphereTracer::trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor near, at::Tensor far,
                         const distance_fun_t &distance_function, float zero_offset,
                         float distance_scale, float min_step, uint32_t max_steps_between_compact,
                         uint32_t max_march_iters, at::Tensor valid_rays_idx,
                         at::Tensor segs_pack_info, at::Tensor segs) {
    init_rays(rays_o, rays_d, valid_rays_idx, segs_pack_info, segs);
    for (uint32_t i = 1; i < max_march_iters && _n_rays_alive > 0;) {
        uint32_t compact_step_size = std::min(i, max_steps_between_compact);
        for (uint32_t j = 0; j < compact_step_size; ++j, ++i) {
            at::Tensor distances = distance_function(get_trace_positions());
            advance_rays(distances, zero_offset, distance_scale, min_step);
        }
        compact_rays();
    }
}

uint32_t SphereTracer::_get_count(RayStatus status) const {
    if (status == OUT)
        throw std::invalid_argument("Cannot query counter of status OUT");
    uint32_t count;
    cudaMemcpy(&count, &_counters[status], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return count;
}

void SphereTracer::_malloc_payload_buffers(uint32_t n_rays) {
    _rays_payload[0] = new TraceBuffer(n_rays);
    _rays_payload[1] = new TraceBuffer(n_rays);
    cudaMalloc(&_rays_payload_hit, sizeof(HitPayload) * n_rays);
}

void SphereTracer::_free_payload_buffers() {
    delete _rays_payload[0];
    delete _rays_payload[1];
    _rays_payload[0] = _rays_payload[1] = nullptr;
    cudaFree(_rays_payload_hit);
    _rays_payload_hit = nullptr;
}