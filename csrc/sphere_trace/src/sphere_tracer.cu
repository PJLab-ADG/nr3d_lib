/** @file   sphere_tracer.cu
 *  @author Nianchen Deng, Shanghai AI Lab
 *  @brief  The implementation of sphere trace class and algorithm.
 */
#include <sphere_trace/sphere_tracer.cuh>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>

using namespace at::indexing;

__device__ RayStatus advance_single_step(const glm::vec2 *__restrict__ segs,    //
                                         float min_step, float d, bool forward, //
                                         int32_t &seg_idx, int32_t &seg_end_idx,
                                         glm::vec4 &hit_region_info, float &t, glm::vec2 &seg) {
    if (forward) {
        t += max(min(d, (hit_region_info.y - hit_region_info.x) * 0.8f), min_step);
        while (t > seg.y && seg_idx < seg_end_idx)
            seg = segs[++seg_idx];
        if (t <= seg.y) {
            t = max(t, seg.x);
            return ALIVE;
        } else {
            // Trace beyond the end of segments, no hit point found
            t = seg.y;
            return OUT;
        }
    } else {
        t -= min((hit_region_info.y - hit_region_info.x) / 2.0f, max(d, min_step));
        while (t < seg.x)
            seg = segs[--seg_idx];
        t = min(t, seg.y);
        return ALIVE;
    }
}

__device__ RayStatus advance_ray(const glm::vec2 *__restrict__ segs,                    //
                                 const glm::vec2 *__restrict__ segs_endpoint_distances, //
                                 float zero_offset, float distance_scale, float min_step,
                                 float hit_threshold, float d, int32_t &seg_idx,
                                 int32_t &seg_end_idx, glm::vec4 &hit_region_info,
                                 glm::ivec2 &hit_seg_region, float &t, glm::vec2 &seg,
                                 uint16_t &n_steps, int8_t &debug_flag) {
    while (true) {
        // Update hit region
        // Note that a valid hit region must have start point locates outside,
        // so if the start point is still inside, we should update the start point.
        // (Consider a ray emitted from inside, we should trace to outside first)
        if (hit_region_info.z < 0.0f || d >= 0.0f) {
            hit_region_info.x = t;
            hit_region_info.z = d;
            hit_seg_region.x = seg_idx;
        } else {
            hit_region_info.y = t;
            hit_region_info.w = d;
            hit_seg_region.y = seg_idx + 1;
        }

        // HIT condition:
        // 1. current distance is small enough
        // 2. hit region is valid (outside start and inside end) and small enough
        if (abs(d) <= hit_threshold) {
            t = t + d;
            debug_flag = 127;
            return HIT;
        }
        if (hit_region_info.z >= 0.0f && hit_region_info.w <= 0.0f &&
            hit_region_info.y - hit_region_info.x <= 1.1f * min_step) {
            // Linearly interpolate the hit point
            float k = hit_region_info.z / (hit_region_info.z - hit_region_info.w);
            t = hit_region_info.x + k * (hit_region_info.y - hit_region_info.x);
            debug_flag = 126;
            return HIT;
        }

        // Advance forward or backward according to the sign of d
        // Always advance forward if the start point of hit region still locates inside
        bool forward = !signbit(d) || hit_region_info.z < 0.0f;
        auto status = advance_single_step(segs, min_step, abs(d), forward, seg_idx, seg_end_idx,
                                          hit_region_info, t, seg);
        n_steps++;
        debug_flag = status == OUT ? -127 : forward ? 1 : -1;

        // If we have distance values at endpoints of segments,
        // we can continue advancing when t is near one of the endpoints
        int32_t boundary = t - seg.x < 5.0e-3f ? 0 : seg.y - t < 5.0e-3f ? 1 : -1;
        if (status == OUT || segs_endpoint_distances == nullptr || boundary < 0)
            return status;
        t = seg[boundary];
        d = segs_endpoint_distances[seg_idx][boundary] * distance_scale - zero_offset;
    }
}

__global__ void init_rays_kernel(uint32_t n_elements, const int64_t *__restrict__ valid_rays_idx,
                                 const glm::ivec2 *__restrict__ segs_pack_info,
                                 const glm::vec2 *__restrict__ segs,
                                 TracePayload *__restrict__ trace_payloads,
                                 float *__restrict__ trace_depths,
                                 glm::vec4 *__restrict__ trace_hit_regions,
                                 glm::ivec2 *__restrict__ trace_hit_seg_regions) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    int32_t ray_idx = valid_rays_idx[i];
    int32_t seg_start_idx = segs_pack_info[i].x;
    int32_t seg_end_idx = seg_start_idx + segs_pack_info[i].y;
    float t = segs[seg_start_idx].x;

    // Initialize the TracePayload structure
    trace_payloads[i] = {
        ray_idx,       // idx
        seg_start_idx, // seg_idx
        seg_end_idx,   // seg_end_idx
        0,             // n_steps
        ALIVE,         // status
        0              // debug_flag
    };

    trace_depths[i] = t;
    trace_hit_regions[i] = {-1.0f, segs[seg_end_idx - 1].y, -1.0f, 1.0f};
    trace_hit_seg_regions[i] = {seg_start_idx, seg_end_idx};
}

__global__ void init_rays_with_distance_hint_kernel(
    uint32_t n_elements, const int64_t *__restrict__ valid_rays_idx,
    const glm::ivec2 *__restrict__ segs_pack_info, const glm::vec2 *__restrict__ segs,
    const glm::vec2 *__restrict__ segs_endpoint_distances, float zero_offset, float distance_scale,
    float min_step, float hit_threshold, TracePayload *__restrict__ trace_payloads,
    float *__restrict__ trace_depths, glm::vec4 *__restrict__ trace_hit_regions,
    glm::ivec2 *__restrict__ trace_hit_seg_regions) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    int32_t ray_idx = valid_rays_idx[i];

    // Determine the range in which we need to trace
    // If a segment has positive distance at enter point and negative distance at exit point,
    // it contains at least one hit point. So all segments after it can be safely ignored.
    glm::ivec2 init_si_range = {segs_pack_info[i].x, segs_pack_info[i].x + segs_pack_info[i].y};
    int32_t si;
    glm::vec2 d_enter_exit;
    for (si = init_si_range.x; si < init_si_range.y; ++si) {
        d_enter_exit = segs_endpoint_distances[si] * distance_scale - zero_offset;
        if (d_enter_exit.x >= 0.0f && d_enter_exit.y <= 0.0f)
            break;
    }
    int32_t seg_idx = init_si_range.x;
    int32_t seg_end_idx = min(si + 1, init_si_range.y);

    // Initialize hit region hint
    glm::vec4 hit_region_info = {-1.0f, segs[seg_end_idx - 1].y, -1.0f, d_enter_exit.y};
    glm::ivec2 hit_seg_region = {seg_idx, seg_end_idx};

    // Trace start from the start point of the first segment
    glm::vec2 seg = segs[seg_idx];
    float t = seg.x;
    float d = segs_endpoint_distances[seg_idx].x * distance_scale - zero_offset;
    uint16_t n_steps = 0;
    int8_t debug_flag = 0;
    auto status = advance_ray(segs, segs_endpoint_distances, zero_offset, distance_scale, min_step,
                              hit_threshold, d, seg_idx, seg_end_idx, hit_region_info,
                              hit_seg_region, t, seg, n_steps, debug_flag);

    // Initialize the TracePayload structure
    trace_payloads[i] = {
        ray_idx,     // idx
        seg_idx,     // seg_idx
        seg_end_idx, // seg_end_idx
        n_steps,     // n_steps
        status,      // status
        debug_flag   // unused
    };
    trace_depths[i] = t;
    trace_hit_regions[i] = hit_region_info;
    trace_hit_seg_regions[i] = hit_seg_region;
}

__global__ void compact_rays_kernel(const uint32_t n_elements, uint32_t *counter,
                                    // Input buffer
                                    const TracePayload *__restrict__ trace_payloads,
                                    const float *__restrict__ trace_depths,
                                    const glm::vec4 *__restrict__ trace_hit_regions,
                                    const glm::ivec2 *__restrict__ trace_hit_seg_regions,
                                    // Output compact buffer
                                    TracePayload *__restrict__ o_trace_payloads,
                                    float *__restrict__ o_trace_depths,
                                    glm::vec4 *__restrict__ o_trace_hit_regions,
                                    glm::ivec2 *__restrict__ o_trace_hit_seg_regions,
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
        o_trace_hit_seg_regions[idx] = trace_hit_seg_regions[i];
    }
}

__global__ void advance_rays_kernel(const uint32_t n_elements, const float *__restrict__ distances,
                                    TracePayload *__restrict__ trace_payloads,
                                    float *__restrict__ trace_depths,
                                    glm::vec4 *__restrict__ trace_hit_regions,
                                    glm::ivec2 *__restrict__ trace_hit_seg_regions,
                                    float zero_offset, float distance_scale, float min_step,
                                    float hit_threshold, const glm::vec2 *__restrict__ segs,
                                    const glm::vec2 *__restrict__ segs_endpoint_distances) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    // As compact may not be performed every time after advance,
    // we need check the payload's status here
    if (trace_payloads[i].status != ALIVE)
        return;

    TracePayload trace_payload = trace_payloads[i];
    float t = trace_depths[i];
    float d = distances[i] * distance_scale - zero_offset;
    glm::vec4 hit_region_info = trace_hit_regions[i];
    glm::ivec2 hit_seg_region = trace_hit_seg_regions[i];
    glm::vec2 seg = segs[trace_payload.seg_idx];

    trace_payload.status = advance_ray(segs, segs_endpoint_distances, zero_offset, distance_scale,
                                       min_step, hit_threshold, d, trace_payload.seg_idx,
                                       trace_payload.seg_end_idx, hit_region_info, hit_seg_region,
                                       t, seg, trace_payload.n_steps, trace_payload.debug_flag);

    // Update payload
    trace_payloads[i] = trace_payload;
    trace_depths[i] = t;
    trace_hit_regions[i] = hit_region_info;
    trace_hit_seg_regions[i] = hit_seg_region;
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
                                const glm::vec4 *__restrict__ trace_hit_region_infos,
                                const glm::ivec2 *__restrict__ trace_hit_seg_regions,
                                glm::vec3 *__restrict__ o_rays_pos,              //
                                glm::vec3 *__restrict__ o_rays_dir,              //
                                int64_t *__restrict__ o_rays_idx,                //
                                float *__restrict__ o_rays_t,                    //
                                int32_t *__restrict__ o_n_steps,                 //
                                RayStatus *__restrict__ o_rays_status,           //
                                glm::vec4 *__restrict__ o_rays_hit_region_infos, //
                                glm::ivec2 *__restrict__ o_rays_hit_seg_regions, //
                                int32_t *__restrict__ o_rays_seg_idxs,           //
                                int32_t *__restrict__ o_rays_seg_end_idxs,       //
                                int8_t *__restrict__ o_rays_debug_flag) {
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
    o_rays_status[i] = trace_payloads[i].status;
    o_rays_hit_region_infos[i] = trace_hit_region_infos[i];
    o_rays_hit_seg_regions[i] = trace_hit_seg_regions[i];
    o_rays_seg_idxs[i] = trace_payloads[i].seg_idx;
    o_rays_seg_end_idxs[i] = trace_payloads[i].seg_end_idx;
    o_rays_debug_flag[i] = trace_payloads[i].debug_flag;
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

__global__ void sample_on_segments_phase1_kernel(
    uint32_t n_elements, float step_size, const glm::vec4 *__restrict__ trace_hit_region_infos,
    const glm::ivec2 *__restrict__ trace_hit_seg_regions, const glm::vec2 *__restrict__ segs,
    int32_t *__restrict__ o_num_samples) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    glm::vec2 hit_region = *(glm::vec2 *)(trace_hit_region_infos + i);
    glm::ivec2 hit_seg_region = trace_hit_seg_regions[i];
    int32_t num_samples = 0;
    for (int32_t si = hit_seg_region.x; si < hit_seg_region.y; ++si) {
        glm::vec2 seg = segs[si];
        seg.x = max(seg.x, hit_region.x);
        seg.y = min(seg.y, hit_region.y);
        num_samples += (int32_t)ceil((seg.y - seg.x) / step_size) + 1;
    }
    o_num_samples[i] = num_samples;
}

__global__ void sample_on_segments_phase2_kernel(
    uint32_t n_elements, float step_size, //
    const glm::vec3 *__restrict__ rays_o, const glm::vec3 *__restrict__ rays_d,
    const TracePayload *__restrict__ trace_payloads,
    const glm::vec4 *__restrict__ trace_hit_region_infos,
    const glm::ivec2 *__restrict__ trace_hit_seg_regions, const glm::vec2 *__restrict__ segs,
    const int32_t *__restrict__ samples_offset, //
    float *__restrict__ o_sample_depths, glm::vec3 *__restrict__ o_sample_positions) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    int32_t ray_idx = trace_payloads[i].idx;
    glm::vec2 hit_region = *(glm::vec2 *)(trace_hit_region_infos + i);
    glm::ivec2 hit_seg_region = trace_hit_seg_regions[i];
    o_sample_depths += samples_offset[i];
    o_sample_positions += samples_offset[i];
    for (int32_t si = hit_seg_region.x; si < hit_seg_region.y; ++si) {
        glm::vec2 seg = segs[si];
        seg.x = max(seg.x, hit_region.x);
        seg.y = min(seg.y, hit_region.y);
        float length = seg.y - seg.x;
        int32_t n_samples_in_seg = max(0, (int32_t)ceil(length / step_size) + 1);
        float step_size_in_seg = length / (n_samples_in_seg - 1);
        float sample_t = seg.x;
        for (int32_t j = 0; j < n_samples_in_seg; ++j, sample_t += step_size_in_seg) {
            *(o_sample_depths++) = sample_t;
            *(o_sample_positions++) = rays_o[ray_idx] + rays_d[ray_idx] * sample_t;
        }
    }
}

__global__ void trace_on_samples_kernel(
    uint32_t n_elements, const TracePayload *__restrict__ trace_payloads,
    const int32_t *__restrict__ samples_offset, const int32_t *__restrict__ num_samples,
    const float *__restrict__ sample_depths, const float *__restrict__ sample_distances,
    HitPayload *__restrict__ o_hit_payloads, uint32_t *__restrict__ hit_counter) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    int32_t n_samples = num_samples[i];
    int32_t sample_offset = samples_offset[i];
    int32_t sample_end = sample_offset + n_samples;
    for (int32_t pi = sample_offset; pi < sample_end - 1; ++pi) {
        float d1 = sample_distances[pi];
        float d2 = sample_distances[pi + 1];
        if (d1 >= 0.0f && d2 <= 0.0f) {
            float t1 = sample_depths[pi];
            float t2 = sample_depths[pi + 1];
            float k = d1 / (d1 - d2);
            float t = t1 + k * (t2 - t1);
            int32_t idx = atomicAdd(hit_counter, 1);
            o_hit_payloads[idx] = {trace_payloads[i].idx, t, trace_payloads[i].n_steps + n_samples};
            return;
        }
    }
}

SphereTracer::SphereTracer(float min_step, float distance_scale, float zero_offset,
                           float hit_threshold)
    : _min_step(min_step), _distance_scale(distance_scale), _zero_offset(zero_offset),
      _hit_threshold(hit_threshold) {
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
                             at::Tensor segs_pack_info, at::Tensor segs,
                             at::optional<at::Tensor> segs_endpoint_distances) {
    _rays_o = rays_o;
    _rays_d = rays_d;
    _segs = segs;
    _segs_endpoint_distances = segs_endpoint_distances;
    _n_rays_alive = _n_total_rays = valid_rays_idx.size(0);

    _buffer_index = 0;
    cudaMemset(_counters, 0, 2 * sizeof(uint32_t));
    _free_payload_buffers();

    if (!_n_rays_alive)
        return;

    _malloc_payload_buffers(_n_rays_alive);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (segs_endpoint_distances.has_value()) {
        linear_kernel(init_rays_with_distance_hint_kernel, 0, stream, _n_rays_alive,
                      (int64_t *)valid_rays_idx.data_ptr(),                                  //
                      (glm::ivec2 *)segs_pack_info.data_ptr(), (glm::vec2 *)segs.data_ptr(), //
                      (glm::vec2 *)segs_endpoint_distances->data_ptr(),                      //
                      _zero_offset, _distance_scale, _min_step, _hit_threshold,              //
                      _rays_payload[_buffer_index]->trace_payloads,
                      _rays_payload[_buffer_index]->trace_depths,
                      _rays_payload[_buffer_index]->trace_hit_regions,
                      _rays_payload[_buffer_index]->trace_hit_seg_regions);
    } else {
        linear_kernel(init_rays_kernel, 0, stream, _n_rays_alive,
                      (int64_t *)valid_rays_idx.data_ptr(),                                  //
                      (glm::ivec2 *)segs_pack_info.data_ptr(), (glm::vec2 *)segs.data_ptr(), //
                      _rays_payload[_buffer_index]->trace_payloads,
                      _rays_payload[_buffer_index]->trace_depths,
                      _rays_payload[_buffer_index]->trace_hit_regions,
                      _rays_payload[_buffer_index]->trace_hit_seg_regions);
    }
}

uint32_t SphereTracer::compact_rays() {
    cudaMemset(&_counters[ALIVE], 0, sizeof(uint32_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(compact_rays_kernel, 0, stream, _n_rays_alive, _counters,
                  _rays_payload[_buffer_index]->trace_payloads,
                  _rays_payload[_buffer_index]->trace_depths,
                  _rays_payload[_buffer_index]->trace_hit_regions,
                  _rays_payload[_buffer_index]->trace_hit_seg_regions, //
                  _rays_payload[1 - _buffer_index]->trace_payloads,
                  _rays_payload[1 - _buffer_index]->trace_depths,
                  _rays_payload[1 - _buffer_index]->trace_hit_regions,
                  _rays_payload[1 - _buffer_index]->trace_hit_seg_regions, //
                  _rays_payload_hit);
    _buffer_index = 1 - _buffer_index;
    return _n_rays_alive = _get_count(ALIVE);
}

void SphereTracer::advance_rays(at::Tensor distances) {
    if (_n_rays_alive == 0)
        return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(advance_rays_kernel, 0, stream, _n_rays_alive,            //
                  distances.data_ptr<float>(),                              //
                  _rays_payload[_buffer_index]->trace_payloads,             //
                  _rays_payload[_buffer_index]->trace_depths,               //
                  _rays_payload[_buffer_index]->trace_hit_regions,          //
                  _rays_payload[_buffer_index]->trace_hit_seg_regions,      //
                  _zero_offset, _distance_scale, _min_step, _hit_threshold, //
                  (glm::vec2 *)_segs.data_ptr(),
                  _segs_endpoint_distances.has_value()
                      ? (glm::vec2 *)_segs_endpoint_distances->data_ptr()
                      : nullptr);
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (status == HIT) {
        if (n_rays > 0)
            linear_kernel(get_hit_rays_kernel, 0, stream, n_rays, (glm::vec3 *)_rays_o.data_ptr(),
                          (glm::vec3 *)_rays_d.data_ptr(), _rays_payload_hit,
                          (glm::vec3 *)rays_pos.data_ptr(), (glm::vec3 *)rays_dir.data_ptr(),
                          rays_idx.data_ptr<int64_t>(), rays_t.data_ptr<float>(),
                          rays_n_steps.data_ptr<int32_t>());
        return {{"n_rays", at::scalar_tensor((int)n_rays)},
                {"pos", rays_pos},
                {"dir", rays_dir},
                {"idx", rays_idx},
                {"t", rays_t},
                {"n_steps", rays_n_steps}};
    } else {
        at::Tensor rays_status = at::empty({n_rays}, at::dtype(at::kByte).device(at::kCUDA));
        at::Tensor rays_hit_region_infos =
            at::empty({n_rays, 4}, at::dtype(at::kFloat).device(at::kCUDA));
        at::Tensor rays_hit_seg_regions =
            at::empty({n_rays, 2}, at::dtype(at::kInt).device(at::kCUDA));
        at::Tensor rays_seg_idxs = at::empty({n_rays}, at::dtype(at::kInt).device(at::kCUDA));
        at::Tensor rays_seg_end_idxs = at::empty({n_rays}, at::dtype(at::kInt).device(at::kCUDA));
        at::Tensor rays_debug_flag = at::empty({n_rays}, at::dtype(at::kChar).device(at::kCUDA));
        if (n_rays > 0)
            linear_kernel(
                get_rays_kernel, 0, stream, n_rays, (glm::vec3 *)_rays_o.data_ptr(),
                (glm::vec3 *)_rays_d.data_ptr(), _rays_payload[_buffer_index]->trace_payloads,
                _rays_payload[_buffer_index]->trace_depths,
                _rays_payload[_buffer_index]->trace_hit_regions,
                _rays_payload[_buffer_index]->trace_hit_seg_regions,
                (glm::vec3 *)rays_pos.data_ptr(), (glm::vec3 *)rays_dir.data_ptr(),
                rays_idx.data_ptr<int64_t>(), rays_t.data_ptr<float>(),
                rays_n_steps.data_ptr<int32_t>(), (RayStatus *)rays_status.data_ptr(),
                (glm::vec4 *)rays_hit_region_infos.data_ptr(),
                (glm::ivec2 *)rays_hit_seg_regions.data_ptr(), rays_seg_idxs.data_ptr<int32_t>(),
                rays_seg_end_idxs.data_ptr<int32_t>(), rays_debug_flag.data_ptr<int8_t>());
        return {{"n_rays", at::scalar_tensor((int)n_rays)},
                {"pos", rays_pos},
                {"dir", rays_dir},
                {"idx", rays_idx},
                {"t", rays_t},
                {"n_steps", rays_n_steps},
                {"status", rays_status},
                {"debug_flag", rays_debug_flag},
                {"hit_region_infos", rays_hit_region_infos},
                {"hit_seg_regions", rays_hit_seg_regions},
                {"seg_idxs", rays_seg_idxs},
                {"seg_end_idxs", rays_seg_end_idxs}};
    }
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SphereTracer::sample_on_segments(float step_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Phase 1: compute number of samples for each ray
    at::Tensor rays_n_samples =
        at::empty({_n_rays_alive}, at::dtype(at::kInt).device(_rays_o.device()));
    linear_kernel(sample_on_segments_phase1_kernel, 0, stream, _n_rays_alive, step_size,
                  _rays_payload[_buffer_index]->trace_hit_regions,
                  _rays_payload[_buffer_index]->trace_hit_seg_regions,
                  (glm::vec2 *)_segs.data_ptr(), rays_n_samples.data_ptr<int32_t>());

    // Compute pack info
    at::Tensor rays_n_samples_cumsum = rays_n_samples.cumsum(0, at::kInt);
    at::Tensor rays_samples_offset = rays_n_samples_cumsum - rays_n_samples;

    // Allocate rays samples buffer
    int32_t total_samples = rays_n_samples_cumsum[-1].item<int32_t>();
    at::Tensor rays_sample_depths = _rays_o.new_empty({total_samples});
    at::Tensor rays_sample_positions = _rays_o.new_empty({total_samples, 3});

    // Phase 2: compute ray samples
    linear_kernel(sample_on_segments_phase2_kernel, 0, stream, _n_rays_alive, step_size,
                  (glm::vec3 *)_rays_o.data_ptr(), (glm::vec3 *)_rays_d.data_ptr(),
                  _rays_payload[_buffer_index]->trace_payloads,
                  _rays_payload[_buffer_index]->trace_hit_regions,
                  _rays_payload[_buffer_index]->trace_hit_seg_regions,
                  (glm::vec2 *)_segs.data_ptr(), rays_samples_offset.data_ptr<int32_t>(),
                  rays_sample_depths.data_ptr<float>(),
                  (glm::vec3 *)rays_sample_positions.data_ptr());
    return std::make_tuple(rays_samples_offset, rays_n_samples, rays_sample_depths,
                           rays_sample_positions);
}

void SphereTracer::trace_on_samples(at::Tensor rays_samples_offset, at::Tensor rays_n_samples,
                                    at::Tensor rays_sample_depths,
                                    at::Tensor rays_sample_distances) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(trace_on_samples_kernel, 0, stream, _n_rays_alive,
                  _rays_payload[_buffer_index]->trace_payloads,
                  rays_samples_offset.data_ptr<int32_t>(), rays_n_samples.data_ptr<int32_t>(),
                  rays_sample_depths.data_ptr<float>(), rays_sample_distances.data_ptr<float>(),
                  _rays_payload_hit, &_counters[HIT]);
}

void SphereTracer::trace(at::Tensor rays_o, at::Tensor rays_d,
                         const distance_fun_t &distance_function,
                         uint32_t max_steps_between_compact, uint32_t max_march_iters,
                         at::Tensor valid_rays_idx, at::Tensor segs_pack_info, at::Tensor segs,
                         at::optional<at::Tensor> segs_endpoint_distances) {
    init_rays(rays_o, rays_d, valid_rays_idx, segs_pack_info, segs, segs_endpoint_distances);
    for (uint32_t i = 1; i < max_march_iters && _n_rays_alive > 0;) {
        uint32_t compact_step_size = std::min(i, max_steps_between_compact);
        for (uint32_t j = 0; j < compact_step_size; ++j, ++i) {
            at::Tensor distances = distance_function(get_trace_positions());
            advance_rays(distances);
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