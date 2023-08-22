#include <sphere_tracer.cuh>
#include <ATen/cuda/CUDAContext.h>

__global__ void init_rays_kernel(uint32_t n_elements, const glm::vec3 *__restrict__ rays_o,
                                 const glm::vec3 *__restrict__ rays_d,
                                 const int64_t *__restrict__ rays_idx,
                                 const float *__restrict__ near, const float *__restrict__ far,
                                 RaysPayload *__restrict__ rays_payload, DenseGrid grid,
                                 bool output_debug = false, glm::vec3 *o_debug_voxelf = nullptr,
                                 glm::ivec3 *o_debug_next_voxel = nullptr,
                                 int *o_debug_steps = nullptr) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    glm::vec3 pos = rays_o[i] + rays_d[i] * near[i];
    glm::vec3 dir = rays_d[i];
    glm::vec3 inv_dir = 1.0f / (dir + 1e-10f);

    // Advance to the first occupied voxel
    int vidx = grid.get_voxel_idx(pos);
    float dt = 0.f;
    if (vidx < 0 || !grid.grid_occ()[vidx]) {
        if (output_debug)
            pos = grid.advance_to_next_occ_voxel(pos, dir, inv_dir, &vidx, &dt, o_debug_steps + i,
                                                 o_debug_voxelf + i, o_debug_next_voxel + i);
        else
            pos = grid.advance_to_next_occ_voxel(pos, dir, inv_dir, &vidx, &dt);
    }

    // Initialize the RaysPayload structure
    rays_payload[i].pos = pos;
    rays_payload[i].dir = dir;
    rays_payload[i].t = near[i] + dt;
    rays_payload[i].max_t = far[i];
    rays_payload[i].last_t = -1.0f;
    rays_payload[i].idx = rays_idx[i];
    rays_payload[i].voxel_idx = vidx;
    rays_payload[i].n_steps = 0;
    rays_payload[i].status = vidx >= 0 && rays_payload[i].t <= rays_payload[i].max_t ? ALIVE : OUT;
}

__global__ void compact_rays_kernel(const uint32_t n_elements,
                                    const RaysPayload *__restrict__ rays_payload,
                                    RaysPayload *__restrict__ o_rays_payload,
                                    RaysPayload *__restrict__ o_rays_payload_hit,
                                    uint32_t *counter) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    RayStatus status = rays_payload[i].status;
    if (status == OUT)
        return;
    uint32_t idx = atomicAdd(counter + status, 1);
    (status == HIT ? o_rays_payload_hit : o_rays_payload)[idx] = rays_payload[i];
}

__global__ void advance_rays_kernel(const uint32_t n_elements, const float *__restrict__ distances,
                                    RaysPayload *__restrict__ payloads, const float zero_offset,
                                    float distance_scale, float hit_threshold, bool hit_at_neg,
                                    DenseGrid grid) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    RaysPayload &payload = payloads[i];

    // As compact may not be performed every time after advance,
    // we need check the payload's status here
    if (payload.status != ALIVE)
        return;

    // Advance by the predicted distance
    float distance = (distances[i] - zero_offset) * distance_scale;
    glm::vec3 pos = payload.pos + distance * payload.dir;
    int voxel_idx = grid.get_voxel_idx(pos);

    // Skip over empty voxels
    if (voxel_idx >= 0 && !grid.grid_occ()[voxel_idx]) {
        float advanced_distance;
        pos = grid.advance_to_next_occ_voxel(pos, payload.dir, 1.f / (payload.dir + 1e-10f),
                                             &voxel_idx, &advanced_distance);
        distance += advanced_distance;
    }

    float t = payload.t + distance;
    if (t - payload.last_t < 1e-6f) {
        float dt = (payload.last_t - payload.t) / 2.0f;
        payload.pos = pos + payload.dir * dt;
        payload.last_t = payload.t;
        payload.t += dt;
        payload.voxel_idx = grid.get_voxel_idx(payload.pos);
        payload.n_steps++;
        payload.status = HIT;
    } else {
        // Update payload
        payload.pos = pos;
        payload.last_t = payload.t;
        payload.t = t;
        payload.voxel_idx = voxel_idx;
        payload.n_steps++;
        payload.status = voxel_idx < 0 || payload.t > payload.max_t ? OUT
                         : hit_at_neg && distance <= hit_threshold  ? HIT
                         : abs(distance) <= hit_threshold           ? HIT
                                                                    : ALIVE;
    }
}

__global__ void unpack_rays_kernel(const uint32_t n_elements,
                                   const RaysPayload *__restrict__ rays_payload, //
                                   glm::vec3 *__restrict__ o_rays_pos,           //
                                   glm::vec3 *__restrict__ o_rays_dir,           //
                                   int64_t *__restrict__ o_rays_idx,             //
                                   int64_t *__restrict__ o_rays_voxel_idx,       //
                                   float *__restrict__ o_rays_t,                 //
                                   int32_t *__restrict__ o_n_steps) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    o_rays_pos[i] = rays_payload[i].pos;
    o_rays_dir[i] = rays_payload[i].dir;
    o_rays_idx[i] = rays_payload[i].idx;
    o_rays_voxel_idx[i] = rays_payload[i].voxel_idx;
    o_rays_t[i] = rays_payload[i].t;
    o_n_steps[i] = rays_payload[i].n_steps;
}

SphereTracer::SphereTracer(bool debug) : _debug(debug) {
    _rays_payload[0] = _rays_payload[1] = _rays_payload_hit = nullptr;
    cudaMalloc(&_counters, sizeof(uint32_t) * 3);
}

SphereTracer::~SphereTracer() {
    _free_payload_buffers();
    cudaFree(_counters);
    _counters = nullptr;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SphereTracer::init_rays(at::Tensor rays_o, at::Tensor rays_d, at::Tensor rays_idx, at::Tensor near,
                        at::Tensor far, const DenseGrid &grid) {
    _n_total_rays = _n_rays_alive = rays_o.size(0);
    _buffer_index = 0;
    cudaMemset(_counters, 0, 3 * sizeof(uint32_t));
    _free_payload_buffers();
    _malloc_payload_buffers(_n_total_rays);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (_debug) {
        at::Tensor debug_voxelf =
            at::empty({_n_rays_alive, 3}, at::dtype(at::kFloat).device(at::kCUDA));
        at::Tensor debug_next_grid =
            at::empty({_n_rays_alive, 3}, at::dtype(at::kInt).device(at::kCUDA));
        at::Tensor debug_steps = at::empty({_n_rays_alive}, at::dtype(at::kInt).device(at::kCUDA));
        linear_kernel(init_rays_kernel, 0, stream, rays_o.size(0), (glm::vec3 *)rays_o.data_ptr(),
                      (glm::vec3 *)rays_d.data_ptr(), rays_idx.data_ptr<long>(),
                      near.data_ptr<float>(), far.data_ptr<float>(),
                      _rays_payload[_buffer_index], grid, true,
                      (glm::vec3 *)debug_voxelf.data_ptr(),
                      (glm::ivec3 *)debug_next_grid.data_ptr(), debug_steps.data_ptr<int>());
        return std::make_tuple(debug_voxelf, debug_next_grid, debug_steps);
    } else {
        linear_kernel(init_rays_kernel, 0, stream, rays_o.size(0), (glm::vec3 *)rays_o.data_ptr(),
                      (glm::vec3 *)rays_d.data_ptr(), rays_idx.data_ptr<long>(),
                      near.data_ptr<float>(), far.data_ptr<float>(),
                      _rays_payload[_buffer_index], grid, false, nullptr, nullptr,
                      nullptr);
        return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
    }
}

uint32_t SphereTracer::compact_rays() {
    cudaMemset(&_counters[ALIVE], 0, sizeof(uint32_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(compact_rays_kernel, 0, stream, _n_rays_alive, _rays_payload[_buffer_index],
                  _rays_payload[1 - _buffer_index], _rays_payload_hit, _counters);
    _buffer_index = 1 - _buffer_index;
    return _n_rays_alive = _get_count(ALIVE);
}

void SphereTracer::advance_rays(at::Tensor distances, float zero_offset, float distance_scale,
                                float hit_threshold, bool hit_at_neg, const DenseGrid &grid) {
    if (_n_rays_alive == 0)
        return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_kernel(advance_rays_kernel, 0, stream, _n_rays_alive, //
                  distances.data_ptr<float>(),                   //
                  _rays_payload[_buffer_index],      //
                  zero_offset, distance_scale, hit_threshold, hit_at_neg, grid);
}

std::map<std::string, at::Tensor> SphereTracer::get_rays(RayStatus status) const {
    if (status == OUT)
        throw std::invalid_argument("Cannot get rays of status OUT");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_rays = this->n_rays(status);
    at::Tensor rays_pos = at::empty({n_rays, 3}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_dir = at::empty({n_rays, 3}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_idx = at::empty({n_rays}, at::dtype(at::kLong).device(at::kCUDA));
    at::Tensor rays_voxel_idx = at::empty({n_rays}, at::dtype(at::kLong).device(at::kCUDA));
    at::Tensor rays_t = at::empty({n_rays}, at::dtype(at::kFloat).device(at::kCUDA));
    at::Tensor rays_n_steps = at::empty({n_rays}, at::dtype(at::kInt).device(at::kCUDA));
    linear_kernel(unpack_rays_kernel, 0, stream, n_rays,
                  status == HIT ? _rays_payload_hit : _rays_payload[_buffer_index],
                  (glm::vec3 *)rays_pos.data_ptr(), (glm::vec3 *)rays_dir.data_ptr(),
                  rays_idx.data_ptr<int64_t>(), rays_voxel_idx.data_ptr<int64_t>(),
                  rays_t.data_ptr<float>(), rays_n_steps.data_ptr<int32_t>());
    return {{"pos", rays_pos}, {"dir", rays_dir},
            {"idx", rays_idx}, {"voxel_idx", rays_voxel_idx},
            {"t", rays_t},     {"n_steps", rays_n_steps}};
}

at::Tensor SphereTracer::get_trace_positions() const {
    return at::from_blob((void *)_rays_payload[_buffer_index], {_n_rays_alive, 3},
                         {sizeof(RaysPayload) / sizeof(float), 1},
                         at::dtype(at::kFloat).device(at::kCUDA));
}

void SphereTracer::trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor rays_idx, at::Tensor near,
                         at::Tensor far, const distance_fun_t &distance_function, float zero_offset,
                         float distance_scale, float hit_threshold, bool hit_at_neg,
                         uint32_t max_steps_between_compact, uint32_t max_march_iters,
                         const DenseGrid &grid) {
    init_rays(rays_o, rays_d, rays_idx, near, far, grid);
    compact_rays();
    for (uint32_t i = 1; i < max_march_iters && _n_rays_alive > 0;) {
        // Compact more frequently in the first couple of steps
        uint32_t compact_step_size = std::min(i, max_steps_between_compact);
        for (uint32_t j = 0; j < compact_step_size; ++j, ++i) {
            at::Tensor distances = distance_function(get_trace_positions());
            advance_rays(distances, zero_offset, distance_scale, hit_threshold, hit_at_neg, grid);
        }
        compact_rays();
    }
}

uint32_t SphereTracer::_get_count(RayStatus status) const {
    uint32_t count;
    cudaMemcpy(&count, &_counters[status], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return count;
}

void SphereTracer::_malloc_payload_buffers(uint32_t n_rays) {
    cudaMalloc(&_rays_payload[0], n_rays * sizeof(RaysPayload));
    cudaMalloc(&_rays_payload[1], n_rays * sizeof(RaysPayload));
    cudaMalloc(&_rays_payload_hit, n_rays * sizeof(RaysPayload));
}

void SphereTracer::_free_payload_buffers() {
    cudaFree(_rays_payload[0]);
    cudaFree(_rays_payload[1]);
    cudaFree(_rays_payload_hit);
    _rays_payload[0] = _rays_payload[1] = _rays_payload_hit = nullptr;
}