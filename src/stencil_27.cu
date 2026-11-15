// 27-point periodic Laplacian on a GPU.
// Each warp owns 32 cells along x. Interior x-neighbors move through registers
// with __shfl_up_sync / __shfl_down_sync. The two lanes at the warp edge, and
// every y/z neighbor, come from the shared-memory tile.
//
// Build on the 4060:
//   nvcc -O3 -arch=sm_89 -o runs/stencil_27 src/stencil_27.cu -lcusparse
//   ./runs/stencil_27
//   ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
//       -o runs/roofline ./runs/stencil_27

#include <cuda_runtime.h>
#include <cusparse.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int kN = 128;
constexpr float kH = 1.f;

__device__ __forceinline__ int wrap(int index, int n) {
    int m = index % n;
    return m < 0 ? m + n : m;
}

__device__ float stencil_from_shared(const float* tile, int pitch_y, int pitch_z, int lx, int ly, int lz, float scale) {
    auto at = [&](int dx, int dy, int dz) -> float {
        return tile[(lz + dz) * pitch_z + (ly + dy) * pitch_y + (lx + dx)];
    };
    float sum = -14.f * at(0, 0, 0);
    const int faces[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const int edges[12][3] = {{1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1},  {1, 0, -1},
                              {-1, 0, 1}, {-1, 0, -1}, {0, 1, 1},  {0, 1, -1},  {0, -1, 1}, {0, -1, -1}};
    const int corners[8][3] = {{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
                               {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
    for (auto offset : faces) sum += at(offset[0], offset[1], offset[2]);
    for (auto offset : edges) sum += 0.5f * at(offset[0], offset[1], offset[2]);
    for (auto offset : corners) sum += 0.25f * at(offset[0], offset[1], offset[2]);
    return sum * scale;
}

// block: (32, 4, 2). The tile keeps a 1-cell halo, so shared memory is 34 x 6 x 4.
__global__ void stencil_tiled(const float* __restrict__ field, float* __restrict__ out, int n, float scale) {
    constexpr int bx = 32, by = 4, bz = 2;
    constexpr int tx = bx + 2, ty = by + 2, tz = bz + 2;
    __shared__ float tile[tz * ty * tx];

    const int origin_x = blockIdx.x * bx;
    const int origin_y = blockIdx.y * by;
    const int origin_z = blockIdx.z * bz;
    const int threads = bx * by * bz;
    const int tid = (threadIdx.z * by + threadIdx.y) * bx + threadIdx.x;

    for (int cell = tid; cell < tx * ty * tz; cell += threads) {
        int lx = cell % tx;
        int ly = (cell / tx) % ty;
        int lz = cell / (tx * ty);
        int gx = wrap(origin_x + lx - 1, n);
        int gy = wrap(origin_y + ly - 1, n);
        int gz = wrap(origin_z + lz - 1, n);
        tile[lz * ty * tx + ly * tx + lx] = field[(gz * n + gy) * n + gx];
    }
    __syncthreads();

    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;
    const int x = origin_x + threadIdx.x;
    const int y = origin_y + threadIdx.y;
    const int z = origin_z + threadIdx.z;
    if (x >= n || y >= n || z >= n) return;

    // x-neighbors for lanes 1..30 come from registers. Lanes 0 and 31 still
    // read the halo, which sits outside this warp.
    const float self = tile[lz * ty * tx + ly * tx + lx];
    const unsigned mask = 0xffffffffu;
    const float from_right = __shfl_down_sync(mask, self, 1);
    const float from_left = __shfl_up_sync(mask, self, 1);
    float xp = (threadIdx.x == 31) ? tile[lz * ty * tx + ly * tx + lx + 1] : from_right;
    float xm = (threadIdx.x == 0) ? tile[lz * ty * tx + ly * tx + lx - 1] : from_left;

    // Rebuild the x faces from the shuffled values, and take every other neighbor from the tile.
    float sum = stencil_from_shared(tile, tx, ty * tx, lx, ly, lz, 1.f);
    // stencil_from_shared already added the shared-memory x faces. Replace them with the register values.
    const float shared_xp = tile[lz * ty * tx + ly * tx + lx + 1];
    const float shared_xm = tile[lz * ty * tx + ly * tx + lx - 1];
    sum += (xp - shared_xp) + (xm - shared_xm);
    out[(z * n + y) * n + x] = sum * scale;
}

__global__ void stencil_naive(const float* __restrict__ field, float* __restrict__ out, int n, float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= n || y >= n || z >= n) return;
    auto at = [&](int dx, int dy, int dz) -> float {
        return field[(wrap(z + dz, n) * n + wrap(y + dy, n)) * n + wrap(x + dx, n)];
    };
    float sum = -14.f * at(0, 0, 0);
    sum += at(1, 0, 0) + at(-1, 0, 0) + at(0, 1, 0) + at(0, -1, 0) + at(0, 0, 1) + at(0, 0, -1);
    const int edges[12][3] = {{1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1},  {1, 0, -1},
                              {-1, 0, 1}, {-1, 0, -1}, {0, 1, 1},  {0, 1, -1},  {0, -1, 1}, {0, -1, -1}};
    const int corners[8][3] = {{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
                               {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
    for (auto offset : edges) sum += 0.5f * at(offset[0], offset[1], offset[2]);
    for (auto offset : corners) sum += 0.25f * at(offset[0], offset[1], offset[2]);
    out[(z * n + y) * n + x] = sum * scale;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

float elapsed_ms(void (*launch)(float*, float*, int, float, int), float* in, float* out, int n, float scale, int repeats) {
    launch(in, out, n, scale, 1);
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) launch(in, out, n, scale, 1);
    cudaEventRecord(stop);
    check(cudaEventSynchronize(stop), "timer");
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}

void launch_tiled(float* in, float* out, int n, float scale, int) {
    dim3 block(32, 4, 2);
    dim3 grid((n + 31) / 32, (n + 3) / 4, (n + 1) / 2);
    stencil_tiled<<<grid, block>>>(in, out, n, scale);
}

void launch_naive(float* in, float* out, int n, float scale, int) {
    dim3 block(8, 8, 4);
    dim3 grid((n + 7) / 8, (n + 7) / 8, (n + 3) / 4);
    stencil_naive<<<grid, block>>>(in, out, n, scale);
}

}  // namespace

int main() {
    const int n = kN;
    const int count = n * n * n;
    const float scale = 1.f / (6.f * kH * kH);
    std::vector<float> host(count), back(count);
    for (int i = 0; i < count; ++i) {
        int x = i % n;
        int y = (i / n) % n;
        int z = i / (n * n);
        host[i] = std::sin(2.0 * M_PI * x / n) * std::cos(2.0 * M_PI * y / n);
    }
    float *dev_in = nullptr, *dev_out = nullptr;
    check(cudaMalloc(&dev_in, count * sizeof(float)), "malloc in");
    check(cudaMalloc(&dev_out, count * sizeof(float)), "malloc out");
    check(cudaMemcpy(dev_in, host.data(), count * sizeof(float), cudaMemcpyHostToDevice), "copy");

    launch_tiled(dev_in, dev_out, n, scale, 1);
    check(cudaDeviceSynchronize(), "tiled");
    check(cudaMemcpy(back.data(), dev_out, count * sizeof(float), cudaMemcpyDeviceToHost), "back");

    // CPU twin, same weights and the same neighbor order as stencil_naive.
    std::vector<float> cpu(count);
    auto wrap = [&](int index) { int m = index % n; return m < 0 ? m + n : m; };
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                auto at = [&](int dx, int dy, int dz) {
                    return host[(wrap(z + dz) * n + wrap(y + dy)) * n + wrap(x + dx)];
                };
                float sum = -14.f * at(0, 0, 0);
                sum += at(1, 0, 0) + at(-1, 0, 0) + at(0, 1, 0) + at(0, -1, 0) + at(0, 0, 1) + at(0, 0, -1);
                const int edges[12][3] = {{1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1},  {1, 0, -1},
                                          {-1, 0, 1}, {-1, 0, -1}, {0, 1, 1},  {0, 1, -1},  {0, -1, 1}, {0, -1, -1}};
                const int corners[8][3] = {{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
                                           {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
                for (auto offset : edges) sum += 0.5f * at(offset[0], offset[1], offset[2]);
                for (auto offset : corners) sum += 0.25f * at(offset[0], offset[1], offset[2]);
                cpu[(z * n + y) * n + x] = sum * scale;
            }
        }
    }
    float worst = 0.f;
    for (int i = 0; i < count; ++i) worst = std::max(worst, std::abs(back[i] - cpu[i]));

    const float tiled_ms = elapsed_ms(launch_tiled, dev_in, dev_out, n, scale, 20);
    const float naive_ms = elapsed_ms(launch_naive, dev_in, dev_out, n, scale, 20);
    const double bytes = 2.0 * count * sizeof(float);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    // CUDA 13 removed cudaDeviceProp::memoryClockRate. The clock is still in kHz.
    int memory_clock_khz = 0;
    int bus_bits = 0;
    check(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0), "memory clock");
    check(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0), "bus width");
    const double peak_gb = 2.0 * memory_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    const double tiled_gb = (bytes / 1e9) / (tiled_ms / 1e3);

    // cuSPARSE SpMV of the same 27 nonzeros. The matrix is built once.
    const int nnz = count * 27;
    std::vector<int> row(count + 1), col(nnz);
    std::vector<float> val(nnz);
    int cursor = 0;
    auto push = [&](int index, float weight) {
        col[cursor] = index;
        val[cursor] = weight * scale;
        ++cursor;
    };
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                int row_index = (z * n + y) * n + x;
                row[row_index] = cursor;
                push(row_index, -14.f);
                const int offsets[26][4] = {
                    {1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, -1, 0},
                    {1, 1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {-1, -1, 0, 1}, {1, 0, 1, 1}, {1, 0, -1, 1},
                    {-1, 0, 1, 1}, {-1, 0, -1, 1}, {0, 1, 1, 1}, {0, 1, -1, 1}, {0, -1, 1, 1}, {0, -1, -1, 1},
                    {1, 1, 1, 2}, {1, 1, -1, 2}, {1, -1, 1, 2}, {1, -1, -1, 2},
                    {-1, 1, 1, 2}, {-1, 1, -1, 2}, {-1, -1, 1, 2}, {-1, -1, -1, 2},
                };
                const float weights[3] = {1.f, 0.5f, 0.25f};
                for (auto offset : offsets) {
                    int xx = (x + offset[0] + n) % n;
                    int yy = (y + offset[1] + n) % n;
                    int zz = (z + offset[2] + n) % n;
                    push((zz * n + yy) * n + xx, weights[offset[3]]);
                }
            }
        }
    }
    row[count] = cursor;

    int *d_row = nullptr, *d_col = nullptr;
    float *d_val = nullptr, *d_sparse = nullptr;
    check(cudaMalloc(&d_row, (count + 1) * sizeof(int)), "row");
    check(cudaMalloc(&d_col, nnz * sizeof(int)), "col");
    check(cudaMalloc(&d_val, nnz * sizeof(float)), "val");
    check(cudaMalloc(&d_sparse, count * sizeof(float)), "sparse out");
    check(cudaMemcpy(d_row, row.data(), (count + 1) * sizeof(int), cudaMemcpyHostToDevice), "row copy");
    check(cudaMemcpy(d_col, col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice), "col copy");
    check(cudaMemcpy(d_val, val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice), "val copy");

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vec_in, vec_out;
    cusparseCreateCsr(&mat, count, count, nnz, d_row, d_col, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vec_in, count, dev_in, CUDA_R_32F);
    cusparseCreateDnVec(&vec_out, count, d_sparse, CUDA_R_32F);
    float alpha = 1.f, beta = 0.f;
    size_t buffer_bytes = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_in, &beta, vec_out,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_bytes);
    void* buffer = nullptr;
    check(cudaMalloc(&buffer, buffer_bytes), "spmv buffer");
    auto sparse_once = [&]() {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_in, &beta, vec_out, CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    };
    sparse_once();
    check(cudaDeviceSynchronize(), "spmv");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 20; ++i) sparse_once();
    cudaEventRecord(stop);
    check(cudaEventSynchronize(stop), "spmv timer");
    float sparse_ms = 0;
    cudaEventElapsedTime(&sparse_ms, start, stop);
    sparse_ms /= 20.f;

    std::printf("gpu %s\n", prop.name);
    std::printf("peak_dram_gb_per_s %.3f\n", peak_gb);
    std::printf("tiled_gb_per_s %.3f\n", tiled_gb);
    std::printf("tiled_fraction_of_peak %.4f\n", tiled_gb / peak_gb);
    std::printf("tiled_ms %.4f\n", tiled_ms);
    std::printf("naive_cuda_ms %.4f\n", naive_ms);
    std::printf("cusparse_ms %.4f\n", sparse_ms);
    std::printf("speedup_vs_cusparse %.3f\n", sparse_ms / tiled_ms);
    std::printf("gpu_vs_cpu_inf %.9e\n", worst);
    std::printf("A_pass_80pct_dram %d\n", tiled_gb / peak_gb > 0.80);
    std::printf("D_pass_inf_1e-6 %d\n", worst < 1e-6f);
    return worst < 1e-6f ? 0 : 2;
}
