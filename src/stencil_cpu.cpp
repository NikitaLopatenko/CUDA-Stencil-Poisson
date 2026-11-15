// 27-point periodic Laplacian. The sum of the weights is zero, so a constant field stays zero.
// Faces 1, edges 1/2, corners 1/4, center -14, then divide by 6 h^2.
// Neighbors are added in a fixed order: faces, edges, corners. The CUDA kernel uses that same order.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kFaces[6][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
};
constexpr int kEdges[12][3] = {
    {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1}, {1, 0, -1},
    {-1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1},
};
constexpr int kCorners[8][3] = {
    {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
    {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1},
};

int wrap(int index, int n) {
    int m = index % n;
    return m < 0 ? m + n : m;
}

template <typename T>
T stencil_at(const T* field, int n, int x, int y, int z, T scale) {
    auto at = [&](int dx, int dy, int dz) -> T {
        return field[(wrap(z + dz, n) * n + wrap(y + dy, n)) * n + wrap(x + dx, n)];
    };
    T sum = T(-14) * at(0, 0, 0);
    for (const auto& offset : kFaces) sum += at(offset[0], offset[1], offset[2]);
    for (const auto& offset : kEdges) sum += T(0.5) * at(offset[0], offset[1], offset[2]);
    for (const auto& offset : kCorners) sum += T(0.25) * at(offset[0], offset[1], offset[2]);
    return sum * scale;
}

template <typename T>
void apply(const T* field, T* out, int n, T h) {
    const T scale = T(1) / (T(6) * h * h);
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                out[(z * n + y) * n + x] = stencil_at(field, n, x, y, z, scale);
            }
        }
    }
}

template <typename T>
T inf_norm(const T* a, const T* b, int count) {
    T worst = 0;
    for (int i = 0; i < count; ++i) {
        T gap = std::abs(a[i] - b[i]);
        if (gap > worst) worst = gap;
    }
    return worst;
}

}  // namespace

int main() {
    const int n = 48;
    const float h = 1.f;
    const int count = n * n * n;
    std::vector<float> field(count), out_f(count);
    std::vector<double> field_d(count), out_d(count);
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                double value = std::sin(2.0 * M_PI * x / n) * std::cos(2.0 * M_PI * y / n) * std::cos(4.0 * M_PI * z / n);
                field[(z * n + y) * n + x] = static_cast<float>(value);
                field_d[(z * n + y) * n + x] = value;
            }
        }
    }
    apply(field.data(), out_f.data(), n, h);
    apply(field_d.data(), out_d.data(), n, static_cast<double>(h));
    std::vector<float> out_from_double(count);
    for (int i = 0; i < count; ++i) out_from_double[i] = static_cast<float>(out_d[i]);
    const double versus_double = static_cast<double>(inf_norm(out_f.data(), out_from_double.data(), count));

    // A constant has a zero Laplacian. This is the same weight check the kernel has to pass.
    std::vector<float> constant(count, 1.f), constant_out(count);
    apply(constant.data(), constant_out.data(), n, h);
    const float constant_residual = inf_norm(constant_out.data(), std::vector<float>(count, 0.f).data(), count);

    std::FILE* raw = std::fopen("runs/cpu_float.bin", "wb");
    if (raw) {
        std::fwrite(out_f.data(), sizeof(float), count, raw);
        std::fclose(raw);
    }
    std::printf("n %d\n", n);
    std::printf("fp32_vs_fp64_inf %.9e\n", versus_double);
    std::printf("constant_laplacian_inf %.9e\n", constant_residual);
    std::printf("fp32_center %.9e\n", out_f[(n / 2 * n + n / 2) * n + n / 2]);
    return versus_double < 1e-6 && constant_residual < 1e-6 ? 0 : 2;
}
