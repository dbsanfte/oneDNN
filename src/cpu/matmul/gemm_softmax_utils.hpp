#ifndef CPU_MATMUL_GEMM_SOFTMAX_UTILS_HPP
#define CPU_MATMUL_GEMM_SOFTMAX_UTILS_HPP

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/gemm_based_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {
namespace softmax_utils {

inline void normalize_softmax_buffer(
        float *values, float *scratch, dim_t len, bool log_softmax) {
    if (len <= 0) return;

    float max_val = -FLT_MAX;
    for (dim_t i = 0; i < len; ++i)
        max_val = std::max(max_val, values[i]);

    float sum = 0.f;
    PRAGMA_OMP_SIMD(reduction(+ : sum))
    for (dim_t i = 0; i < len; ++i) {
        float shifted = values[i] - max_val;
        if (log_softmax) {
            scratch[i] = shifted;
            sum += std::exp(shifted);
        } else {
            float ex = std::exp(shifted);
            scratch[i] = ex;
            sum += ex;
        }
    }

    if (log_softmax) {
        const float log_sum = sum > 0.f ? std::log(sum) : 0.f;
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < len; ++i)
            values[i] = scratch[i] - log_sum;
    } else {
        const float inv_sum = sum > 0.f ? 1.f / sum : 0.f;
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < len; ++i)
            values[i] = scratch[i] * inv_sum;
    }
}

template <typename data_t>
inline void load_row(const data_t *src, dim_t len, float *buffer) {
    for (dim_t i = 0; i < len; ++i)
        buffer[i] = static_cast<float>(src[i]);
}

template <>
inline void load_row<float>(const float *src, dim_t len, float *buffer) {
    utils::array_copy(buffer, src, len);
}

template <typename data_t>
inline void store_row(data_t *dst, dim_t len, const float *buffer) {
    for (dim_t i = 0; i < len; ++i)
        dst[i] = data_t(buffer[i]);
}

template <>
inline void store_row<float>(float *dst, dim_t len, const float *buffer) {
    utils::array_copy(dst, buffer, len);
}

template <typename dst_data_t>
status_t apply_softmax_post_op(dst_data_t *dst_ptr,
        const memory_desc_wrapper &dst_d, const gemm_based::params_t &params) {
    if (!params.has_softmax_post_op_) return status::success;

    const int ndims = dst_d.ndims();
    int axis = params.softmax_axis_;
    if (axis < 0) axis += ndims;
    const dim_t axis_dim = dst_d.dims()[axis];
    if (axis_dim <= 0) return status::success;

    const dim_t total_elems = dst_d.nelems();
    if (total_elems == 0) return status::success;

    const auto &strides = dst_d.blocking_desc().strides;
    const int nthr = dnnl_get_max_threads();

    const bool axis_is_last = axis == ndims - 1;
    const bool axis_is_column = (axis == 0 && ndims == 2);

    if (!axis_is_last && !axis_is_column) return status::unimplemented;

    if (axis_is_last) {
        const size_t outer = static_cast<size_t>(total_elems) / axis_dim;
        const dim_t ld = (axis == 0) ? axis_dim : strides[axis - 1];

        parallel(nthr, [&](int ithr, int nthr) {
            size_t start {}, end {};
            balance211(outer, nthr, ithr, start, end);
            std::vector<float> raw(axis_dim);
            std::vector<float> scratch(axis_dim);
            for (size_t row = start; row < end; ++row) {
                dst_data_t *row_ptr = dst_ptr + row * ld;
                load_row(row_ptr, axis_dim, raw.data());
                normalize_softmax_buffer(raw.data(), scratch.data(), axis_dim,
                        params.softmax_log_);
                store_row(row_ptr, axis_dim, raw.data());
            }
        });
        return status::success;
    }

    const dim_t rows = dst_d.dims()[0];
    const dim_t cols = dst_d.dims()[1];
    const dim_t row_stride = strides[0];
    const dim_t col_stride = strides[1];

    parallel(nthr, [&](int ithr, int nthr) {
        dim_t start {}, end {};
        balance211(cols, nthr, ithr, start, end);
        std::vector<float> raw(rows);
        std::vector<float> scratch(rows);
        for (dim_t col = start; col < end; ++col) {
            dst_data_t *col_ptr = dst_ptr + col * col_stride;
            for (dim_t r = 0; r < rows; ++r)
                raw[r] = static_cast<float>(col_ptr[r * row_stride]);

            normalize_softmax_buffer(
                    raw.data(), scratch.data(), rows, params.softmax_log_);

            for (dim_t r = 0; r < rows; ++r)
                col_ptr[r * row_stride] = dst_data_t(raw[r]);
        }
    });

    return status::success;
}

} // namespace softmax_utils
} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
