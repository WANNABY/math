#ifndef MATH_NDOP_H_
#define MATH_NDOP_H_

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <numbers>
#include <functional>
#include <optional>
#include <vector>

#include "math/math.h"

namespace math::utils {

struct DOP14_Traits
{
    constexpr static size_t get_num_axises() noexcept { return std::size(axises); }
    constexpr static size_t get_num_planes() noexcept { return get_num_axises() * 2; }

    constexpr std::pair<size_t, size_t> get_axis_indices(size_t index) const noexcept {
        assert(index < get_num_axises());
        return {index, index + get_num_axises()};
    }

    constexpr static float3 get_axis_direction(size_t index) noexcept {
        assert(index < axises.size());
        return axises[index];
    }

    constexpr static float3 get_plane_normal(size_t index) noexcept {
        assert(index < get_num_planes());
        return index >= get_num_axises() ? -axises[index - get_num_axises()] : axises[index];
     }

private:
    constexpr static auto inv_sqrt3 = std::numbers::inv_sqrt3_v<float>;
    constexpr static std::array<float3, 7> axises {
        float3{1.0f, 0.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        float3{0.0f, 0.0f, 1.0f},
        float3{inv_sqrt3, inv_sqrt3, inv_sqrt3},
        float3{inv_sqrt3, inv_sqrt3, -inv_sqrt3},
        float3{inv_sqrt3, -inv_sqrt3, inv_sqrt3},
        float3{-inv_sqrt3, inv_sqrt3, inv_sqrt3},
    };

    //static_assert(approx_equal(len_squared(axises[i]), 1.0f));
};

//template <auto Traits>
class DOP : DOP14_Traits {
   public:
    struct AxisSpan {
        float min, max;

        constexpr float get_extent() const noexcept { return max - min; }
        constexpr float get_center() const noexcept { return std::midpoint(min, max); }
    };

   public:
    constexpr explicit DOP() noexcept = default;

    template <typename Iterator, typename UnaryFn = std::identity>
    constexpr static DOP compute(
        Iterator vertices_begin, Iterator vertices_end, UnaryFn&& fn_get_position = {}) noexcept;

    constexpr AxisSpan get_axis(size_t index) const noexcept;

    std::vector<float3> get_points() const noexcept;
    constexpr std::optional<float3> get_center() const noexcept;
    constexpr bool is_empty() const noexcept;
    constexpr float3 get_size() const noexcept;

    DOP transformed(const float4x4& transform) const noexcept;
    constexpr DOP unite_with(const DOP& rhv) const noexcept;
    constexpr DOP unite_with(const float3& point) const noexcept;

    constexpr void expand (const float3& point) noexcept;
    
    constexpr bool contains(const float3& point) const noexcept;

   private:
    std::array<float, get_num_planes()> distances_ 
    {
        [](){
            std::array<float, get_num_planes()> result;
            std::fill(std::begin(result), std::end(result), std::numeric_limits<float>::lowest());
            return result;
        }()
    };
};

constexpr DOP::AxisSpan DOP::get_axis(size_t index) const noexcept {
    const auto [pos_index, neg_index] = get_axis_indices(index);
    return {-distances_[neg_index], distances_[pos_index]};
}

constexpr std::optional<float3> DOP::get_center() const noexcept {
	if (is_empty()) {
		return std::nullopt;
	}

	return float3{
		get_axis(0).get_center(),
		get_axis(1).get_center(),
		get_axis(2).get_center(),
	};
}

constexpr bool DOP::is_empty() const noexcept { 
    return std::any_of(std::begin(distances_), std::end(distances_), [](const auto& d) { return d <= std::numeric_limits<float>::lowest(); });
}

constexpr float3 DOP::get_size() const noexcept {
	if (is_empty()) {
		return float3{};
	}

	return {
		get_axis(0).get_extent(),
		get_axis(1).get_extent(),
		get_axis(2).get_extent(),
	};
}

template <typename Iterator, typename UnaryFn>
constexpr DOP DOP::compute(Iterator vertices_begin, Iterator vertices_end, UnaryFn&& fn_get_position) noexcept {
    return std::accumulate(vertices_begin, vertices_end, DOP{}, [fn_get_position = std::forward<UnaryFn>(fn_get_position)](DOP bounds, const auto& element) {
        const auto vertex = fn_get_position(element);
        return bounds.unite_with(vertex);
    });
}

inline DOP DOP::transformed(const float4x4& transform) const noexcept {
    auto&& points = get_points();
    return compute(points.begin(), points.end(), [&transform](const auto& pos) {
        const auto transformed_position = mul(transform, pos);
        return xyz(transformed_position) / transformed_position.w;
     });
}

constexpr DOP DOP::unite_with(const DOP& rhv) const noexcept {
    DOP result = *this;
    for (size_t i = 0; i < get_num_planes(); ++i) {
        result.distances_[i] = std::max(distances_[i], rhv.distances_[i]);
    }
    return result;
}

constexpr DOP DOP::unite_with(const float3& point) const noexcept {
    DOP result = *this;
    result.expand(point);
    return result;
}

constexpr void DOP::expand(const float3& point) noexcept {
    for (size_t i = 0; i < get_num_planes(); ++i) {
        const auto projected_distance = dot(point, get_plane_normal(i));
        distances_[i] = std::max(distances_[i], projected_distance);
    }
}

constexpr bool DOP::contains(const float3& point) const noexcept {
    constexpr float epsilon = 1e-5f;
    for (size_t i = 0; i < get_num_planes(); ++i) {
        const auto projected_distance = dot(point, get_plane_normal(i));
        if ( projected_distance > distances_[i] + epsilon) {
            return false;
        }
    }

    return true;
}

inline std::vector<float3> DOP::get_points() const noexcept {
    struct AdjacencyInfo {
        size_t i, j, k;
        float3x3 planes;
        float inv_determinant;

        auto compute_planes_intersection_point (const DOP& self) const noexcept {
            const float3 distances {self.distances_[i], self.distances_[j], self.distances_[k]};
            const auto solve_for = [this, distances ](auto set_column_func) noexcept {
                auto m = planes;
                set_column_func(m, distances);
                return det(m) * inv_determinant;
            };

            return float3{solve_for(set_ox<float3x3>), solve_for(set_oy<float3x3>), solve_for(set_oz<float3x3>)};
        };
    };
    
    // precalculations, done once per traits type
    static const auto planes_adjacency = [](){
        std::vector<AdjacencyInfo> result;

        for (size_t i = 0; i < get_num_planes(); ++i) {
            for (size_t j = i + 1; j < get_num_planes(); ++j ) {
                for (size_t k = j + 1; k < get_num_planes(); ++k) {        
                    const auto i_normal = get_plane_normal(i);
                    const auto j_normal = get_plane_normal(j);
                    const auto k_normal = get_plane_normal(k);

                    if (dot(i_normal, j_normal) < 0.0f || dot(i_normal, k_normal) < 0.0f || dot(j_normal, k_normal) < 0.0f) {
                        continue;
                    }

                    const float3x3 planes{i_normal.x, i_normal.y, i_normal.z, j_normal.x, j_normal.y, j_normal.z, k_normal.x, k_normal.y, k_normal.z};
                    auto d = det(planes);
                    if (approx_equal(d, 0.0f)) {
                        continue;
                    }

                    result.push_back({i, j, k, planes, 1.0f / d});
                }
            }
        }
        
        return result;
    }();


    std::vector<float3> result;
    result.reserve(planes_adjacency.size());

    for (const auto& adjacency : planes_adjacency) {
        const auto point = adjacency.compute_planes_intersection_point(*this);
        if (!contains(point)) {
            continue;
        }
        
        result.push_back(point);
    }

    return result;
}

} // namespace math::utils

#endif
