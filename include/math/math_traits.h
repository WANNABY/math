#ifndef MATH_VECTOR_TRAITS_H_
#define MATH_VECTOR_TRAITS_H_

#include <type_traits>
#include "math/matrix.h"
#include "math/vector_int.h"


namespace math {

// Checks whether Vec is a floating-point vector type.
template<typename V>
constexpr bool is_floating_point_vector() noexcept
{
	return std::is_same<V, float2>::value
		|| std::is_same<V, float3>::value
		|| std::is_same<V, float4>::value;
}

// Checks wheterh V is an integral vector.
template<typename V>
constexpr bool is_integral_vector()
{
	return std::is_same<V, int2>::value
		|| std::is_same<V, int4>::value
		|| std::is_same<V, ubyte4>::value
		|| std::is_same<V, uint2>::value
		|| std::is_same<V, uint4>::value;
}

// Checks whether M is a matrix;
template<typename M>
constexpr bool is_matrix() noexcept
{
	return std::is_same<M, float3x3>::value
		|| std::is_same<M, float4x4>::value;
}

// vector_traits provides various properties of vector types.
template<typename T>
struct vector_traits;

template<>
struct vector_traits<float2> final {
	using component_type = float;

	static constexpr size_t component_count = 2;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

template<>
struct vector_traits<float3> final {
	using component_type = float;

	static constexpr size_t component_count = 3;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

template<>
struct vector_traits<float4> final {
	using component_type = float;

	static constexpr size_t component_count = 4;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

template<typename T>
struct vector_traits<vec_int_2<T>> final {
	using component_type = T;

	static constexpr size_t component_count = 2;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

template<typename T>
struct vector_traits<vec_int_3<T>> final {
	using component_type = T;

	static constexpr size_t component_count = 3;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

template<typename T>
struct vector_traits<vec_int_4<T>> final {
	using component_type = T;
	static constexpr size_t component_count = 4;
	static constexpr size_t byte_count		= sizeof(component_type) * component_count;
};

} // namespace math

#endif // MATH_VECTOR_TRAITS_H_
