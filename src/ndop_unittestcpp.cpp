#include "math/utils/ndop.h"
#include "CppUnitTest.h"

#include <ranges>

namespace {
	auto generate_points_on_line(math::float3 start, math::float3 end, int n) {
		using namespace std::ranges;
		return views::iota(0, n)
			| views::transform([=, k = 1.0f / n](int x) {return math::lerp(start, end, x * k); })
			| views::common;
	}

	auto generate_points_on_xy_circle(math::float3 center, float radius, int n) {
		using namespace std::ranges;
		return views::iota(0, n)
			| views::transform([=, k = std::numbers::pi_v<float> * 2.0f / n](int x) { return center + math::float3{ cos(x * k), sin(x * k), 0.0f } * radius; })
			| views::common;
	}

	auto remove_duplicate_points(std::vector<math::float3>& points) {
		std::ranges::sort(points, {}, [](math::float3 v) { return std::tuple{ v.x, v.y, v.z }; });
		points.erase(std::unique(points.begin(), points.end(), [](auto x, auto y) { return math::approx_equal(x, y); }), points.end());

	}
}

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using math::utils::DOP;

namespace unittest {

	TEST_CLASS(ndop_tests) {
public:
	TEST_METHOD(emptyness)
	{
		//auto points = generate_points_on_line({ -1, -1, -1 }, { 1, 1, 1 }, 100);
		//auto dop = DOP::compute(points.cbegin(), points.cend());

		{
			DOP empty;
			Assert::IsTrue(empty.is_empty());
		}

		{
			DOP dop;
			dop.expand({ 0, 0, 0 });

			Assert::IsFalse(dop.is_empty());
		}

	}

	TEST_METHOD(two_points_dop_size_and_center) {
		constexpr math::float3 start{ 10, 10, 10 };
		constexpr math::float3 length { 10, 0, 0 };

		DOP dop;
		Assert::IsFalse(dop.get_center().has_value());

		dop.expand(start);
		Assert::IsFalse(dop.is_empty());
		Assert::IsTrue(math::approx_equal(dop.get_size(), {}));
		Assert::IsTrue(math::approx_equal(dop.get_center().value(), start));

		dop.expand(start + length);

		Assert::IsFalse(dop.is_empty());
		Assert::IsTrue(math::approx_equal(dop.get_size(), length));
		Assert::IsTrue(math::approx_equal(dop.get_center().value(), start + length / 2));
	}

	TEST_METHOD(points_along_axises) {
		DOP dop;
		dop.expand({ 0.0, 0.0, 0.0 });
		dop.expand({ 1.0, 0.0, 0.0 });
		dop.expand({ 0.0, 1.0, 0.0 });
		dop.expand({ 0.0, 0.0, 1.0 });


		Assert::IsFalse(dop.is_empty());
		Assert::IsTrue(math::approx_equal(dop.get_size(), {1.0, 1.0, 1.0}));

		Assert::IsTrue(dop.contains({0.0, 0.0, 0.0}));
		Assert::IsTrue(dop.contains({ 1.0, 0.0, 0.0 }));
		Assert::IsTrue(dop.contains({ 0.0, 1.0, 0.0 }));
		Assert::IsTrue(dop.contains({ 0.0, 0.0, 1.0 }));

		Assert::IsFalse(dop.contains({ -1.0, 0.0, 0.0 }));
		Assert::IsFalse(dop.contains({ 1.0, 1.0, 0.0 }));

		//{
		//	DOP rotated = dop.transformed(math::rotation_matrix_ox<math::float4x4>(math::pi_4));
		//	Assert::IsTrue(math::approx_equal(rotated.get_size(), {}));
		//}
	}

	TEST_METHOD(ox_sphere) {
		constexpr math::float3 center{};//{ 192, 300, -400 };
		constexpr float diameter = 1.0f;
		auto points = generate_points_on_xy_circle(center, diameter / 2.0f, 100);
		std::vector a(points.begin(), points.end());
		auto dop = DOP::compute(points.begin(), points.end());

		Assert::IsTrue(math::approx_equal(dop.get_size(), { diameter, diameter, 0.0}));
		Assert::IsTrue(math::approx_equal(dop.get_center().value(), center));

		auto vertices = dop.get_points();
		remove_duplicate_points(vertices);

		for (auto point : vertices) {
			Assert::IsTrue(math::len(point - center) <= diameter);
		}
	}

	TEST_METHOD(cross_dop) {
		DOP dop;
		dop.expand({ -1, 0, 0 });
		dop.expand({ 1, 0, 0 });
		dop.expand({ 0, -1, 0 });
		dop.expand({ 0,  1, 0 });
		dop.expand({ 0, 0, -1 });
		dop.expand({ 0, 0,  1 });

		Assert::IsTrue(math::approx_equal(dop.get_size(), { 2, 2, 2 }));
		auto points = dop.get_points();
		remove_duplicate_points(points);

		Assert::IsTrue(points.size() >= 6);

		Assert::IsTrue(std::ranges::all_of(points, [](auto vec) {return math::approx_equal(math::len(vec), 1.0f); }));
	}

	};

}
