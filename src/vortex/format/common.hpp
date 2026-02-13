#pragma once

#include <tuple>

#include <vortex/memory/view.hpp>

namespace vortex::format {

    namespace detail {

        template<typename V>
        auto trim_shape_and_stride(const viewable<V>& buffer_, size_t n) {
            auto& buffer = buffer_.derived_cast();

            auto shape = buffer.shape();
            auto stride = buffer.stride();

            while (shape.size() > n) {
                if (shape.back() == 1) {
                    shape.erase(shape.end() - 1);
                    stride.erase(stride.end() - 1);
                } else {
                    break;
                }
            }
            while (shape.size() < n) {
                shape.push_back(buffer.count() > 0 ? 1 : 0);
                stride.push_back(0);
            }

            return std::make_tuple(std::move(shape), std::move(stride));
        }
    }

}