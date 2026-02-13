#pragma once

#include <xtensor/views/xstrided_view.hpp>

#include <fmt/format.h>

#include <vortex/util/exception.hpp>

namespace vortex {

    class ring_buffer_xt {
    public:

        ring_buffer_xt() {
            clear();
        }
        ring_buffer_xt(size_t head, size_t tail) {
            reset(head, tail);
        }

        template<typename B1, typename B2>
        void store(B1& buffer, B2& input, size_t count, size_t offset = 0) {
            // check for unexpected buffer resize
            auto size = buffer.shape(0);
            if (_head >= size) {
                throw traced<std::runtime_error>(fmt::format("head is past end of buffer: {} >= {}", _head, size));
            }

            if (count >= size) {

                // move head to start for efficiency
                clear();

                // copy tail of input
                auto src = xt::strided_view(input, { xt::range(offset + count - size, offset + count), xt::ellipsis() });
                auto& dst = buffer;
                dst = src;

                // head remains at zero

            } else {

                size_t copied = 0;
                // NOTE: loop will execute at most twice (for a copy split across end of ring)
                while (copied < count) {

                    // choose maximum possible chunk
                    size_t n = std::min(count - copied, size - _head);

                    // copy chunk
                    auto src = xt::strided_view(input, { xt::range(offset + copied, offset + copied + n), xt::ellipsis() });
                    auto dst = xt::strided_view(buffer, { xt::range(_head, _head + n), xt::ellipsis() });
                    xt::noalias(dst) = src;

                    // update state
                    _head = (_head + n) % size;
                    copied += n;
                }

            }
        }

        template<typename B1, typename B2>
        void load(B1& buffer, B2& output, size_t count, size_t offset = 0) {
            // check for unexpected buffer resize
            auto size = buffer.shape(0);
            if (_tail >= size) {
                throw traced<std::runtime_error>(fmt::format("tail is past end of buffer: {} >= {}", _tail, size));
            }
            if (count > size) {

                throw traced<std::runtime_error>(fmt::format("load size is larger than buffer size: {} > {}", count, size));

            } else {

                size_t copied = 0;
                // NOTE: loop will execute at most twice (for a copy split across end of ring)
                while (copied < count) {

                    // choose maximum possible chunk
                    size_t n = std::min(count - copied, size - _tail);

                    // copy chunk
                    auto src = xt::strided_view(buffer, { xt::range(_tail, _tail + n), xt::ellipsis() });
                    auto dst = xt::strided_view(output, { xt::range(offset + copied, offset + copied + n), xt::ellipsis() });
                    xt::noalias(dst) = src;

                    // update state
                    _tail = (_tail + n) % size;
                    copied += n;
                }

            }
        }

        void reset(size_t head, size_t tail) {
            _head = head;
            _tail = tail;
        }

        void clear() {
            reset(0, 0);
        }

        const auto& head() const { return _head; }
        const auto& tail () const { return _tail; }

    protected:

        size_t _head, _tail;
    };

}
