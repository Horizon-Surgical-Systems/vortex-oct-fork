#pragma once

#include <functional>
#include <array>
#include <optional>
#include <memory>
#include <type_traits>

#include <vortex/format/plan.hpp>

#include <vortex/memory/cuda.hpp>

#include <vortex/util/sync.hpp>

namespace vortex::engine {

    template<typename block_t>
    struct adapter {
        using spectra_stream_t = typename block_t::spectra_stream_t;
        using ascan_stream_t = typename block_t::ascan_stream_t;
        using io_stream_ts = decltype(std::declval<block_t>().streams());

        using spectra_stream_factory_t = std::function<spectra_stream_t()>;
        using ascan_stream_factory_t = std::function<ascan_stream_t()>;
        using shape_t = std::array<size_t, 3>;
        using stride_t = std::array<ptrdiff_t, 3>;

        struct detail {
            struct base {
                template<typename T>
                base(const T* target) : _target(target) {}

                bool operator==(const base& o) const { return _target == o._target; }
                bool operator<(const base& o) const { return _target < o._target; }

            private:
                const void* _target;
            };
        };

        struct acquisition : detail::base {
            using detail::base::base;

            std::function<std::optional<cuda::device_t>()> device;
            std::function<spectra_stream_factory_t()> stream_factory;

            std::function<shape_t()> output_shape;
            std::function<stride_t()> output_stride;
            std::function<size_t()> channels_per_sample;

            std::function<void()> prepare, start, stop;

            using callback_t = std::function<void(size_t, std::exception_ptr)>;
            std::function<void(block_t&, spectra_stream_t&, callback_t&&)> next_async;
            std::function<void(block_t&, spectra_stream_t&)> recycle;
        };

        struct processor : detail::base {
            using detail::base::base;

            std::function<std::optional<cuda::device_t>()> device;
            std::function<ascan_stream_factory_t()> stream_factory;

            std::function<shape_t()> output_shape, input_shape;

            std::function<size_t()> channel;

            using callback_t = std::function<void(std::exception_ptr)>;
            std::function<void(block_t&, const spectra_stream_t&, ascan_stream_t&, cuda::event_t*, cuda::event_t*, callback_t&&)> next_async;
        };

        struct io : detail::base {
            using detail::base::base;

            std::function<void()> prepare, start, stop;

            using callback_t = std::function<void(size_t, std::exception_ptr)>;
            std::function<void(block_t&, io_stream_ts, callback_t&&)> next_async;
        };

        struct formatter : detail::base {
            using detail::base::base;

            std::function<format::format_plan_t(const block_t&)> next;
            std::function<void(format::format_plan_t&)> finish;
        };

        struct endpoint : detail::base {
            using detail::base::base;

            std::function<void(const std::optional<cuda::device_t>&, const std::optional<cuda::device_t>&)> allocate;

            std::function<void(const format::format_plan_t&, const block_t&, const spectra_stream_t&, const ascan_stream_t&)> handle;
        };

    };

    namespace bind {

        struct base_t {};

        template<typename block_t, typename T>
        auto acquisition(std::shared_ptr<T> a, const base_t& = base_t()) {
            if (!a) {
                throw std::invalid_argument("non-null acquisition required");
            }

            using adapter = adapter<block_t>;
            auto w = typename adapter::acquisition(a.get());

            w.device = []() -> std::optional<cuda::device_t> { return {}; };
            w.stream_factory = []() {
                return []() -> typename adapter::spectra_stream_t {
                    return sync::lockable<cuda::cuda_host_tensor_t<typename block_t::acquire_element_t>>();
                };
            };

            w.output_shape = [a]() { return a->config().shape(); };
            w.output_stride = [a]() { return a->config().stride(); };
            w.channels_per_sample = [a]() { return a->config().channels_per_sample(); };

            w.prepare = [a]() { a->prepare(); };
            w.start = [a]() { a->start(); };
            w.stop = [a]() { a->stop(); };

            //w.next_async = [a](block_t& block, typename adapter::spectra_stream_t& stream_, typename adapter::acquisition::callback_t&& callback) {
            //    std::visit([&](auto& stream) { a->next_async(block.id, block.length, view(stream), std::forward<typename adapter::acquisition::callback_t>(callback)); }, stream_);
            //};
            w.recycle = [a](block_t&, typename adapter::spectra_stream_t&) {};

            return w;
        }

        template<typename block_t, typename T>
        auto processor(std::shared_ptr<T> a, const base_t& = base_t()) {
            if (!a) {
                throw std::invalid_argument("non-null processor required");
            }

            using adapter = adapter<block_t>;
            auto w = typename adapter::processor(a.get());

            w.device = [a]() -> std::optional<cuda::device_t> { return a->config().device; };
            w.stream_factory = []() {
                return []() -> typename adapter::ascan_stream_t {
                    return sync::lockable<cuda::cuda_device_tensor_t<typename block_t::process_element_t>>();
                };
            };

            w.input_shape = [a]() { return a->config().input_shape(); };
            w.output_shape = [a]() { return a->config().output_shape(); };

            w.channel = [a]() { return a->config().channel; };

            w.next_async = [a](block_t& block,
                const typename adapter::spectra_stream_t& input_stream_, typename adapter::ascan_stream_t& output_stream_,
                cuda::event_t* start, cuda::event_t* done, typename adapter::processor::callback_t&& callback) {
                std::visit([&](auto& input_stream, auto& output_stream) {
                    try {
                        view_as_cuda([&](auto input_buffer, auto output_buffer) {
                            a->next_async(
                                block.id, input_buffer.range(block.length), output_buffer.range(block.length),
                                start, done, std::forward<typename adapter::processor::callback_t>(callback)
                            );
                        }, input_stream, output_stream);
                    } catch (const unsupported_view&) {
                        callback(std::current_exception());
                    }
                }, input_stream_, output_stream_);
            };

            return w;
        }

        template<typename block_t, typename T>
        auto io(std::shared_ptr<T> a, const base_t& = base_t()) {
            if (!a) {
                throw std::invalid_argument("non-null IO required");
            }

            using adapter = adapter<block_t>;
            auto w = typename adapter::io(a.get());

            w.prepare = [a]() { a->prepare(); };
            w.start = [a]() { a->start(); };
            w.stop = [a]() { a->stop(); };

            w.next_async = [a](block_t& block, typename adapter::io_stream_ts streams, typename adapter::io::callback_t&& callback) {
                try {
                    view_tuple_as_cpu([&](auto buffers) {
                        a->next_async(block.id, block.length, buffers, std::forward<typename adapter::io::callback_t>(callback));
                    }, streams);
                } catch (const unsupported_view&) {
                    callback(0, std::current_exception());
                }
            };

            return w;
        }

        template<typename block_t, typename T>
        auto formatter(std::shared_ptr<T> a, const base_t& = base_t()) {
            if (!a) {
                throw std::invalid_argument("non-null formatter required");
            }

            using adapter = adapter<block_t>;
            auto w = typename adapter::formatter(a.get());

            w.next = [a](const block_t& block) { return a->next(block.sample, block.length, block.markers); };

            w.finish = [a](format::format_plan_t& plan) { a->finish(plan); };

            return w;
        }

        template<typename block_t, typename T>
        auto endpoint(std::shared_ptr<T> a, const base_t& = base_t()) {
            if (!a) {
                throw std::invalid_argument("non-null endpoint required");
            }

            using adapter = adapter<block_t>;
            auto w = typename adapter::endpoint(a.get());

            w.allocate = [a](const std::optional<cuda::device_t>& spectra, const std::optional<cuda::device_t>& ascans) { a->allocate(spectra, ascans); };

            w.handle = [a](const format::format_plan_t& plan, const block_t& block,
                const typename adapter::spectra_stream_t& spectra_, const typename adapter::ascan_stream_t& ascans_) {
                std::visit([&](auto& spectra, auto& ascans) {
                    a->handle(plan, block, spectra, ascans);
                }, spectra_, ascans_);
            };

            return w;
        }

    }

}
