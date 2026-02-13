#pragma once

#include <vortex/driver/cuda/copy.hpp>

#include <vortex/endpoint/common.hpp>

#include <vortex/format.hpp>

#include <vortex/memory/cuda.hpp>

#include <vortex/util/sync.hpp>

namespace vortex::endpoint {

    template<typename derived_t_>
    struct cuda_tensor : detail::notify {
        using derived_t = derived_t_;

        auto& derived_cast() {
            return *static_cast<derived_t*>(this);
        }
        const auto& derived_cast() const {
            return *static_cast<const derived_t*>(this);
        }

        const auto& stream() { return _stream; }

    protected:

        cuda::stream_t _stream;

    };

    namespace detail {

        template<typename executor_t, typename tensor_t, typename derived_t>
        struct cuda_tensor_impl : cuda_tensor<derived_t> {

            cuda_tensor_impl(std::shared_ptr<executor_t> executor, std::vector<size_t> shape, std::shared_ptr<spdlog::logger> log = nullptr)
                : _executor(std::move(executor)), _shape(std::move(shape)), _log(std::move(log)) {
                _tensor = std::make_shared<sync::lockable<tensor_t>>();
                _check();
            }

            template<size_t N>
            cuda_tensor_impl(std::shared_ptr<executor_t> executor, const std::array<size_t, N>& shape, std::shared_ptr<spdlog::logger> log = nullptr)
                : _executor(std::move(executor)), _log(std::move(log)) {
                _shape.emplace(shape.begin(), shape.end());
                _tensor = std::make_shared<sync::lockable<tensor_t>>();
                _check();
            }

            cuda_tensor_impl(std::shared_ptr<executor_t> executor, std::shared_ptr<sync::lockable<tensor_t>> tensor, std::shared_ptr<spdlog::logger> log = nullptr)
                : _executor(std::move(executor)), _tensor(std::move(tensor)), _log(std::move(log)) {
                if (!_tensor) {
                    throw std::invalid_argument("non-null tensor required");
                }
            }

            const auto& tensor() const {
                return _tensor;
            }

            const auto& executor() const {
                return _executor;
            }

        protected:

            std::shared_ptr<executor_t> _executor;

            std::optional<std::vector<size_t>> _shape;
            std::shared_ptr<sync::lockable<tensor_t>> _tensor;

            std::shared_ptr<spdlog::logger> _log;

        private:

            void _check() {
                if (!_executor) {
                    throw std::invalid_argument("non-null executor required");
                }
            }

        };

        template<typename T, typename executor_t, typename source_selector_t, typename derived_t>
        struct cuda_host_tensor_impl : cuda_tensor_impl<executor_t, cuda::cuda_host_tensor_t<T>, derived_t> {
            using base_t = cuda_tensor_impl<executor_t, cuda::cuda_host_tensor_t<T>, derived_t>;
            using base_t::base_t;

            void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
                if (!_shape) {
                    // no automatic allocation
                    return;
                }

                // check if A-scans will arrive in device memory
                const auto& device = source_selector_t::select(std::optional<cuda::device_t>{}, spectra, ascans);
                if (device) {

                    // needs a device volume for formatting
                    if (_log) { _log->debug("allocating [{}] buffer on device {} for formatting", shape_to_string(*_shape), *device); }
                    cuda::device(*device);
                    _device_buffer.resize(*_shape);

                }

                // allocate
                if (_log) { _log->debug("allocating [{}] buffer on host for endpoint", shape_to_string(*_shape)); }
                std::unique_lock<std::shared_mutex> lock(_tensor->mutex());
                _tensor->resize(*_shape);
            }

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = view(source_selector_t::select(block.streams(), spectra, ascans));
                std::vector<size_t> block_segments, volume_segments;

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& a) {
                            overloaded{
                                [&] <typename V>(const cpu_viewable<V>& buffer) {
                                    std::unique_lock<std::shared_mutex> lock(_tensor->mutex());

                                    // format on host
                                    _executor->execute(view(*_tensor), buffer, a);
                                },
                                [&] <typename V>(const cuda::cuda_viewable<V>& buffer) {
                                    // format on device
                                    cuda::device(buffer.derived_cast().device());
                                    _executor->execute(_stream, view(_device_buffer), buffer, a);
                                }
                            }(source);
                        },
                        [&](const format::action::finish_volume& a) {
                            // copy formatted segments to host, if needed
                            _transfer(volume_segments);

                            // remainder of usual processing
                            _default(_log, block_segments, volume_segments, a);
                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }

                // handle any remaining segments
                _transfer(volume_segments);
                _finish(_log, block_segments, volume_segments);
            }

        protected:

            void _transfer(std::vector<size_t> segments) {
                if (segments.empty() || !_device_buffer.valid()) {
                    return;
                }
                std::unique_lock<std::shared_mutex> lock(_tensor->mutex());

                // build save ranges
                std::sort(segments.begin(), segments.end());
                auto chunks = detail::combine(segments);

                auto device_view = view(_device_buffer);
                auto tensor_view = view(*_tensor);

                // copy the data to the host for saving
                for (auto& c : chunks) {
                    if (_log) { _log->trace("transferring chunk [{}-{}) to host", c.min(), c.max()); }
                    cuda::copy(
                        device_view, device_view.offset({ c.min() }),
                        tensor_view, tensor_view.offset({ c.min() }),
                        device_view.offset({ c.max() }) - device_view.offset({ c.min() }),
                        &_stream
                    );
                }
            }

            cuda::cuda_device_tensor_t<T> _device_buffer;

            using base_t::_executor, base_t::_tensor, base_t::_stream, base_t::_shape, base_t::_log, base_t::_notify, base_t::_default, base_t::_finish;

        };

        template<typename T, typename executor_t, typename source_selector_t, typename derived_t>
        struct cuda_device_tensor_impl : cuda_tensor_impl<executor_t, cuda::cuda_device_tensor_t<T>, derived_t> {
            using base_t = cuda_tensor_impl<executor_t, cuda::cuda_device_tensor_t<T>, derived_t>;
            using base_t::base_t;

            void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
                if (!_shape) {
                    // no automatic allocation
                    return;
                }

                // check if A-scans will arrive in device memory
                const auto& device = source_selector_t::select(std::optional<cuda::device_t>{}, spectra, ascans);
                if (!device) {
                    throw std::runtime_error("A-scans must arrive in device memory for device tensor endpoints");
                }

                // allocate
                if (_log) { _log->debug("allocating [{}] buffer on device {} for endpoint", shape_to_string(*_shape), *ascans); }
                cuda::device(*ascans);
                _tensor->resize(*_shape);
            }

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = source_selector_t::select(block.streams(), spectra, ascans);
                _handle(plan, block, view(source));
            }

        protected:

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cpu_viewable<V>& source) {
                raise(_log, "device tensor endpoint unexpectedly received data on host");
            }

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cuda::cuda_viewable<V>& source_) {
                auto& source = source_.derived_cast();
                std::vector<size_t> block_segments, volume_segments;

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& a) {
                            std::unique_lock<std::shared_mutex> lock(_tensor->mutex());

                            cuda::device(source.device());
                            _executor->execute(_stream, view(*_tensor), source, a);
                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }

                _finish(_log, block_segments, volume_segments);
            }

            using base_t::_executor, base_t::_tensor, base_t::_stream, base_t::_shape, base_t::_log, base_t::_notify, base_t::_default, base_t::_finish;

        };

        template<typename T, typename executor_t, typename source_selector_t, typename derived_t>
        struct cuda_device_tensor_with_galvo_impl : cuda_device_tensor_impl<T, executor_t, source_selector_t, derived_t> {
            using base_t = cuda_device_tensor_impl<T, executor_t, source_selector_t, derived_t>;
            using base_t::base_t;

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = source_selector_t::select(block.streams(), spectra, ascans);
                _handle(plan, block, view(source));
            }

        protected:

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cpu_viewable<V>& source) {
                raise(_log, "device tensor endpoint unexpectedly received data on host");
            }

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cuda::cuda_viewable<V>& source_) {
                auto& source = source_.derived_cast();

                std::vector<size_t> block_segments, volume_segments;

                // ensure galvo signal device buffers are appropriately sized
                // NOTE: if size is unchanged, no allocation occurs
                cuda::device(source.device());
                _sample_actual.resize(block.sample_actual.shape());
                _sample_target.resize(block.sample_target.shape());

                // copy galvo actual onto the device
                cuda::copy(view(block.sample_actual), view(_sample_actual), &_stream);
                cuda::copy(view(block.sample_target), view(_sample_target), &_stream);

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& a) {
                            std::unique_lock<std::shared_mutex> lock(_tensor->mutex());

                            cuda::device(source.device());
                            _executor->execute(_stream, view(*_tensor), source, a, view(_sample_target), view(_sample_actual));
                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }

                _finish(_log, block_segments, volume_segments);
            }

            cuda::cuda_device_tensor_t<double> _sample_target, _sample_actual;

            using base_t::_executor, base_t::_tensor, base_t::_stream, base_t::_shape, base_t::_log, base_t::_notify, base_t::_default, base_t::_finish;

        };

    }

    template<size_t index, typename T>
    struct streams_stack_cuda_host_tensor : detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_streams_t<index>, streams_stack_cuda_host_tensor<index, T>> {
        using detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_streams_t<index>, streams_stack_cuda_host_tensor<index, T>>::cuda_host_tensor_impl;
    };
    template<typename T>
    struct spectra_stack_cuda_host_tensor : detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_spectra_t, spectra_stack_cuda_host_tensor<T>> {
        using detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_spectra_t, spectra_stack_cuda_host_tensor<T>>::cuda_host_tensor_impl;
    };
    template<typename T>
    struct ascan_stack_cuda_host_tensor : detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_ascans_t, ascan_stack_cuda_host_tensor<T>> {
        using detail::cuda_host_tensor_impl<T, stack_format_executor_t, detail::select_ascans_t, ascan_stack_cuda_host_tensor<T>>::cuda_host_tensor_impl;
    };

    template<typename T>
    struct spectra_stack_cuda_device_tensor : detail::cuda_device_tensor_impl<T, stack_format_executor_t, detail::select_spectra_t, spectra_stack_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, stack_format_executor_t, detail::select_spectra_t, spectra_stack_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };
    template<typename T>
    struct ascan_stack_cuda_device_tensor : detail::cuda_device_tensor_impl<T, stack_format_executor_t, detail::select_ascans_t, ascan_stack_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, stack_format_executor_t, detail::select_ascans_t, ascan_stack_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };

    template<typename T>
    struct spectra_radial_cuda_device_tensor : detail::cuda_device_tensor_impl<T, radial_format_executor_t, detail::select_spectra_t, spectra_radial_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, radial_format_executor_t, detail::select_spectra_t, spectra_radial_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };
    template<typename T>
    struct ascan_radial_cuda_device_tensor : detail::cuda_device_tensor_impl<T, radial_format_executor_t, detail::select_ascans_t, ascan_radial_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, radial_format_executor_t, detail::select_ascans_t, ascan_radial_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };

    template<typename T>
    struct spectra_spiral_cuda_device_tensor : detail::cuda_device_tensor_impl<T, spiral_format_executor_t, detail::select_spectra_t, spectra_radial_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, spiral_format_executor_t, detail::select_spectra_t, spectra_radial_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };
    template<typename T>
    struct ascan_spiral_cuda_device_tensor : detail::cuda_device_tensor_impl<T, spiral_format_executor_t, detail::select_ascans_t, ascan_radial_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_impl<T, spiral_format_executor_t, detail::select_ascans_t, ascan_radial_cuda_device_tensor<T>>::cuda_device_tensor_impl;
    };

    //template<typename T>
    //struct radial_galvo_cuda_device_tensor : detail::cuda_device_tensor_with_galvo_impl<T, radial_format_executor_t, radial_galvo_cuda_device_tensor<T>> {
    //    using detail::cuda_device_tensor_with_galvo_impl<T, radial_format_executor_t, radial_galvo_cuda_device_tensor<T>>::cuda_device_tensor_with_galvo_impl;
    //};

    //template<typename T>
    //struct ascan_spiral_galvo_cuda_device_tensor : detail::cuda_device_tensor_with_galvo_impl<T, spiral_format_executor_t, detail::select_ascans_t, ascan_spiral_galvo_cuda_device_tensor<T>> {
    //    using detail::cuda_device_tensor_with_galvo_impl<T, spiral_format_executor_t, detail::select_ascans_t, ascan_spiral_galvo_cuda_device_tensor<T>>::cuda_device_tensor_with_galvo_impl;
    //};
    //template<typename T>
    //struct spectra_spiral_galvo_cuda_device_tensor : detail::cuda_device_tensor_with_galvo_impl<T, spiral_format_executor_t, detail::select_spectra_t, spectra_spiral_galvo_cuda_device_tensor<T>> {
    //    using detail::cuda_device_tensor_with_galvo_impl<T, spiral_format_executor_t, detail::select_spectra_t, spectra_spiral_galvo_cuda_device_tensor<T>>::cuda_device_tensor_with_galvo_impl;
    //};

    template<typename T>
    struct spectra_position_cuda_device_tensor : detail::cuda_device_tensor_with_galvo_impl<T, position_format_executor_t, detail::select_spectra_t, spectra_position_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_with_galvo_impl<T, position_format_executor_t, detail::select_spectra_t, spectra_position_cuda_device_tensor<T>>::cuda_device_tensor_with_galvo_impl;
    };
    template<typename T>
    struct ascan_position_cuda_device_tensor : detail::cuda_device_tensor_with_galvo_impl<T, position_format_executor_t, detail::select_ascans_t, ascan_position_cuda_device_tensor<T>> {
        using detail::cuda_device_tensor_with_galvo_impl<T, position_format_executor_t, detail::select_ascans_t, ascan_position_cuda_device_tensor<T>>::cuda_device_tensor_with_galvo_impl;
    };

}
