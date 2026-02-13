#pragma once

#include <algorithm>

#include <vortex/endpoint/common.hpp>

#include <vortex/memory/cuda.hpp>

#include <vortex/storage.hpp>

namespace vortex::endpoint {

    struct stream_dump_storage {

        stream_dump_storage(std::shared_ptr<stream_dump_t> storage, size_t lead_samples = 0)
            : _lead_samples(lead_samples), _storage(std::move(storage)) {
            if (!_storage) {
                throw std::invalid_argument("non-null storage required");
            }
        }

        void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
            // no buffers required
        }

        template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
        void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
            if(!_storage->ready()) {
                return;
            }

            view_tuple_as_cpu([&](auto buffers) {
                _storage->next(block.id, buffers);
            }, block.streams(_lead_samples));
        }

        const auto& storage() const {
            return _storage;
        }

    protected:

        size_t _lead_samples;
        std::shared_ptr<stream_dump_t> _storage;
    };

    struct marker_log_storage {

        marker_log_storage(std::shared_ptr<marker_log_t> storage)
            : _storage(std::move(storage)) {
            if (!_storage) {
                throw std::invalid_argument("non-null storage required");
            }
        }

        void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
            // no buffers required
        }

        template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
        void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
            if (!_storage->ready()) {
                return;
            }

            _storage->write(block.markers);
        }

        const auto& storage() const {
            return _storage;
        }

    protected:

        std::shared_ptr<marker_log_t> _storage;
    };

    enum class buffer_strategy_t {
        none,
        segment,
        volume
    };

    namespace detail {

        struct default_volume_shape_t {
            template<typename config_t>
            static auto shape(const config_t& config) {
                return config.volume_shape();
            }
        };
        struct broct_volume_shape_t {
            template<typename config_t>
            static auto shape(const config_t& config) {
                return config.broct_volume_shape();
            }
        };

        template<typename storage_t, typename source_selector_t>
        struct stream_storage : detail::notify {
            stream_storage(std::shared_ptr<storage_t> storage, std::shared_ptr<spdlog::logger> log)
                : _storage(std::move(storage)), _log(std::move(log)) {
                if (!_storage) {
                    throw std::invalid_argument("non-null storage required");
                }
            }

            void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
                // no buffers required
            }

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = view(source_selector_t::select(block.streams(), spectra, ascans));
                std::vector<size_t> block_segments, volume_segments;

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& action) {
                            if (!_storage->ready()) {
                                return;
                            }

                            overloaded{
                                [&] <typename V>(const cuda::cuda_viewable<V>& buffer_) {
                                    // select the indicated range
                                    const auto& buffer = buffer_.derived_cast().range(action.block_offset, action.block_offset + action.count);

                                    // upsize transfer buffer if needed
                                    _host_buffer.resize(buffer.shape(), false);

                                    // copy to host
                                    cuda::copy(buffer, view(_host_buffer), &_stream);
                                    _stream.sync();

                                    // write out
                                    _storage->write(view(_host_buffer));
                                },
                                [&] <typename V>(const cpu_viewable<V>& buffer_) {
                                    const auto& buffer = buffer_.derived_cast();

                                    // select the indicated range and write directly to storage
                                    auto data = buffer.range(action.block_offset, action.block_offset + action.count);
                                    _storage->write(data);
                                }
                            }(source);

                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }
                
                _finish(_log, block_segments, volume_segments);
            }

        public:

            const auto& storage() const {
                return _storage;
            }

        protected:

            std::shared_ptr<storage_t> _storage;

            std::shared_ptr<spdlog::logger> _log;

            cuda::stream_t _stream;
            cuda::cuda_host_tensor_t<typename storage_t::element_t> _host_buffer;

        };

        template<typename executor_t, typename storage_t, typename source_selector_t, typename volume_shape_t = default_volume_shape_t>
        struct stack_storage : detail::notify {

            // initialize in direct mode
            stack_storage(std::shared_ptr<storage_t> storage, std::shared_ptr<spdlog::logger> log)
                : _storage(std::move(storage)), _buffer_strategy(buffer_strategy_t::none), _log(std::move(log)) {
                if (!_storage) {
                    throw std::invalid_argument("non-null storage required");
                }
            }

            // initialize in executor mode
            stack_storage(std::shared_ptr<executor_t> executor, std::shared_ptr<storage_t> storage, std::shared_ptr<spdlog::logger> log)
                : stack_storage(std::move(executor), std::move(storage), buffer_strategy_t::volume, std::move(log)) { }
            stack_storage(std::shared_ptr<executor_t> executor, std::shared_ptr<storage_t> storage, buffer_strategy_t buffer_strategy, std::shared_ptr<spdlog::logger> log)
                : _executor(std::move(executor)), _storage(std::move(storage)), _buffer_strategy(buffer_strategy), _log(std::move(log)) {
                if (!_executor) {
                    throw std::invalid_argument("non-null executor required");
                }
                if (!_storage) {
                    throw std::invalid_argument("non-null storage required");
                }
            }

            void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {
                const auto& device = source_selector_t::select(std::optional<cuda::device_t>{}, spectra, ascans);

                if (_executor) {

                    auto shape = _buffer_shape(volume_shape_t::shape(_storage->config()));

                    // check if data will arrive in device memory
                    if (device) {

                        // needs a device volume for formatting
                        if (_log) { _log->debug("allocating [{}] buffer on device {} for formatting", shape_to_string(shape), *device); }
                        cuda::device(*device);
                        _device_buffer.resize(shape);

                    }

                    // needs a host volume for transferring
                    if (_log) { _log->debug("allocating [{}] buffer on host for formatting", shape_to_string(shape)); }
                    _host_buffer.resize(shape);

                } else {

                    if (device) {
                        raise(_log, "direct mode requires data to arrive in host memory");
                    }

                    // no buffers needed in direct mode
                }
            }

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = view(source_selector_t::select(block.streams(), spectra, ascans));

                std::vector<size_t> block_segments, volume_segments;

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& action) {
                            if (!_storage->ready()) {
                                return;
                            }

                            if (_executor) {
                                // executor mode

                                auto shape = _buffer_shape(volume_shape_t::shape(_storage->config()), block);

                                // remap the copy action based on the buffer strategy
                                auto remap_action = action;
                                if (_buffer_strategy == buffer_strategy_t::volume) {
                                    // perform normal copy action on a volume-sized buffer
                                } else if (_buffer_strategy == buffer_strategy_t::segment) {
                                    // remap copy action to the single segment in the buffer
                                    remap_action.buffer_segment = 0;
                                } else if (_buffer_strategy == buffer_strategy_t::none) {
                                    // remap copy action to the start of the block-sized buffer
                                    remap_action.buffer_segment = 0;
                                    remap_action.buffer_record = 0;
                                }

                                overloaded{
                                    [&] <typename V>(const cuda::cuda_viewable<V>& buffer) {
                                        // upsize buffers if needed
                                        _device_buffer.resize(shape, false);

                                        // format on device
                                        cuda::device(buffer.derived_cast().device());
                                        _executor->execute(_stream, view(_device_buffer), buffer, remap_action);
                                    },
                                    [&] <typename V>(const cpu_viewable<V>& buffer) {
                                        // upsize buffers if needed
                                         _host_buffer.resize(shape, false);

                                         // format on host
                                         _executor->execute(view(_host_buffer), buffer, remap_action);
                                    }
                                }(source);

                                if (_buffer_strategy == buffer_strategy_t::none) {
                                    // transfer the block now
                                    _transfer(action);
                                }

                            } else {
                                // direct mode

                                overloaded{
                                    [&] <typename V>(const cpu_viewable<V>& buffer) {
                                        // write directly to storage
                                        auto data = buffer.derived_cast().range(action.block_offset, action.block_offset + action.count);
                                        _storage->write_partial_bscan(action.buffer_segment, action.buffer_record, data);
                                    },
                                    [&] <typename V>(const cuda::cuda_viewable<V>&) {
                                        raise(_log, "direct mode requires data to arrive in host memory");
                                    }
                                }(source);

                            }
                        },
                        [&](const format::action::finish_segment& a) {
                            if (_buffer_strategy == buffer_strategy_t::segment) {
                                // transfer the only segment now
                                _transfer(std::vector<size_t>{ a.segment_index_buffer });
                            }

                            // remainder of usual processing
                            _default(_log, block_segments, volume_segments, a);
                        },
                        [&](const format::action::finish_volume& a) {
                            if (_buffer_strategy == buffer_strategy_t::volume) {
                                // transfer all available segments now
                                _transfer(volume_segments);
                            }
                            if (_storage->ready()) {
                                _storage->advance_volume(false);
                            }

                            // remainder of usual processing
                            _default(_log, block_segments, volume_segments, a);
                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }

                if (_buffer_strategy == buffer_strategy_t::volume) {
                    // handle any remaining segments
                    _transfer(volume_segments);
                }
                _finish(_log, block_segments, volume_segments);
            }

            const auto& executor() const {
                return _executor;
            }
            const auto& storage() const {
                return _storage;
            }
            const auto& stream() const {
                return _stream;
            }

        protected:

            template<typename shape_t>
            auto _buffer_shape(const shape_t& in) {
                std::vector<size_t> out(in.begin(), in.end());

                if (_buffer_strategy == buffer_strategy_t::volume) {
                    // buffer size is volume size
                } else if (_buffer_strategy == buffer_strategy_t::segment) {
                    // buffer size is one segment
                    out[0] = 1;
                } else if (_buffer_strategy == buffer_strategy_t::none) {
                    // buffer size is one record and will be resized to block length later on
                    out[0] = 1;
                    out[1] = 1;
                }

                return out;
            }
            template<typename shape_t, typename block_t>
            auto _buffer_shape(const shape_t& in, const block_t& block) {
                std::vector<size_t> out(in.begin(), in.end());

                if (_buffer_strategy == buffer_strategy_t::volume) {
                    // buffer size is volume size
                } else if (_buffer_strategy == buffer_strategy_t::segment) {
                    // buffer size is one segment
                    out[0] = 1;
                } else if(_buffer_strategy == buffer_strategy_t::none) {
                    // buffer size is length of the block
                    out[0] = 1;
                    out[1] = block.length;
                }

                return out;
            }

            void _transfer(std::vector<size_t> segments) {
                if (segments.empty() || !_storage->ready()) {
                    return;
                }

                // upsize buffers if needed
                auto shape = _buffer_shape(volume_shape_t::shape(_storage->config()));
                _host_buffer.resize(shape, false);

                // remove invalid segments (due to interim resize) and build ranges
                segments.erase(std::remove_if(segments.begin(), segments.end(), [&](auto& s) { return s >= _storage->config().bscans_per_volume(); }), segments.end());
                std::sort(segments.begin(), segments.end());
                auto chunks = detail::combine(segments);

                // no copy needed if device buffer is empty
                if (_device_buffer.valid()) {
                    // upsize buffers if needed
                    _device_buffer.resize(shape, false);

                    // copy the data to the host for saving
                    for (auto chunk : chunks) {
                        if (_buffer_strategy == buffer_strategy_t::segment) {
                            // remap to single segment in buffer
                            chunk.min() = 0;
                            chunk.max() = 1;
                        }

                        auto device_view = view(_device_buffer);
                        auto host_view = view(_host_buffer);
                        vortex::cuda::copy(
                            device_view, device_view.offset({ chunk.min() }),
                            host_view, host_view.offset({ chunk.min() }),
                            device_view.offset({ chunk.max() }) - device_view.offset({ chunk.min() }),
                            &_stream
                        );
                    }
                }

                // wait for copies to complete
                // NOTE: sync even when formatting to host because the executor dispatches asynchronous copies
                _stream.sync();

                // write to disk
                for (auto& dst_chunk : chunks) {
                    auto src_chunk = dst_chunk;
                    if (_buffer_strategy == buffer_strategy_t::segment) {
                        // remap to single segment in buffer
                        src_chunk.min() = 0;
                        src_chunk.max() = 1;
                    }

                    auto data = view(_host_buffer).range(src_chunk.min(), src_chunk.max());
                    _storage->write_multi_bscan(dst_chunk.min(), data);
                }
            }

            void _transfer(const format::action::copy& a) {
                if (!_storage->ready()) {
                    return;
                }

                // no copy needed if device buffer is empty
                if (_device_buffer.valid()) {
                    // copy the data to the host for saving
                    cuda::copy(view(_device_buffer), view(_host_buffer), a.count, &_stream);
                }

                // wait for copies to complete
                // NOTE: sync even when formatting to host because the executor dispatches asynchronous copies
                _stream.sync();

                // write to disk
                auto data = view(_host_buffer).index({ 0 }).range(a.count);
                _storage->write_partial_bscan(a.buffer_segment, a.buffer_record, data);
            }

            std::shared_ptr<executor_t> _executor;
            std::shared_ptr<storage_t> _storage;

            buffer_strategy_t _buffer_strategy;

            std::shared_ptr<spdlog::logger> _log;

            cuda::stream_t _stream;
            cuda::cuda_device_tensor_t<typename storage_t::element_t> _device_buffer;
            cuda::cuda_host_tensor_t<typename storage_t::element_t> _host_buffer;
        };

    }

    template<size_t index, typename element_t>
    struct streams_stream_storage : detail::stream_storage<simple_stream_t<element_t>, detail::select_streams_t<index>> {
        using detail::stream_storage<simple_stream_t<element_t>, detail::select_streams_t<index>>::stream_storage;
    };
    template<typename element_t>
    struct spectra_stream_storage : detail::stream_storage<simple_stream_t<element_t>, detail::select_spectra_t> {
        using detail::stream_storage<simple_stream_t<element_t>, detail::select_spectra_t>::stream_storage;
    };
    template<typename element_t>
    struct ascan_stream_storage : detail::stream_storage<simple_stream_t<element_t>, detail::select_ascans_t> {
        using detail::stream_storage<simple_stream_t<element_t>, detail::select_ascans_t>::stream_storage;
    };

    template<size_t index, typename element_t>
    struct streams_stack_storage : detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_streams_t<index>> {
        using detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_streams_t<index>>::stack_storage;
    };
    template<typename element_t>
    struct spectra_stack_storage : detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_spectra_t> {
        using detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_spectra_t>::stack_storage;
    };
    template<typename element_t>
    struct ascan_stack_storage : detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_ascans_t> {
        using detail::stack_storage<stack_format_executor_t, simple_stack_t<element_t>, detail::select_ascans_t>::stack_storage;
    };

#if defined(VORTEX_ENABLE_HDF5)
    template<size_t index, typename element_t>
    struct streams_hdf5_stack_storage : detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_streams_t<index>> {
        using detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_streams_t<index>>::stack_storage;
    };
    template<typename element_t>
    struct spectra_hdf5_stack_storage : detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_spectra_t> {
        using detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_spectra_t>::stack_storage;
    };
    template<typename element_t>
    struct ascan_hdf5_stack_storage : detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_ascans_t> {
        using detail::stack_storage<stack_format_executor_t, hdf5_stack_t<element_t>, detail::select_ascans_t>::stack_storage;
    };
#endif

    struct broct_storage : detail::stack_storage<broct_format_executor_t, broct_storage_t, detail::select_ascans_t, detail::broct_volume_shape_t> {
        using detail::stack_storage<broct_format_executor_t, broct_storage_t, detail::select_ascans_t, detail::broct_volume_shape_t>::stack_storage;
    };

}
