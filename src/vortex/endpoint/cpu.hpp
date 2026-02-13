#pragma once

#include <vortex/endpoint/common.hpp>

#include <vortex/format.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/sync.hpp>

namespace vortex::endpoint {

    namespace detail {

        template<typename tensor_t, typename executor_t, typename source_selector_t>
        struct cpu_tensor : detail::notify {
            using base_t = detail::notify;

            cpu_tensor(std::shared_ptr<executor_t> executor, std::vector<size_t> shape, std::shared_ptr<spdlog::logger> log = nullptr)
                : _executor(std::move(executor)), _shape(std::move(shape)), _log(std::move(log)) {
                _tensor = std::make_shared<sync::lockable<tensor_t>>();
                _check();
            }

            template<size_t N>
            cpu_tensor(std::shared_ptr<executor_t> executor, const std::array<size_t, N>& shape, std::shared_ptr<spdlog::logger> log = nullptr)
                : _executor(std::move(executor)), _log(std::move(log)) {
                _shape.emplace(shape.begin(), shape.end());
                _tensor = std::make_shared<sync::lockable<tensor_t>>();
                _check();
            }

            cpu_tensor(std::shared_ptr<executor_t> executor, std::shared_ptr<sync::lockable<tensor_t>> tensor, std::shared_ptr<spdlog::logger> log = nullptr)
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

            void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {           
                if (!_shape) {
                    // no automatic allocation
                    return;
                }

                // check if A-scans will arrive in host memory
                const auto& device = source_selector_t::select(std::optional<cuda::device_t>{}, spectra, ascans);
                if (device) {
                    throw std::runtime_error("A-scans must arrive in host memory for CPU tensor endpoints");
                }

                // allocate
                if (_log) { _log->debug("allocating [{}] buffer on host for endpoint", shape_to_string(*_shape)); }
                _tensor->resize(*_shape);
            }

            template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
            void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                const auto& source = source_selector_t::select(block.streams(), spectra, ascans);
                _handle(plan, block, view(source));
            }

            using base_t::aggregate_segment_callback, base_t::update_callback;

        protected:

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cuda::cuda_viewable<V>& source) {
                raise(_log, "device tensor endpoint unexpectedly received data on device");
            }

            template<typename block_t, typename V>
            void _handle(const format::format_plan_t& plan, const block_t& block, const cpu_viewable<V>& source_) {
                auto& source = source_.derived_cast();

                std::vector<size_t> block_segments, volume_segments;

                // process actions
                for (auto& action : plan) {
                    std::visit(overloaded{
                        [&](const format::action::copy& a) {
                            std::unique_lock<std::shared_mutex> lock(_tensor->mutex());

                            _executor->execute(view(*_tensor), source, a);
                        },
                        [&](const auto& a) { _default(_log, block_segments, volume_segments, a); }
                    }, action);
                }

                _finish(_log, block_segments, volume_segments);
            }

            std::shared_ptr<executor_t> _executor;

            std::optional<std::vector<size_t>> _shape;
            std::shared_ptr<sync::lockable<tensor_t>> _tensor;

            std::shared_ptr<spdlog::logger> _log;

            using base_t::_notify, base_t::_default;

        private:

            void _check() {
                if (!_executor) {
                    throw std::invalid_argument("non-null executor required");
                }
            }

        };

    }

    template<size_t index, typename T>
    struct streams_stack_cpu_tensor : detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_streams_t<index>> {
        using detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_streams_t<index>>::cpu_tensor;
    };
    template<typename T>
    struct spectra_stack_cpu_tensor : detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_spectra_t> {
        using detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_spectra_t>::cpu_tensor;
    };
    template<typename T>
    struct ascan_stack_cpu_tensor : detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_ascans_t> {
        using detail::cpu_tensor<cpu_tensor_t<T>, stack_format_executor_t, detail::select_ascans_t>::cpu_tensor;
    };

}
