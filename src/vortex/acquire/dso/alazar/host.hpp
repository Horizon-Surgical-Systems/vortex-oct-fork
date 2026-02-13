/** \rst

    Alazar card acquisition component

    This file provides the configuration and component for an
    Alazar-based acquisition.

    The configuration wraps the most commonly-used elements of the Alazar
    API in an object-oriented model.  The configuration struct contains all
    information needed to configure the card and acquisition.

    The component exposes a simple API for initialization and acquisition
    of blocks.  All work is handled in a background thread.  Both
    synchronous and asynchronous (via callbacks) options are available.

 \endrst */

#pragma once

#include <vortex/acquire/dso/alazar/base.hpp>

namespace vortex::acquire {

    template<typename clock_t, typename trigger_t, typename option_t>
    struct alazar_config_t : detail::alazar_config_t<clock_t, trigger_t, option_t> {
        using base_t = detail::alazar_config_t<clock_t, trigger_t, option_t>;
        using base_t::device;

        auto create_board() {
            return alazar::board_t(device.system_index, device.board_index);
        }

        virtual void validate() {
            base_t::validate(create_board());
        }

        virtual void apply(alazar::board_t& board, std::shared_ptr<spdlog::logger>& log) {
            base_t::apply(board, log);
        }
    };

    template<typename config_t>
    class alazar_acquisition_t : public detail::alazar_acquisition_t<config_t, alazar::board_t> {
    public:

        using base_t = detail::alazar_acquisition_t<config_t, alazar::board_t>;
        using base_t::base_t;
        using callback_t = typename base_t::callback_t;

        template<typename V>
        size_t next(const cpu_viewable<V>& buffer) {
            return next(0, buffer);
        }
        template<typename V>
        size_t next(size_t id, const cpu_viewable<V>& buffer) {
            return _next(id, buffer);
        }

        template<typename V>
        void next_async(const cpu_viewable<V>& buffer, callback_t&& callback) {
            return next_async(0, buffer, std::forward<callback_t>(callback));
        }
        template<typename V>
        void next_async(size_t id, const cpu_viewable<V>& buffer, callback_t&& callback) {
            return _next_async(id, buffer, std::forward<callback_t>(callback));
        }

    protected:

        using base_t::_next, base_t::_next_async;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto acquisition(std::shared_ptr<vortex::acquire::alazar_acquisition_t<vortex::acquire::alazar_config_t<Args...>>> a) {
        using adapter = adapter<block_t>;
        auto w = acquisition<block_t>(a, base_t());

        w.stream_factory = []() {
            return []() -> typename adapter::spectra_stream_t {
                return sync::lockable<cuda::cuda_host_tensor_t<typename block_t::acquire_element_t>>();
            };
        };

        w.next_async = [a](block_t& block, typename adapter::spectra_stream_t& stream_, typename adapter::acquisition::callback_t&& callback) {
            std::visit([&](auto& stream) {
                try {
                    view_as_cpu([&](auto buffer) {
                        a->next_async(block.id, buffer.range(block.length), std::forward<typename adapter::acquisition::callback_t>(callback));
                    }, stream);
                } catch (const unsupported_view&) {
                    callback(0, std::current_exception());
                }
            }, stream_);
        };

        return w;
    }
}

#endif
