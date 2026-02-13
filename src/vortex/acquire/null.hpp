#pragma once

#include <array>

#include <vortex/memory/view.hpp>

#include <vortex/util/cast.hpp>

namespace vortex::acquire {

    struct null_config_t {

        std::array<size_t, 3> shape() const { return { records_per_block(), samples_per_record(), channels_per_sample() }; };
        std::array<ptrdiff_t, 3> stride() const { return { downcast<ptrdiff_t>(samples_per_record() * channels_per_sample()), downcast<ptrdiff_t>(channels_per_sample()), 1 }; };

        size_t& channels_per_sample() { return _channels_per_sample; }
        const size_t& channels_per_sample() const { return _channels_per_sample; }
        size_t& samples_per_record() { return _samples_per_record; }
        const size_t& samples_per_record() const { return _samples_per_record; }
        size_t& records_per_block() { return _records_per_block; }
        const size_t& records_per_block() const { return _records_per_block; }

        virtual void validate() { }

    protected:

        size_t _samples_per_record = 1024;
        size_t _records_per_block = 1000;
        size_t _channels_per_sample = 1;

    };

    template<typename output_element_t_, typename config_t_>
    class null_acquisition_t {
    public:

        using config_t = config_t_;
        using output_element_t = output_element_t_;
        using callback_t = std::function<void(size_t, std::exception_ptr)>;

        void initialize(config_t config) {
            std::swap(_config, config);
        }

        const config_t& config() const {
            return _config;
        }

        void prepare() {}
        void start() {}
        void stop() {}

        template<typename V>
        size_t next(const viewable<V>& buffer) {
            return next(0, buffer);
        }
        template<typename V>
        size_t next(size_t id, const viewable<V>& buffer) {
            return buffer.derived_cast().shape(0);
        }

        template<typename V>
        void next_async(const viewable<V>& buffer, callback_t&& callback) {
            next_async(0, buffer, std::forward<callback_t>(callback));
        }
        template<typename V>
        void next_async(size_t id, const viewable<V>& buffer, callback_t&& callback) {
            std::invoke(callback, buffer.derived_cast().shape(0), std::exception_ptr{});
        }

    protected:

        config_t _config;
    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto acquisition(std::shared_ptr<vortex::acquire::null_acquisition_t<Args...>> a) {
        using adapter = adapter<block_t>;
        auto w = acquisition<block_t>(a, base_t());

        w.next_async = [a](block_t& block, typename adapter::spectra_stream_t& stream_, typename adapter::acquisition::callback_t&& callback) {
            std::visit([&](auto& stream) {
                a->next_async(block.id, view(stream).range(block.length), std::forward<typename adapter::acquisition::callback_t>(callback));
            }, stream_);
        };

        return w;
    }
}

#endif
