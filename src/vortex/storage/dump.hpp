#pragma once

#include <fstream>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/storage/detail/raw.hpp>

namespace vortex::storage {

    struct stream_dump_config_t {
        std::string path;

        size_t stream = 0;
        size_t divisor = 1;

        bool buffering = false;
    };

    namespace dump {

        template<typename config_t_>
        class stream_dump_v0_t {
        public:

            using config_t = config_t_;

            stream_dump_v0_t(std::shared_ptr<spdlog::logger> log = nullptr)
                : _log(std::move(log)) {

            }

            virtual ~stream_dump_v0_t() {
                close();
            }

            const auto& config() const {
                return _config;
            }

            virtual void open(config_t config) {
                // close any previously open file
                close();

                // accept configuration
                std::swap(_config, config);

                // open new file
                if (_log) { _log->debug("opening file \"{}\"", _config.path); }
                if (!_config.buffering) {
                    // NOTE: disable buffering before the file is open
                    _out.rdbuf()->pubsetbuf(nullptr, 0);
                }
                _out.exceptions(~std::ios::goodbit);
                _out.open(_config.path, std::ios::binary);
            }

            template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
            void next(size_t id, const std::tuple<Vs...>& streams) {
                if (_log) { _log->trace("dumping block {}", id); }

                runtime_get(streams, _config.stream, [&](auto& stream) {
                    _dump(stream);
                });
            }

            void close() {
                if (_out.is_open()) {
                    if (_log) { _log->debug("closing file \"{}\"", _config.path); }
                    _out.close();
                }
            }

            auto ready() const {
                return _out.is_open();
            }

        protected:

            template<typename V>
            void _dump(const cpu_viewable<V>& buffer_) {
                using T = typename V::element_t;

                auto do_write = [&](auto&& chunk) {
                    detail::write_raw(_out, &(*chunk.begin()), chunk.size());
                };

                auto do_consolidate = [&](auto&& chunk) {
                    if (chunk.is_contiguous()) {
                        // write out directly
                        do_write(chunk);
                    } else {
                        // reorganize in memory
                        xt::xtensor<T, 2> chunk_contiguous = chunk;
                        do_write(chunk_contiguous);
                    }
                };

                auto do_downsampling = [&](auto&& chunk) {
                    if (_config.divisor > 1) {
                        // downsampling
                        do_consolidate(xt::view(chunk, xt::range(0, chunk.shape(0), _config.divisor), xt::all()));
                    } else {
                        // passthrough
                        do_consolidate(chunk);
                    }

                };

                auto do_shaping = [&](auto&& chunk) {
                    if (chunk.dimension() > 2) {
                        // flatten along dimensions after the first
                        do_downsampling(xt::reshape_view(chunk, { chunk.shape(0), chunk.size() / chunk.shape(0) }));
                    } else if (chunk.dimension() == 1) {
                        // promote to 2D
                        do_downsampling(xt::view(chunk, xt::all(), xt::newaxis()));
                    } else {
                        // already correct shape
                        do_downsampling(chunk);
                    }
                };

                // execute processing
                auto buffer = buffer_.derived_cast().to_xt();
                do_shaping(buffer);
            }

            std::shared_ptr<spdlog::logger> _log;

            config_t _config;

            std::ofstream _out;

        };

    }

    template<typename config_t>
    using stream_dump_t = dump::stream_dump_v0_t<config_t>;

}