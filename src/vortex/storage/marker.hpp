#pragma once

#include <fstream>
#include <vector>

#include <fmt/ostream.h>

#include <vortex/marker/marker.hpp>

#include <vortex/storage/detail/raw.hpp>

#include <vortex/util/variant.hpp>

namespace vortex::storage {

    struct marker_log_config_t {
        std::string path;

        bool binary = false;

        bool buffering = false;
    };

    namespace marker {

        template<typename config_t_>
        class marker_v0_t {
        public:

            using config_t = config_t_;

            marker_v0_t(std::shared_ptr<spdlog::logger> log = nullptr)
                : _log(std::move(log)) {

            }

            virtual ~marker_v0_t() {
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

            template<typename marker_t>
            void write(const std::vector<marker_t>& markers) {
                if (markers.size() == 0) {
                    return;
                }
                if (_log) { _log->trace("writing {} markers", markers.size()); }

                for (auto& marker : markers) {
                    if (_config.binary) {
                        _write_binary(_out, marker);
                    } else {
                        _write_text(_out, marker);
                    }
                }
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

            template<typename marker_t>
            void _write_text(std::ostream& out, const marker_t& marker) {
                std::visit(vortex::overloaded{
                    [&](const vortex::marker::scan_boundary& m) { fmt::print(out, "{:12d} C {} 0 - {} {}\n", m.sample, m.sequence, m.volume_count_hint, m.flags.value); },
                    [&](const vortex::marker::volume_boundary& m) { fmt::print(out, "{:12d} V {} {} {} {} {}\n", m.sample, m.sequence, m.index_in_scan, m.reversed ? "R" : "F", m.segment_count_hint, m.flags.value); },
                    [&](const vortex::marker::segment_boundary& m) { fmt::print(out, "{:12d} S {} {} {} {} {}\n", m.sample, m.sequence, m.index_in_volume, m.reversed ? "R" : "F", m.record_count_hint, m.flags.value); },
                    [&](const vortex::marker::active_lines& m) { fmt::print(out, "{:12d} A 0 0 - 0 0\n", m.sample); },
                    [&](const vortex::marker::inactive_lines& m) { fmt::print(out, "{:12d} I 0 0 - 0 0\n", m.sample); },
                    [&](const vortex::marker::event& m) { fmt::print(out, "{:12d} E {} 0 - 0 0\n", m.sample, m.id); },
                    [&](const auto& m) { fmt::print(out, "{:12d} ? 0 0 X 0 0\n", m.sample); }
                }, marker);
            }

            template<typename marker_t>
            void _write_binary(std::ostream& out, const marker_t& marker) {
                // sample
                std::visit([&](const auto& m) { detail::write_raw(out, m.sample); }, marker);
                // type
                std::visit(vortex::overloaded{
                    [&](const vortex::marker::scan_boundary& m) { out << "C"; },
                    [&](const vortex::marker::volume_boundary& m) { out << "V"; },
                    [&](const vortex::marker::segment_boundary& m) { out << "S"; },
                    [&](const vortex::marker::active_lines& m) { out << "A"; },
                    [&](const vortex::marker::inactive_lines& m) { out << "I"; },
                    [&](const vortex::marker::event& m) { out << "E"; },
                    [&](const auto& m) { out << "?"; }
                }, marker);
                // marker-specific fields
                std::visit(vortex::overloaded{
                    [&](const vortex::marker::scan_boundary& m) {
                        detail::write_raw(out, m.sequence);
                        detail::write_raw(out, 0LL);
                        out.put('-');
                        detail::write_raw(out, m.volume_count_hint);
                        detail::write_raw(out, m.flags.value);
                    },
                    [&](const vortex::marker::volume_boundary& m) {
                        detail::write_raw(out, m.sequence);
                        detail::write_raw(out, m.index_in_scan);
                        out.put(m.reversed ? 'R' : 'F');
                        detail::write_raw(out, m.segment_count_hint);
                        detail::write_raw(out, m.flags.value);
                    },
                    [&](const vortex::marker::segment_boundary& m) {
                        detail::write_raw(out, m.sequence);
                        detail::write_raw(out, m.index_in_volume);
                        out.put(m.reversed ? 'R' : 'F');
                        detail::write_raw(out, m.record_count_hint);
                        detail::write_raw(out, m.flags.value);
                    },
                    [&](const vortex::marker::event& m) {
                        detail::write_raw(out, m.id);
                        detail::write_raw(out, 0LL);
                        out.put('-');
                        detail::write_raw(out, 0LL);
                        detail::write_raw(out, 0LL);
                    },
                    [&](const auto& m) {
                        // placeholders
                        detail::write_raw(out, 0LL);
                        detail::write_raw(out, 0LL);
                        out.put('-');
                        detail::write_raw(out, 0LL);
                        detail::write_raw(out, 0LL);
                    }
                }, marker);
            }

            std::shared_ptr<spdlog::logger> _log;

            config_t _config;

            std::ofstream _out;

        };

    }

    template<typename config_t>
    using marker_log_t = marker::marker_v0_t<config_t>;

}