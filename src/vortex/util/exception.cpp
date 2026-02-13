#include <vortex/util/exception.hpp>

#include <fmt/format.h>

using namespace vortex;

tracer::tracer() {
#if defined(VORTEX_ENABLE_BACKWARD)
        _st.load_here();
#endif
}

std::string tracer::format() const {

#if defined(VORTEX_ENABLE_BACKWARD)
    if (_st.size() > 0) {
        std::string buffer;

        backward::TraceResolver tr;
        tr.load_stacktrace(_st);

        // NOTE: skip call of this function
        for (size_t i = 0; i < _st.size(); i++) {
            auto trace = tr.resolve(_st[i]);
            fmt::format_to(std::back_inserter(buffer), "#{} {} {} [{}]\n", i, trace.object_filename, trace.object_function, trace.addr);
        }

        return buffer;
    }
#endif

    return "--- stack trace empty ---";
}
