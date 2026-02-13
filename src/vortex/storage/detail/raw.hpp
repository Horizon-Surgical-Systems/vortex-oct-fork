#pragma once

#include <iostream>

namespace vortex::storage::detail {

    template<typename T>
    inline void write_raw(std::ostream& s, const T& o) {
        s.write(reinterpret_cast<const char*>(&o), sizeof(T));
    }
    template<>
    inline void write_raw(std::ostream& s, const std::string& o) {
        s.write(o.data(), o.size());
    }
    template<typename T>
    inline void write_through_raw(std::ostream& s, const T& o) {
        s.rdbuf()->sputn(reinterpret_cast<const char*>(&o), sizeof(T));
    }
    template<typename T>
    inline void read_raw(std::istream& s, T& o) {
        s.read(reinterpret_cast<char*>(&o), sizeof(T));
    }

    template<typename T>
    inline void write_raw(std::ostream& s, const T* o, size_t count) {
        s.write(reinterpret_cast<const char*>(o), count * sizeof(T));
    }
    template<typename T>
    inline void write_through_raw(std::ostream& s, const T* o, size_t count) {
        s.rdbuf()->sputn(reinterpret_cast<const char*>(o), count * sizeof(T));
    }
    template<typename T>
    inline void read_raw(std::istream& s, T* o, size_t count) {
        s.read(reinterpret_cast<char*>(o), count * sizeof(T));
    }

}