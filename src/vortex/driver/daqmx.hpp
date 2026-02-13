#pragma once

#include <string>

#include <NIDAQmx.h>

#include <vortex/core.hpp>

namespace vortex::daqmx {

    enum class edge_t : int32 {
        rising = DAQmx_Val_Rising,
        falling = DAQmx_Val_Falling
    };

    enum class terminal_t : int32 {
        referenced = DAQmx_Val_RSE,
        unreferenced = DAQmx_Val_NRSE,
        differential = DAQmx_Val_Diff,
        pseudodifferential = DAQmx_Val_PseudoDiff
    };

    enum class sample_mode_t : int32 {
        finite = DAQmx_Val_FiniteSamps,
        continuous = DAQmx_Val_ContSamps,
        hardware = DAQmx_Val_HWTimedSinglePoint
    };

    class daqmx_t {
    public:

        //
        // task creation and destruction
        //

        daqmx_t();
        daqmx_t(std::string task_name);

        // no copying
        daqmx_t(const daqmx_t& other) = delete;
        daqmx_t& operator=(const daqmx_t& other) = delete;

        // moving allowed
        daqmx_t(daqmx_t&& other);
        daqmx_t& operator=(daqmx_t && other);

        virtual ~daqmx_t();

        //
        // channel creation
        //

        void create_digital_output(const std::string& line_name);
        void create_digital_input(const std::string& line_name);
        
        void create_analog_voltage_output(const std::string& port_name, float64 min, float64 max);
        void create_analog_voltage_input(const std::string& port_name, float64 min, float64 max, terminal_t terminal = terminal_t::referenced);
          
        //
        // clocking
        //
            
        void configure_sample_clock(const std::string& source, sample_mode_t sample_mode, size_t samples_per_second, size_t samples_per_channel, size_t divisor = 1, edge_t edge = edge_t::rising);
        
        //
        // buffer requirements
        //

        void set_output_buffer_size(size_t samples_per_channel);
        void set_input_buffer_size(size_t samples_per_channel);

        //
        // signal generation
        //

        void set_regeneration(bool enable);

        //
        // read/write
        //

        void write_analog(size_t samples_per_channel, const xt::xtensor<float64, 2>& buffer);
        void write_analog(size_t samples_per_channel, const xt::xtensor<float64, 2>& buffer, const seconds& timeout);

        void write_digital(size_t samples_per_channel, const xt::xtensor<uInt32, 2>& buffer);
        void write_digital(size_t samples_per_channel, const xt::xtensor<uInt32, 2>& buffer, const seconds& timeout);

        void read_analog(size_t samples_per_channel, xt::xtensor<float64, 2>& buffer);
        void read_analog(size_t samples_per_channel, xt::xtensor<float64, 2>& buffer, const seconds& timeout);

        void read_digital(size_t samples_per_channel, xt::xtensor<uInt32, 2>& buffer);
        void read_digital(size_t samples_per_channel, xt::xtensor<uInt32, 2>& buffer, const seconds& timeout);

        //
        // task control
        //

        void start_task();
        void stop_task();
        void clear_task();

        //
        // accessors
        //

        bool valid() const;
        const std::string& name() const;
        TaskHandle handle() const;
        
        bool running() const;

    protected:

        TaskHandle _task = nullptr;
        std::string _name;
        
        bool _started = false;

    };

    class exception : public std::runtime_error {
    public:
        using runtime_error::runtime_error;
    };
    class buffer_overflow : public exception {
    public:
        using exception::exception;
    };
    class buffer_underflow : public exception {
    public:
        using exception::exception;
    };
    class wait_timeout : public exception {
    public:
        using exception::exception;
    };
    class incomplete_operation : public exception {
    public:
        using exception::exception;
    };
    class unsupported_operation : public exception {
    public:
        using exception::exception;
    };

    std::string to_string(int32 error);
}
