#include <vortex/driver/motion.hpp>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>

#include <Reflexxes/ReflexxesAPI.h>
#include <Reflexxes/RMLPositionFlags.h>
#include <Reflexxes/RMLPositionInputParameters.h>
#include <Reflexxes/RMLPositionOutputParameters.h>

#include <fmt/format.h>

#include <vortex/util/cast.hpp>

using namespace vortex::motion;

xt::xtensor<double, 2> vortex::motion::plan(size_t dimension, double dt, const state_t<const double*>& start, const state_t<const double*>& end, const limits_t<double>* limits, const options_t& options) {
    auto dims = downcast<unsigned int>(dimension);
    ReflexxesAPI reflexxes(dims, dt);

    RMLPositionInputParameters inputs(dims);
    RMLPositionOutputParameters outputs(dims);
    
    // PHASE_SYNCHRONIZATION_IF_POSSIBLE will ensure time synchronization is performed
    RMLPositionFlags flags;
    flags.SynchronizationBehavior = RMLFlags::PHASE_SYNCHRONIZATION_IF_POSSIBLE;
    flags.BehaviorAfterFinalStateOfMotionIsReached = RMLPositionFlags::KEEP_TARGET_VELOCITY;
    flags.EnableTheCalculationOfTheExtremumMotionStates = false;

    // use all axes
    inputs.SelectionVector->Set(true);

    // set minimum synchronization time to fixed duration
    size_t target_samples = 0;
    if (options.fixed_samples) {
        // NOTE: increase by 1 to account for the final sample
        target_samples = *options.fixed_samples + 1;
    }
    if(target_samples > 0) {
        inputs.SetMinimumSynchronizationTime(target_samples * dt);
    }

    // copy in endpoints
    inputs.SetCurrentPositionVector(start.position);
    inputs.SetCurrentVelocityVector(start.velocity);

    inputs.SetTargetPositionVector(end.position);
    inputs.SetTargetVelocityVector(end.velocity);   

    // set derivative constraints
    for (size_t i = 0; i < dimension; i++) {
        inputs.MaxVelocityVector->VecData[i] = limits[i].velocity;
        inputs.MaxAccelerationVector->VecData[i] = limits[i].acceleration;
    }

    // solve the trajectory
    auto result = reflexxes.RMLPosition(inputs, &outputs, flags);
    if (result < 0) {
        throw std::runtime_error(fmt::format("motion plan ({},{}) @ ({},{}) -> ({},{}) @ ({},{}) failed: {}", start.position[0], start.position[1], start.velocity[0], start.velocity[1], end.position[0], end.position[1], end.velocity[0], end.velocity[1], result));
    }

    size_t initial = 0;
    if (!options.include_initial) {
        // skip the first sample
        initial++;
    }

    auto count = size_t(std::ceil(outputs.SynchronizationTime / dt));
    // enforce fixed duration
    if (target_samples > 0) {
        // NOTE: tolerate an off-by-one rounding error
        if(!(count == target_samples || count == target_samples + 1)) {
            throw std::runtime_error(fmt::format("motion plan ({},{}) @ ({},{}) -> ({},{}) @ ({},{}) did not meet fixed duration: |{} - {}| > 1", start.position[0], start.position[1], start.velocity[0], start.velocity[1], end.position[0], end.position[1], end.velocity[0], end.velocity[1], count, target_samples));
        }
        count = target_samples;
    }
    
    // time scaling to ensure the sample at dt * count falls exactly on the final position
    auto time_scale = outputs.SynchronizationTime / (dt * count);

    if (options.include_final) {
        // add a sample for the endpoint since indexing from zero
        count++;
    }

    // allocate storage
    xt::xtensor<double, 2> path({ count - initial, dimension });
    if (path.size() == 0) {
        return {};
    }
    
    // create an adaptor to integrate nicely with xtensor
    std::array<size_t, 1> shape = { { dimension } };
    auto point = xt::adapt(outputs.NewPositionVector->VecData, dimension, xt::no_ownership(), shape);

    // read out the trajectory
    for (size_t idx = initial; idx < count; idx++) {
        auto t = (idx * dt) * time_scale;
        result = reflexxes.RMLPositionAtAGivenSampleTime(t, &outputs);
        if (result < 0) {
            throw std::runtime_error(fmt::format("motion plan ({},{}) @ ({},{}) -> ({},{}) @ ({},{}) failed at t={}: {}", start.position[0], start.position[1], start.velocity[0], start.velocity[1], end.position[0], end.position[1], end.velocity[0], end.velocity[1], t, result));
        }

        xt::view(path, idx - initial, xt::all()) = point;
    }

    if (!options.bypass_limits_check) {
        // enforce position limits
        // TODO: fix buffer overrun warning here
        auto min = xt::amin(path, 0, xt::evaluation_strategy::immediate);
        auto max = xt::amax(path, 0, xt::evaluation_strategy::immediate);
        for (size_t d = 0; d < dimension; d++) {
            if (min(d) < limits[d].position.min()) {
                throw std::runtime_error(fmt::format("axis {} violated lower position limit during motion planning: {} < {}", d, min(d), limits[d].position.min()));
            }
            if (max(d) > limits[d].position.max()) {
                throw std::runtime_error(fmt::format("axis {} violated upper position limit during motion planning: {} > {}", d, max(d), limits[d].position.max()));
            }
        }
    }

    return path;
}