#ifndef MEDIAPIPE_FRAMEWORK_OUTPUT_STREAM_POLLER_H_
#define MEDIAPIPE_FRAMEWORK_OUTPUT_STREAM_POLLER_H_

#include <memory>

#include "mediapipe/framework/graph_output_stream.h"

namespace mediapipe
{
// The public interface of output stream poller.
class OutputStreamPoller
{
public:
    OutputStreamPoller(const OutputStreamPoller&) = delete;
    OutputStreamPoller& operator=(const OutputStreamPoller&) = delete;
    OutputStreamPoller(OutputStreamPoller&&) = default;
    // Move assignment needs to be explicitly defaulted to allow
    // ASSIGN_OR_RETURN on `StatusOr<OutputStreamPoller>`.
    OutputStreamPoller& operator=(OutputStreamPoller&&) = default;

    // Resets OutputStramPollerImpl and cleans the internal packet queue.
    void Reset()
    {
        auto poller = internal_poller_impl_.lock();
        CHECK(poller) << "OutputStreamPollerImpl is already destroyed.";
        poller->Reset();
    }

    // Gets the next packet (block until it is available or the stream is
    // done).  Returns true if successful.
    ABSL_MUST_USE_RESULT bool Next(Packet* packet)
    {
        auto poller = internal_poller_impl_.lock();
        if (!poller)
        {
            return false;
        }
        return poller->Next(packet);
    }

    void SetMaxQueueSize(int queue_size)
    {
        auto poller = internal_poller_impl_.lock();
        CHECK(poller) << "OutputStreamPollerImpl is already destroyed.";
        return poller->SetMaxQueueSize(queue_size);
    }

    // Returns the number of packets in the queue.
    int QueueSize()
    {
        auto poller = internal_poller_impl_.lock();
        CHECK(poller) << "OutputStreamPollerImpl is already destroyed.";
        return poller->QueueSize();
    }

private:
    OutputStreamPoller(
        std::shared_ptr<internal::OutputStreamPollerImpl> internal_poller_impl)
        : internal_poller_impl_(internal_poller_impl)
    {
    }

    std::weak_ptr<internal::OutputStreamPollerImpl> internal_poller_impl_;

    // Friend class to connect OutputStreamPoller with
    // internal::OutputStreamPollerImpl.
    friend class CalculatorGraph;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_OUTPUT_STREAM_POLLER_H_
