// Two classes: Timestamp and TimestampDiff for specifying timestamps
// within the CalculatorFramework (and mediapipe in general).  MediaPipe
// timestamps are in units of _microseconds_.
//
// There are several special values (All these values must be constructed
// using the static methods provided):
//   Unset:       The default initialization value, not generally
//                valid when a timestamp is required.

//   Unstarted:   The timestamp before any valid timestamps.  This is
//                the input timestamp during Open().

//   PreStream:   应该被最先处理的数据
//                A value for specifying that a packet contains "header"
//                data that should be processed before any other
//                timestamp.  Like PostStream, if this value is sent then
//                it must be the only value that is sent on the stream.

//   Min:         The minimum range timestamp to see in Process().
//                Any number of "range" timestamp can be sent over a
//                stream, provided that they are sent in monotonically
//                increasing order.

//   Max:         The maximum range timestamp to see in Process().

//   PostStream:  A value for specifying that a packet pertains to
//                the entire stream.  This "summary" timestamp occurs
//                after all the "range" timestamps.  If this timestamp
//                is sent on a stream, it must be the only packet sent.

//   OneOverPostStream:
//                The value immediately following PostStream.
//                This should only be used internally.

//   Done:        The timestamp after all valid timestamps.
//                This is the input timestamp during Close().

#ifndef MEDIAPIPE_FRAMEWORK_TIMESTAMP_H_
#define MEDIAPIPE_FRAMEWORK_TIMESTAMP_H_

#include <cmath>
#include <string>

#include "mediapipe/framework/deps/safe_int.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe
{
// A safe int checks each arithmetic operation to make sure it will not
// have underflow/overflow etc.  This type is used internally by Timestamp
// and TimestampDiff.
MEDIAPIPE_DEFINE_SAFE_INT_TYPE(TimestampBaseType,
                               int64,
                               mediapipe::intops::LogFatalOnError);

class TimestampDiff;

// A class which represents a timestamp in the calculator framework.
// There are several special values which can only be created with the
// static functions provided in this class.
class Timestamp
{
public:
    Timestamp();
    // Construction of Timestamp() is explicit (TimestampDiff is not explicit).
    explicit Timestamp(int64 timestamp);
    explicit Timestamp(TimestampBaseType timestamp);

    // Timestamps are in microseconds.
    static constexpr double kTimestampUnitsPerSecond = 1000000.0;

    // Use the default copy constructor, assignment operator, and destructor.

    // Get the underlying int64 value being used.  This should generally be
    // avoided, but may be necessary for things like serialization.
    int64 Value() const
    {
        return timestamp_.value();
    }
    // Return the value in units of seconds (the underlying value is in
    // microseconds).
    double Seconds() const
    {
        return Value() / kTimestampUnitsPerSecond;
    }
    // Return the value in units of microseconds.  The underlying value is
    // already in microseconds, but this function should be preferred over
    // Value() in case the underlying representation changes.
    int64 Microseconds() const
    {
        return Value();
    }
    // This provides a human readable string for the special values.
    std::string DebugString() const;

    // For use by framework. Clients or Calculator implementations should not
    // call this.
    static Timestamp CreateNoErrorChecking(int64 timestamp);

    // Create a timestamp from a seconds value.
    static Timestamp FromSeconds(double seconds)
    {
        return Timestamp(
            TimestampBaseType{std::round(seconds * kTimestampUnitsPerSecond)});
    }

    // Special values.
    static Timestamp Unset();
    static Timestamp Unstarted();
    static Timestamp PreStream();
    static Timestamp Min();
    static Timestamp Max();
    static Timestamp PostStream();
    static Timestamp OneOverPostStream();
    static Timestamp Done();

    // A special value is any of the values which cannot be constructed directly
    // but must be constructed using the helper functions given above.
    bool IsSpecialValue() const
    {
        return *this <= Min() || *this >= Max();
    }

    // A range value is anything between Min() and Max() (inclusive).
    // Any number of packets with range values can be sent over a
    // stream as long as they are sent in monotonically increasing order.
    // IsRangeValue() isn't quite the opposite of IsSpecialValue() since it
    // is valid to start a stream at Timestamp::Min() and continue until
    // Timestamp::Max() (both of which are special values).  PreStream()
    // and PostStream() are not considered a range value even though
    // they can be sent over a stream (they are "summary" timestamps not
    // "range" timestamps).
    //
    // Notice that arithmetic may only be performed if IsRangeValue()
    // is true.  Arithmetic on Min and Max is valid but is almost certainly
    // bad design.
    bool IsRangeValue() const
    {
        return *this >= Min() && *this <= Max();
    }

    // Returns true iff this can be the timestamp of a Packet in a
    // stream.  Any number of RangeValue timestamps may be in a stream
    // (in monotonically increasing order).  Also, exactly one PreStream,
    // or one PostStream packet is allowed.
    bool IsAllowedInStream() const
    {
        // This is a simplified expression for
        // IsRangeValue() or PreStream() or PostStream().
        return *this >= PreStream() && *this <= PostStream();
    }

    // Common operators.
    bool operator==(const Timestamp other) const
    {
        return timestamp_ == other.timestamp_;
    }
    bool operator!=(const Timestamp other) const
    {
        return !(timestamp_ == other.timestamp_);
    }
    bool operator<(const Timestamp other) const
    {
        return timestamp_ < other.timestamp_;
    }
    bool operator<=(const Timestamp other) const
    {
        return timestamp_ <= other.timestamp_;
    }
    bool operator>(const Timestamp other) const
    {
        return timestamp_ > other.timestamp_;
    }
    bool operator>=(const Timestamp other) const
    {
        return timestamp_ >= other.timestamp_;
    }
    // Addition and subtraction of Timestamp and TimestampDiff values.
    // Notice that subtracting two Timestamp values gives a TimestampDiff
    // while adding a TimestampDiff to a Timestamp gives a Timestamp.
    // Note that no special values are allowed to be the result of these
    // operations.
    //
    // The addition or subtraction of a TimestampDiff to a Timestamp is clamped
    // within the valid range of values [Timestamp::Min(), Timestamp::Max()].
    //
    // Not all operations are allowed, in particular, you cannot add two
    // Timestamps, and you cannot subtract a Timestamp from a TimestampDiff.
    TimestampDiff operator-(const Timestamp other) const;
    Timestamp operator+(const TimestampDiff other) const;
    Timestamp operator-(const TimestampDiff other) const;
    // Unary negation of a Timestamp is not allowed.

    // Provided for convenience.
    Timestamp operator+=(const TimestampDiff other);
    Timestamp operator-=(const TimestampDiff other);
    Timestamp operator++();
    Timestamp operator--();
    Timestamp operator++(int);
    Timestamp operator--(int);

    // Returns the next timestamp in the range [Min .. Max], or
    // OneOverPostStream() if no Packets may follow one with this timestamp.
    // CHECKs that this->IsAllowedInStream().
    Timestamp NextAllowedInStream() const;

    // Returns the previous timestamp in the range [Min .. Max], or
    // Unstarted() if no Packets may preceed one with this timestamp.
    Timestamp PreviousAllowedInStream() const;

private:
    TimestampBaseType timestamp_;
};

// A class which represents the difference between two timestamps in the
// calculator framework.
class TimestampDiff
{
public:
    TimestampDiff() : timestamp_(0)
    {
    }
    // This constructor is not explicit.
    TimestampDiff(int64 timestamp) : timestamp_(timestamp)
    {  // NOLINT
    }
    // This constructor is not explicit.
    TimestampDiff(TimestampBaseType timestamp)  // NOLINT
        : timestamp_(timestamp)
    {
    }

    // Use the default copy constructor, assignment operator, and destructor.

    // Get the underlying int64 value being used.  This should generally be
    // avoided, but may be necessary for things like serialization.
    int64 Value() const
    {
        return timestamp_.value();
    }
    // Return the value in units of seconds (the underlying value is in
    // microseconds).
    double Seconds() const
    {
        return Value() / Timestamp::kTimestampUnitsPerSecond;
    }
    // Return the value in units of microseconds.  The underlying value is
    // already in microseconds, but this function should be preferred over
    // Value() in case the underlying representation changes.
    int64 Microseconds() const
    {
        return Value();
    }
    std::string DebugString() const;

    bool operator==(const TimestampDiff other) const
    {
        return timestamp_ == other.timestamp_;
    }
    bool operator!=(const TimestampDiff other) const
    {
        return !(timestamp_ == other.timestamp_);
    }
    bool operator<(const TimestampDiff other) const
    {
        return timestamp_ < other.timestamp_;
    }
    bool operator<=(const TimestampDiff other) const
    {
        return timestamp_ <= other.timestamp_;
    }
    bool operator>(const TimestampDiff other) const
    {
        return timestamp_ > other.timestamp_;
    }
    bool operator>=(const TimestampDiff other) const
    {
        return timestamp_ >= other.timestamp_;
    }
    // Unary negation of a TimestampDiff is allowed.
    const TimestampDiff operator-() const
    {
        return TimestampDiff(-timestamp_);
    }
    // See the addition and subtraction functions in Timestamp for details.
    TimestampDiff operator+(const TimestampDiff other) const;
    TimestampDiff operator-(const TimestampDiff other) const;
    Timestamp operator+(const Timestamp other) const;

    // Special values.

    static TimestampDiff Unset()
    {
        return TimestampDiff(Timestamp::Unset().Value());
    }

private:
    TimestampBaseType timestamp_;
};

// Convenience functions which are equivalent to using arg.DebugString().
std::ostream& operator<<(std::ostream& os, Timestamp arg);
std::ostream& operator<<(std::ostream& os, TimestampDiff arg);

// Implementation details.

inline Timestamp::Timestamp() : timestamp_(kint64min)
{
}

inline Timestamp::Timestamp(int64 timestamp) : timestamp_(timestamp)
{
    CHECK(!IsSpecialValue())
        << "Cannot directly create a Timestamp with a special value: "
        << CreateNoErrorChecking(timestamp);
}

inline Timestamp::Timestamp(TimestampBaseType timestamp) : timestamp_(timestamp)
{
    CHECK(!IsSpecialValue())
        << "Cannot directly create a Timestamp with a special value: "
        << CreateNoErrorChecking(timestamp.value());
}

inline Timestamp Timestamp::CreateNoErrorChecking(int64 timestamp)
{
    Timestamp tmp;
    tmp.timestamp_ = TimestampBaseType(timestamp);
    return tmp;
}

inline Timestamp Timestamp::Unset()
{
    return Timestamp();
}

inline Timestamp Timestamp::Unstarted()
{
    return CreateNoErrorChecking(kint64min + 1);
}

inline Timestamp Timestamp::PreStream()
{
    return CreateNoErrorChecking(kint64min + 2);
}

inline Timestamp Timestamp::Min()
{
    return CreateNoErrorChecking(kint64min + 3);
}

inline Timestamp Timestamp::Max()
{
    return CreateNoErrorChecking(kint64max - 3);
}

inline Timestamp Timestamp::PostStream()
{
    return CreateNoErrorChecking(kint64max - 2);
}

inline Timestamp Timestamp::OneOverPostStream()
{
    return CreateNoErrorChecking(kint64max - 1);
}

inline Timestamp Timestamp::Done()
{
    return CreateNoErrorChecking(kint64max);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TIMESTAMP_H_
