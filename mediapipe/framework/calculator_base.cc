// Definitions for CalculatorBase.

#include "mediapipe/framework/calculator_base.h"

#include <algorithm>

namespace mediapipe
{
CalculatorBase::CalculatorBase()
{
}

CalculatorBase::~CalculatorBase()
{
}

Timestamp CalculatorBase::SourceProcessOrder(const CalculatorContext* cc) const
{
    Timestamp result = Timestamp::Max();
    for (const OutputStreamShard& output : cc->Outputs())
    {
        result = std::min(result, output.NextTimestampBound());
    }
    return result;
}

}  // namespace mediapipe
