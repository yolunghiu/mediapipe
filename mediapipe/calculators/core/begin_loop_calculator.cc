#include "mediapipe/calculators/core/begin_loop_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

// A calculator to process std::vector<NormalizedLandmarkList>.
typedef BeginLoopCalculator<std::vector<::mediapipe::NormalizedLandmarkList>>
    BeginLoopNormalizedLandmarkListVectorCalculator;
REGISTER_CALCULATOR(BeginLoopNormalizedLandmarkListVectorCalculator);

// A calculator to process std::vector<int>.
typedef BeginLoopCalculator<std::vector<int>> BeginLoopIntCalculator;
REGISTER_CALCULATOR(BeginLoopIntCalculator);

// A calculator to process std::vector<NormalizedRect>.
typedef BeginLoopCalculator<std::vector<::mediapipe::NormalizedRect>>
    BeginLoopNormalizedRectCalculator;
REGISTER_CALCULATOR(BeginLoopNormalizedRectCalculator);

// A calculator to process std::vector<Detection>.
typedef BeginLoopCalculator<std::vector<::mediapipe::Detection>>
    BeginLoopDetectionCalculator;
REGISTER_CALCULATOR(BeginLoopDetectionCalculator);

// A calculator to process std::vector<Matrix>.
typedef BeginLoopCalculator<std::vector<Matrix>> BeginLoopMatrixCalculator;
REGISTER_CALCULATOR(BeginLoopMatrixCalculator);

// A calculator to process std::vector<std::vector<Matrix>>.
typedef BeginLoopCalculator<std::vector<std::vector<Matrix>>>
    BeginLoopMatrixVectorCalculator;
REGISTER_CALCULATOR(BeginLoopMatrixVectorCalculator);

// A calculator to process std::vector<uint64_t>.
typedef BeginLoopCalculator<std::vector<uint64_t>> BeginLoopUint64tCalculator;
REGISTER_CALCULATOR(BeginLoopUint64tCalculator);

}  // namespace mediapipe
