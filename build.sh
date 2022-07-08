#export HTTP_PROXY=127.0.0.1:7890
bazel run --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/examples/first_steps/2_2
