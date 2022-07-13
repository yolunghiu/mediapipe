#include <cstdlib>
#include <typeinfo>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kMultiHandLandmarksOutputStream[] = "landmarks";

// /home/ark/res/deep-learning/mediapipe/mediapipe/graphs/hand_tracking/hand.pbtxt
ABSL_FLAG(std::string,
          calculator_graph_config_file,
          "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string,
          input_video_path,
          "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string,
          output_video_path,
          "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph()
{
    using namespace std;

    // 读取config文件，将文件内容输出到log中
    string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file),
                                                    &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    // 根据配置文件创建Graph
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    // 初始化Graph
    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    // video or camera
    LOG(INFO) << "Initialize the camera or load the video.";
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video)
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    else
        capture.open(cv::CAP_ANY);
    RET_CHECK(capture.isOpened());

    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 24);
#endif

    // 从poller中获取结果，同步机制，也可使用回调函数
    LOG(INFO) << "Start running the calculator graph.";
    //    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
    //    graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller multi_hand_landmarks_poller,
                     graph.AddOutputStreamPoller(kMultiHandLandmarksOutputStream));

    // run graph
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";
    bool grab_frames = true;
    while (grab_frames)
    {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;  // BGR image
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
        {
            if (!load_video)
            {
                LOG(INFO) << "Ignore empty frames from camera.";
                continue;
            }
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }

        cv::Mat camera_frame;  // convert BGR to RGB
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        // 水平翻转
        if (!load_video)
        {
            cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
        }

        // Wrap Mat into an ImageFrame.
        auto input_frame =
            absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB,
                                                     camera_frame.cols,
                                                     camera_frame.rows,
                                                     mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream,
            mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        // 获取运行结果，失败的话就结束
        // 获取output_video
        /* mediapipe::Packet packet;
         if (!poller.Next(&packet)) break;
         auto& output_frame = packet.Get<mediapipe::ImageFrame>();*/
        // 获取multi_hand_landmarks
        mediapipe::Packet multi_hand_landmarks_packet;
        if (!multi_hand_landmarks_poller.Next(&multi_hand_landmarks_packet)) break;
        const auto& multi_hand_landmarks =
            multi_hand_landmarks_packet.Get<vector<mediapipe::NormalizedLandmarkList>>();
        LOG(INFO) << "Type of multi_hand_landmarks: " << typeid(multi_hand_landmarks).name();
        LOG(INFO) << "Size of lanmark list: " << multi_hand_landmarks.size();

        // Convert back to opencv for display or saving.
        /*cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        if (save_video)
        {
            if (!writer.isOpened())
            {
                LOG(INFO) << "Prepare video writer.";
                writer.open(absl::GetFlag(FLAGS_output_video_path),
                            mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                            capture.get(cv::CAP_PROP_FPS),
                            output_frame_mat.size());
                RET_CHECK(writer.isOpened());
            }
            writer.write(output_frame_mat);
        }
        else
        {
            cv::imshow(kWindowName, output_frame_mat);
            // Press any key to exit.
            const int pressed_key = cv::waitKey(5);
            if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
        }*/
    }

    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
