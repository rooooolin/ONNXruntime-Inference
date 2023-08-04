#include <iostream>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
using namespace std;

struct BoundingBox
{
    float x1, y1, x2, y2, score;
};

class Detector;
class Detector
{

public:
    Detector(string value);
    ~Detector();
    vector<BoundingBox> run(string image_path,string output_path);

private:
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    string model_path;
    Ort::Session *model;
    Ort::Env *env;

    int in_channel = 3;
    int in_width = 1280;
    int in_height = 640;
    int out_channel = 5;
    int out_width = 320;
    int out_height = 160;
    size_t num_input_elements = in_channel * in_width * in_height;
    size_t num_output_elements = out_channel * out_width * out_height;

    float score_thresh = 0.5;
    float nms_thresh = 0.5;
    string scale = "hw";

    // if rect the predicted object
    bool vis = true;
    bool is_cuda = true;
    int gpu_id= 1;
    cv::Scalar mean = {123.675, 116.28, 103.53};
    cv::Scalar std = {58.395, 57.12, 57.375};
    void data_preprocess(string image_path, cv::Mat &img, cv::Mat &ori);
    void hwc2chw(cv::Mat img, cv::Mat &dst);
    void load_onnx_model(string model_path);
    float iou(const BoundingBox &box1, const BoundingBox &box2);
    vector<BoundingBox> NMS(std::vector<BoundingBox> &boxes, float threshold);
    vector<BoundingBox> parse_det_offset(vector<float> pos, vector<float> reg_height, vector<float> reg_width, vector<float> offset_y, vector<float> offset_x, cv::Mat ori);
    void visimg(BoundingBox box, cv::Mat ori,string image_path,string output_path);
};