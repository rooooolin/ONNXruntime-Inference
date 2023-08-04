
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include "onnxruntime_cxx_api.h"
#include "inference.h"
#include "utils.h"
using namespace std;
using namespace cv;

Detector::Detector(string value)
{
    model_path = value;
    string model_ext = get_extension(model_path);
    cout << "init model:" << model_path << ends;
    if (model_ext == ".onnx")
    {
        load_onnx_model(model_path);
    }
    cout << "compelete" << endl;
}

Detector::~Detector()
{
    delete model;
    delete env;
}

vector<BoundingBox> Detector::run(string image_path, string output_path)
{
    Mat img, ori;
    data_preprocess(image_path, img, ori);

    // define the input and output of he model
    vector<float> input_values;
    vector<float> output_values(num_output_elements);
    vector<const char *> input_node_names = {"input"};
    vector<const char *> output_node_names = {"output"};
    vector<int64_t> input_dim = {1, in_channel, in_height, in_width};
    vector<int64_t> output_dim = {1, out_channel, out_height, out_width};
    vector<Ort::Value> input_tensors;
    vector<Ort::Value> output_tensors;

    // img:[in_height,in_width,3] -> input_values:[in_height*in_width*3]
    cv::Mat channels[3];
    cv::split(img, channels);
    for (int i = 0; i < img.channels(); i++)
    {
        std::vector<float> data = std::vector<float>(channels[i].reshape(1, img.cols * img.rows));
        input_values.insert(input_values.end(), data.begin(), data.end());
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // input_tensor:[1,in_height*in_width*3]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), num_input_elements, input_dim.data(),
        input_dim.size()));
    // output_tensor:[1,out_height*out_width*5]
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, output_values.data(), num_output_elements,
        output_dim.data(), output_dim.size()));

    // run model predict
    model->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
               input_tensors.data(), 1, output_node_names.data(),
               output_tensors.data(), 1);

    int _len = out_height * out_width;
    // output_values : [out_height*out_width*5]
    // pos: [1,out_height*out_width], the first out_height*out_width element in output_values
    vector<float> pos(output_values.begin(), output_values.begin() + _len);
    vector<float> reg_height(output_values.begin() + _len, output_values.begin() + _len * 2);
    vector<float> reg_width(output_values.begin() + _len * 2, output_values.begin() + _len * 3);
    vector<float> offset_y(output_values.begin() + _len * 3, output_values.begin() + _len * 4);
    vector<float> offset_x(output_values.begin() + _len * 4, output_values.end());
    // post process
    vector<BoundingBox> result = parse_det_offset(pos, reg_height, reg_width, offset_y, offset_x, ori);

    if (vis)
    {
        for (const auto &box : result)
        {
            visimg(box, ori, image_path, output_path);
        }
    }
    cout << image_path << " : " << result.size() << " targets" << endl;
    return result;
};

void Detector::hwc2chw(Mat img, Mat &dst)
{
    vector<float> dst_data;
    vector<Mat> bgrChannels(3);
    split(img, bgrChannels);
    for (int i = 0; i < bgrChannels.size(); i++)
    {
        vector<float> data = vector<float>(bgrChannels[i].reshape(1, 1));
        dst_data.insert(dst_data.end(), data.begin(), data.end());
    }
}
void Detector::data_preprocess(string image_path, Mat &img, Mat &ori)
{
    img = imread(image_path);
    ori = img.clone();

    cv::resize(img, img, Size(in_width, in_height));
    cvtColor(img, img, COLOR_BGR2RGB);

    // (img-mean) / std
    std::vector<cv::Mat> rgb(3);
    cv::split(img, rgb);
    for (auto i = 0; i < rgb.size(); i++)
    {
        rgb[i].convertTo(rgb[i], CV_32FC1, 1.0 / std[i], (0.0 - mean[i]) / std[i]);
    }
    cv::merge(rgb, img);
}

void Detector::load_onnx_model(string model_path)
{
    if (is_cuda)
    {
        cout << " with GPU:" << gpu_id << " ";
    }
    else
    {
        cout << " with CPU:" << gpu_id << " ";
    }
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    OrtCUDAProviderOptions cuda_options{
        gpu_id,
        OrtCudnnConvAlgoSearch::EXHAUSTIVE,
        std::numeric_limits<size_t>::max(),
        0,
        true};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    model = new Ort::Session(*env, model_path.c_str(), session_options);
}
float Detector::iou(const BoundingBox &box1, const BoundingBox &box2)
{
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float union_area = area1 + area2 - intersection;

    return intersection / union_area;
}
vector<BoundingBox> Detector::NMS(std::vector<BoundingBox> &boxes, float threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox &a, const BoundingBox &b)
              { return a.score > b.score; });

    std::vector<BoundingBox> result;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (!suppressed[i])
        {
            result.push_back(boxes[i]);
            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (iou(boxes[i], boxes[j]) > threshold)
                {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}
vector<BoundingBox> Detector::parse_det_offset(vector<float> pos, vector<float> reg_height, vector<float> reg_width, vector<float> offset_y, vector<float> offset_x, Mat ori)
{
    int ori_width = ori.cols;
    int ori_height = ori.rows;
    float score;
    std::vector<BoundingBox> boxes = {};
    for (int i = 0; i < out_height; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            score = pos.at(i * out_width + j);
            if (score > score_thresh)
            {
                float h = exp(reg_height.at(i * out_width + j));
                float w = exp(reg_width.at(i * out_width + j));
                float o_y = offset_y.at(i * out_width + j);
                float o_x = offset_x.at(i * out_width + j);
                float x1 = max(float(0), float((j + o_x + 0.5) - w / 2));
                float y1 = max(float(0), float((i + o_y + 0.5) - h / 2));
                float x2 = x1 + w;
                float y2 = y1 + h;

                // resize to original image
                float ori_x1 = x1 * ori_width / out_width;
                float ori_x2 = x2 * ori_width / out_width;
                float ori_y1 = y1 * ori_height / out_height;
                float ori_y2 = y2 * ori_height / out_height;

                BoundingBox box = {ori_x1, ori_y1, ori_x2, ori_y2, score};
                boxes.push_back(box);
            }
        }
    }
    // cout << boxes.size() << ends;
    std::vector<BoundingBox> result = NMS(boxes, nms_thresh);
    return result;
}
void Detector::visimg(BoundingBox box, Mat ori, string image_path, string output_path)
{
    Rect rect = Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    float score = box.score;

    rectangle(ori, rect, Scalar(0, 255, 0), 2);
    stringstream s;
    s<<fixed<<setprecision(2)<<score;
    putText(ori, s.str(), Point(rect.x, rect.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

    string file_name = get_filename(image_path);
    imwrite(output_path + "/" + file_name, ori);
}