#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <iomanip>
#include "inference.h"
#include "utils.h"
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // hyper-parameters
    //
    string video_ext[4] = {"mp4", "mov", "avi", "mkv"};
    // the path of model: onnx / trt / engine
    string model_path = "path/to/onnx";
    // inputs: webcam / video [eg: xxx.mp4] / image or image path [use ";" to split the mutilple image or image path]
    string streams = "path/to/images"; ///frankfurt_000000_000294_leftImg8bit.png
    // outputs: path to save predicted image and the result in JSON format
    string output_path = "path/to/save";
    // if write result in json format
    bool write_rst = true;

    // load and init the model
    Detector detector(model_path);
    // create the result file , the program will wirte the result in the file if write_rst else wirte nothing
    int json_indent = 4;
    string out = output_path + "/rst.json";
    fstream fout(out, ios::out | ios::trunc);
    if (fout.fail())
    {
        cout << "failed to create the rst.json" << endl;
        exit(0);
    }
    fout << "[" << endl;

    // if the inputs is webcam or video
    bool is_cam_or_video = false;
    string file_ext = get_extension(streams);
    int video_ext_n = sizeof(video_ext) / sizeof(*video_ext);
    bool video_exits = find(video_ext, video_ext + video_ext_n, file_ext) != video_ext + video_ext_n;
    if (streams == "webcam" or video_exits)
    {
        int aa = 0;
    }
    // if the inputs is image or image path
    else
    {
        cout << "load data:"
             << " ";
        vector<string> stream_list;
        vector<string> images;
        string_split(streams, ";", stream_list);
        for (int i = 0; i < stream_list.size(); i++)
        {
            string f = stream_list[i];

            if (endswith(f, ".jpg") or endswith(f, ".png") or endswith(f, ".bmp"))
            {
                images.push_back(f);
            }
            else
            {
                vector<string> all_files = get_all_files(f);
                images.insert(images.end(),all_files.begin(),all_files.end());
                // string pattern = f + "/*.jpg";
                // get_files(pattern, images);
                // pattern = f + "/*.png";
                // get_files(pattern, images);
            }
        }
        cout << images.size() << endl;
        vector<BoundingBox> result;
        for (int i = 0; i < images.size(); i++)
        {
            string image = images[i];
            string image_id = get_filename(image);
            image_id.erase(image_id.size() - 4);
            // predict
            cout<<"["<<i+1<<"/"<<images.size()<<"] ";
            result = detector.run(image, output_path);

            if (write_rst)
            {
                for (const auto &box : result)
                {
                    if (json_indent == 0)
                    {
                        fout << "{\"image_id\":\"" << image_id << "\",\"category_id\":1"
                             << ",\"bbox\":[" << setprecision(9) << box.x1 << "," << setprecision(9) << box.y1 << "," << setprecision(9) << box.x2 << "," << setprecision(9) << box.y2 << "],\"score\":" << setprecision(9) << box.score << "},";
                    }
                    else
                    {
                        fout << setw(json_indent) << setfill(' ') << ""
                             << "{" << endl;
                        fout << setw(2 * json_indent) << setfill(' ') << ""
                             << "\"image_id\": \"" << image_id << "\"," << endl;
                        fout << setw(2 * json_indent) << setfill(' ') << ""
                             << "\"category_id\": 1," << endl;
                        fout << setw(2 * json_indent) << setfill(' ') << ""
                             << "\"bbox\": [" << endl;
                        fout << setw(3 * json_indent) << setfill(' ') << "" << setprecision(9) << box.x1 << "," << endl;
                        fout << setw(3 * json_indent) << setfill(' ') << "" << setprecision(9) << box.y1 << "," << endl;
                        fout << setw(3 * json_indent) << setfill(' ') << "" << setprecision(9) << box.x2 << "," << endl;
                        fout << setw(3 * json_indent) << setfill(' ') << "" << setprecision(9) << box.y2 << endl;
                        fout << setw(2 * json_indent) << setfill(' ') << ""
                             << "]," << endl;
                        fout << setw(2 * json_indent) << setfill(' ') << ""
                             << "\"score\": " << setprecision(9) << box.score << endl;
                        fout << setw(json_indent) << setfill(' ') << ""
                             << "}," << endl;
                    }
                }
            }
        }
    }
    int index = -1;
    if (json_indent > 0)
    {
        index = -2;
    }
    fout.seekp(index, ios::end);
    fout << endl
         << "]" << endl;
    fout.close();

    return 0;
}