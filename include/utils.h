#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void get_file_ext(string path, string &ext);
void string_split(const string str, const string splits, vector<string> &res);
void get_files(string pattern, vector<string> &files);
int endswith(string s, string sub);
string get_filename(const string& path);
string get_extension(const string& path);
vector<string> get_all_files(const string& folder_path);
