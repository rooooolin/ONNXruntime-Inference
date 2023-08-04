
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;
using namespace std;
using namespace cv;

string get_filename(const string& path) {
    size_t slash_pos = path.find_last_of("/\\");

    if (slash_pos == string::npos) {
        return path;
    } else {
        return path.substr(slash_pos + 1);
    }
}
string get_extension(const string& path) {
    size_t dot_pos = path.find_last_of(".");

    if (dot_pos == string::npos) {
        return "";
    } else {
        return path.substr(dot_pos);
    }
}


void get_file_ext(string path, string &ext)
{
    for (int i = path.size() - 1; i > 0; i--)
    {
        if (path[i] == '.')
        {
            ext = path.substr(i + 1);
            return;
        }
    }
    ext = path;
}
void string_split(const string str, const string splits, vector<string> &res)
{
    if (str == "")
    {
        return;
    }
    string strs = str + splits;
    int step = splits.size();
    size_t pos = strs.find(splits);
    while (pos != strs.npos)
    {
        string tmp = strs.substr(0, pos);
        res.push_back(tmp);
        strs = strs.substr(pos + step, strs.size());
        pos = strs.find(splits);
    }
}
void get_files(string pattern, vector<string> &files)
{
    vector<cv::String> fn;
    glob(pattern, fn, false);
    size_t count = fn.size();
    for (size_t i = 0; i < fn.size(); i++)
    {
        files.push_back(fn[i]);
    }
}
vector<string> get_all_files(const string& folder_path) {
    vector<string> file_paths;

    for (const auto& entry : bfs::recursive_directory_iterator(folder_path)) {
        if (entry.status().type() == bfs::regular_file) {
            file_paths.push_back(entry.path().string());
        }
    }

    return file_paths;
}

int endswith(string s, string sub) {
	if (s.rfind(sub) == -1) {
		return 0;
	}
	else {
		return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
	}

}

