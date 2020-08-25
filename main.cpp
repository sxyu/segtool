#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

namespace {
using namespace std;
using namespace cv;

// Filesystem
bool ends_with(const std::string& s, const std::string& ending) {
    if (s.size() >= ending.size()) {
        return !s.compare(s.size() - ending.size(), ending.size(), ending);
    } else {
        return false;
    }
}
std::string remove_ext(const std::string& s) {
    size_t p = s.rfind('.');
    if (p == std::string::npos) return s;
    return s.substr(0, p);
}

const char* CV_WIN_NAME = "Segment";
const float VIEW_ZOOM_SPEED = 0.99;
const int MORPH_SZ = 15;

// OpenCV

void help() {
    cout << "\nGrabCut segmentation GUI (c) Alex Yu 2020\n"
            "Usage: segtool <image_name>\n\nGUI controls:\n"
            "Space: run GrabCut iteration\n"
            "S: Save mask to overwrite original mask (original backed up to "
            "*_mask_orig.png)\n"
            "R: Reset GrabCut\n"
            "T: Clear current active strokes (does not modify actual mask)\n\n"
            "=/-: Increase/decrease brush size\n\n"
            "u/i/o/p: Blend/image/masked image/binary mask views\n\n"
            "MIDDLE CLICK + Drag: Pan\n"
            "CTRL+MIDDLE CLICK + Drag vertically: Zoom\n\n"
            "LEFT CLICK: Paint FG\n"
            "RIGHT CLICK: Paint BG\n"
            "CTRL+LEFT CLICK: Paint maybe FG\n"
            "CTRL+RIGHT CLICK: Paint maybe BG\n";
}
const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 130, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
Mat getBinMask(const Mat& comMask) { return comMask & 1; }
// GrabCut application
class GrabCut {
   public:
    enum class ImageType {
        BLEND = 0,
        IMAGE = 1,
        MASKED_IMAGE = 2,
        MASK = 3
    } image_type;
    int radius = 5;

    void reset();
    void setImage(Mat _image, Mat _mask);
    void showImage() const;
    void mouseEvent(int event, int x, int y, int flags, void* param);
    void nextIter();
    cv::Mat getMask() const { return (mask & 1) * 255; }

    // Current iteration
    int iterCount;

    // View rectangle
    Rect view;

    // Window size
    Size size = cv::Size(1000, 720);

   private:
    void setLblsInMask(Point p, bool isBg, bool isPr);
    // Image and mask
    Mat image, mask;
    // Dynamic mask brushstrokes for visualization
    Mat brushstrokes;
    // Internal grabcut model
    Mat bgdModel, fgdModel;
    // Mouse drag flags
    bool paintingFG = false, paintingBG = false, draggingView = false;
    cv::Point dragStart;
};

void GrabCut::reset() {
    if (brushstrokes.empty())
        brushstrokes.create(mask.rows, mask.cols, CV_8UC3);
    brushstrokes.setTo(Scalar(0, 0, 0));
    iterCount = 0;
}
void GrabCut::setImage(Mat _image, Mat _mask) {
    if (_image.empty() || _mask.empty()) return;
    image = _image;
    mask = _mask;
    view = cv::Rect(0, 0, image.cols, image.rows);
    reset();
    grabCut(image, mask, cv::Rect(), bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
}
void GrabCut::showImage() const {
    if (image.empty() || mask.empty()) return;

    Mat res;
    if (image_type == ImageType::IMAGE) {
        res = image(view).clone();
    } else if (image_type == ImageType::MASK) {
        res = (mask(view).clone() & 1) * 255;
        cvtColor(res, res, cv::COLOR_GRAY2BGR);
    } else {
        Mat im_crop = image(view);
        im_crop.copyTo(res, mask(view) & 1);
        if (image_type == ImageType::BLEND) {
            res = res / 4 * 3 + im_crop / 4;
        }
    }
    for (int r = 0; r < res.rows; ++r) {
        auto* resPtr = res.ptr<cv::Vec3b>(r);
        const auto* maskDynPtr = brushstrokes.ptr<cv::Vec3b>(r + view.y);
        for (int c = 0; c < res.cols; ++c) {
            int cc = c + view.x;
            if (maskDynPtr[cc][0] != 0 || maskDynPtr[cc][1] != 0 ||
                maskDynPtr[cc][2] != 0) {
                resPtr[c] = maskDynPtr[cc];
            }
        }
    }
    float scale =
        std::min((float)size.width / res.cols, (float)size.height / res.rows);
    resize(res, res, cv::Size(0, 0), scale, scale);
    imshow(CV_WIN_NAME, res);
}

void GrabCut::setLblsInMask(Point p_screen, bool isBg, bool isPr) {
    float scale = std::min((float)size.width / view.width,
                           (float)size.height / view.height);
    Point p(p_screen.x / scale + view.x, p_screen.y / scale + view.y);
    if (!isPr) {
        if (isBg) {
            circle(mask, p, radius, GC_BGD, -1);
            circle(brushstrokes, p, radius, BLUE, -1);
        } else {
            circle(mask, p, radius, GC_FGD, -1);
            circle(brushstrokes, p, radius, RED, -1);
        }
    } else {
        if (isBg) {
            circle(mask, p, radius, GC_PR_BGD, -1);
            circle(brushstrokes, p, radius, LIGHTBLUE, -1);
        } else {
            circle(mask, p, radius, GC_PR_FGD, -1);
            circle(brushstrokes, p, radius, PINK, -1);
        }
    }
}

void GrabCut::mouseEvent(int event, int x, int y, int flags, void*) {
    switch (event) {
        case EVENT_LBUTTONDOWN:
            paintingFG = true;
            break;
        case EVENT_RBUTTONDOWN:
            paintingBG = true;
            break;
        case EVENT_MBUTTONDOWN:
            draggingView = true;
            dragStart.x = x;
            dragStart.y = y;
            break;
        case EVENT_LBUTTONUP:
            if (paintingFG) {
                setLblsInMask(Point(x, y), false, flags & EVENT_FLAG_CTRLKEY);
                paintingFG = false;
                showImage();
            }
            break;
        case EVENT_RBUTTONUP:
            if (paintingBG) {
                setLblsInMask(Point(x, y), true, flags & EVENT_FLAG_CTRLKEY);
                paintingBG = false;
                showImage();
            }
            break;
        case EVENT_MBUTTONUP:
            if (draggingView) {
                draggingView = false;
            }
            break;
        case EVENT_MOUSEMOVE:
            if (paintingFG) {
                setLblsInMask(Point(x, y), false, flags & EVENT_FLAG_CTRLKEY);
                showImage();
            } else if (paintingBG) {
                setLblsInMask(Point(x, y), true, flags & EVENT_FLAG_CTRLKEY);
                showImage();
            } else if (draggingView) {
                float scale = std::min((float)size.width / view.width,
                                       (float)size.height / view.height);
                int dx = (int)round((x - dragStart.x) / scale);
                int dy = (int)round((y - dragStart.y) / scale);
                if (flags & EVENT_FLAG_CTRLKEY) {
                    float rate =
                        dy < 0 ? VIEW_ZOOM_SPEED : 1.f / VIEW_ZOOM_SPEED;
                    view.x += view.width * (1. - rate) * ((float)x / mask.cols);
                    view.y +=
                        view.height * (1. - rate) * (1. - (float)y / mask.rows);
                    view.width *= rate;
                    view.height *= rate;
                } else {
                    view.x -= dx;
                    view.y -= dy;
                }
                view.width = std::min(std::max(0, view.width), mask.cols);
                view.height = std::min(std::max(0, view.height), mask.rows);
                view.x = std::min(std::max(0, view.x), mask.cols - view.width);
                view.y = std::min(std::max(0, view.y), mask.rows - view.height);
                dragStart.x = x;
                dragStart.y = y;
                showImage();
            }
            break;
    }
}
void GrabCut::nextIter() {
    grabCut(image, mask, cv::Rect(), bgdModel, fgdModel, 1);
    iterCount++;
    brushstrokes.setTo(Scalar(0, 0, 0));
}
GrabCut app;

void on_mouse(int event, int x, int y, int flags, void* param) {
    app.mouseEvent(event, x, y, flags, param);
}

void run_grabcut(cv::Mat im, cv::Mat mask, const std::string& save_path) {
    thread_local cv::Mat ele = getStructuringElement(
        cv::MORPH_ELLIPSE, Size(2 * MORPH_SZ + 1, 2 * MORPH_SZ + 1),
        Point(MORPH_SZ, MORPH_SZ));
    cv::Mat maskDil, maskEro;
    cv::dilate(mask, maskDil, ele);
    cv::erode(mask, maskEro, ele);
    cv::Mat gcMask = (maskDil - mask) / 255 * 2;
    gcMask += (mask - maskEro) / 255 * 3;
    gcMask += maskEro / 255;

    app.setImage(im, gcMask.clone());
    app.showImage();
    while (true) {
        char c = cv::waitKey(0);
        switch (c) {
            case 27:
                return;
            case 'h':
                help();
                break;
            case 'u':
                app.image_type = GrabCut::ImageType::BLEND;
                app.showImage();
                break;
            case 'i':
                app.image_type = GrabCut::ImageType::IMAGE;
                app.showImage();
                break;
            case 'o':
                app.image_type = GrabCut::ImageType::MASKED_IMAGE;
                app.showImage();
                break;
            case 'p':
                app.image_type = GrabCut::ImageType::MASK;
                app.showImage();
                break;
            case 'r':
                app.setImage(im, gcMask.clone());
                app.showImage();
                break;
            case 't':
                app.reset();
                app.showImage();
                break;
            case 's':
                std::cout << "Saving..\n";
                cv::imwrite(save_path, app.getMask());
                break;
            case '=':
                app.radius++;
                break;
            case '-':
                if (app.radius > 1) app.radius--;
                break;
            case ' ': {
                app.nextIter();
                app.showImage();
                cout << "Update: iter " << app.iterCount << endl;
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    namedWindow(CV_WIN_NAME, WINDOW_AUTOSIZE);
    setMouseCallback(CV_WIN_NAME, on_mouse, 0);
    help();

    for (int i = 1; i < argc; ++i) {
        std::string path(argv[i]);

        if (ends_with(path, "_mask.png") || ends_with(path, "_mask_orig.png")) {
            continue;
        }

        std::cout << path << "\n";
        cv::Mat im = cv::imread(path);
        if (im.empty()) {
            std::cerr << "WARNING: empty image, skipped\n";
            continue;
        }

        std::string path_no_ext = remove_ext(path);
        std::string mask_path = path_no_ext + "_mask.png";

        cv::Mat mask = cv::imread(mask_path);
        if (mask.empty() || (mask.type() != CV_8U && mask.type() != CV_8UC3 &&
                             mask.type() != CV_8UC4)) {
            std::cerr << "WARNING: expected grayscale mask " << mask_path
                      << ", skipped\n";
            continue;
        }
        if (mask.type() != CV_8U) {
            cv::extractChannel(mask, mask, mask.channels() - 1);
        }
        // Backup
        cv::imwrite(path_no_ext + "_mask_orig.png", mask);
        run_grabcut(im, mask, mask_path);
    }
    cv::destroyAllWindows();
}
