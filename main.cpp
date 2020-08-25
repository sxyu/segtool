#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

namespace {
using namespace std;
using namespace cv;

// String/path utils
bool endsWith(const std::string& s, const std::string& ending) {
    if (s.size() >= ending.size()) {
        return !s.compare(s.size() - ending.size(), ending.size(), ending);
    } else {
        return false;
    }
}
std::string removeExt(const std::string& s) {
    size_t p = s.rfind('.');
    if (p == std::string::npos) return s;
    return s.substr(0, p);
}

// Constants
const char* CV_WIN_NAME = "Segment";
const float VIEW_ZOOM_SPEED = 0.99;
const int MORPH_SZ = 15;

void help() {
    cout << "\nGrabCut segmentation GUI (c) Alex Yu 2020\n"
            "Usage: segtool <image_name>\n\nGUI controls:\n"
            "Space: run GrabCut iteration\n"
            "S: Save mask to overwrite original mask *_mask.png, and outputs "
            "the GrabCut state *_mask_gc.png to allow resume (original "
            "backed up to "
            "*_mask_orig.png)\n"
            "R: Reset GrabCut\n"
            "C: Reset zoom/pan\n"
            "U: Clear current active strokes (basic undo)\n\n"
            "=/-: Increase/decrease brush size\n\n"
            "1-4: Blend/image/masked image/binary mask views\n\n"
            "MIDDLE CLICK + Drag: Pan\n"
            "CTRL+MIDDLE CLICK + Drag vertically: Zoom\n\n"
            "LEFT CLICK: Paint FG\n"
            "RIGHT CLICK: Paint BG\n"
            "CTRL+LEFT CLICK: Paint 'maybe FG'\n"
            "CTRL+RIGHT CLICK: Paint 'maybe BG'\n"
            "SHIFT+LEFT CLICK: Erase brushstroke\n";
}
const Vec3b RED = Vec3b(0, 0, 255);
const Vec3b PINK = Vec3b(230, 130, 255);
const Vec3b BLUE = Vec3b(255, 130, 0);
const Vec3b LIGHTBLUE = Vec3b(255, 255, 160);
const Vec3b GREEN = Vec3b(0, 255, 0);
Mat getBinMask(const Mat& comMask) { return comMask & 1; }

// GrabCut application
class GrabCut {
   public:
    // Clear brushstrokes
    void clear();
    // Set the image
    void setImage(Mat _image, Mat _mask);
    void showImage() const;
    void mouseEvent(int event, int x, int y, int flags, void* param);

    // Grabcut iteration
    void nextIter();

    cv::Mat getMaskBinary() const { return (mask & 1) * 255; }
    cv::Mat getMask() const { return mask; }

    // Visualization type
    enum class ImageType {
        BLEND = 0,
        IMAGE = 1,
        MASKED_IMAGE = 2,
        MASK = 3
    } image_type;

    // Brush radius
    int radius = 5;

    // Current iteration
    int iterCount;

    // View rectangle
    Rect2f view;

    // Window size
    Size2f size = cv::Size2f(1280, 720);

   private:
    void setLblsInMask(Point p, bool isPr);
    // Image and mask
    Mat image, mask;
    // Dynamic mask brushstrokes to be applied to mask; 255=no change
    Mat brushstrokes;
    // Internal grabcut model
    Mat bgdModel, fgdModel;
    // Mouse drag flags
    bool paintingFG = false, paintingBG = false, draggingView = false,
         erasing = false;
    cv::Point dragStart;
};

void GrabCut::clear() {
    if (brushstrokes.empty()) brushstrokes.create(mask.rows, mask.cols, CV_8U);
    brushstrokes.setTo(Scalar(255));
}
void GrabCut::setImage(Mat _image, Mat _mask) {
    if (_image.empty() || _mask.empty()) return;
    image = _image;
    mask = _mask;
    view = cv::Rect(0, 0, image.cols, image.rows);
    clear();
    iterCount = 0;
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
    const thread_local cv::Vec3b colors[4] = {BLUE, RED, LIGHTBLUE, PINK};
    for (int r = 0; r < res.rows; ++r) {
        auto* resPtr = res.ptr<cv::Vec3b>(r);
        const auto* brushPtr = brushstrokes.ptr<uint8_t>(r + (int)view.y);
        for (int c = 0; c < res.cols; ++c) {
            int cc = c + (int)view.x;
            if (brushPtr[cc] != 255) {
                resPtr[c] = colors[brushPtr[cc]] * 0.5 + resPtr[c] * 0.5;
            }
        }
    }
    float scale =
        std::min((float)size.width / res.cols, (float)size.height / res.rows);
    resize(res, res, cv::Size(0, 0), scale, scale, INTER_NEAREST);

    imshow(CV_WIN_NAME, res);
}

void GrabCut::setLblsInMask(Point p_screen, bool isPr) {
    float scale = std::min(size.width / view.width, size.height / view.height);
    Point p((int)(p_screen.x / scale + view.x),
            (int)(p_screen.y / scale + view.y));
    if (erasing) {
        circle(brushstrokes, p, radius, 255, -1);
    } else if (!isPr) {
        if (paintingBG) {
            circle(brushstrokes, p, radius, GC_BGD, -1);
        } else {
            circle(brushstrokes, p, radius, GC_FGD, -1);
        }
    } else {
        if (paintingBG) {
            circle(brushstrokes, p, radius, GC_PR_BGD, -1);
        } else {
            circle(brushstrokes, p, radius, GC_PR_FGD, -1);
        }
    }
}

void GrabCut::mouseEvent(int event, int x, int y, int flags, void*) {
    switch (event) {
        case EVENT_LBUTTONDOWN:
            paintingFG = true;
            if (flags & EVENT_FLAG_SHIFTKEY) erasing = true;
            break;
        case EVENT_RBUTTONDOWN:
            paintingBG = true;
            if (flags & EVENT_FLAG_SHIFTKEY) erasing = true;
            break;
        case EVENT_MBUTTONDOWN:
            draggingView = true;
            dragStart.x = x;
            dragStart.y = y;
            break;
        case EVENT_LBUTTONUP:
        case EVENT_RBUTTONUP:
            setLblsInMask(Point(x, y), flags & EVENT_FLAG_CTRLKEY);
            paintingFG = paintingBG = false;
            showImage();
            break;
        case EVENT_MBUTTONUP:
            if (draggingView) {
                draggingView = false;
            }
            break;
        case EVENT_MOUSEMOVE:
            if (paintingFG || paintingBG) {
                setLblsInMask(Point(x, y), flags & EVENT_FLAG_CTRLKEY);
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
                    float aspect = (float)size.height / size.width;
                    // Auto resize to fit window
                    if ((float)view.height / view.width < aspect - 1e-12) {
                        view.y -= (view.width * aspect - view.height) * 0.5f;
                        view.height = view.width * aspect;
                    } else if ((float)view.height / view.width >
                               aspect + 1e-12) {
                        view.x -= (view.height / aspect - view.width) * 0.5f;
                        view.width = view.height / aspect;
                    }
                } else {
                    view.x -= dx;
                    view.y -= dy;
                }
                view.width =
                    std::min<float>(std::max(0.f, view.width), mask.cols);
                view.height =
                    std::min<float>(std::max(0.f, view.height), mask.rows);
                view.x =
                    std::min(std::max(0.f, view.x), mask.cols - view.width);
                view.y =
                    std::min(std::max(0.f, view.y), mask.rows - view.height);
                dragStart.x = x;
                dragStart.y = y;
                showImage();
            }
            break;
    }
}
void GrabCut::nextIter() {
    for (int r = 0; r < mask.rows; ++r) {
        auto* maskPtr = mask.ptr<uint8_t>(r);
        const auto* brushPtr = brushstrokes.ptr<uint8_t>(r + (int)view.y);
        for (int c = 0; c < mask.cols; ++c) {
            int cc = c + (int)view.x;
            if (brushPtr[cc] != 255) {
                maskPtr[c] = brushPtr[c];
            }
        }
    }

    grabCut(image, mask, cv::Rect(), bgdModel, fgdModel, 1);
    iterCount++;
    clear();
}
GrabCut app;

void on_mouse(int event, int x, int y, int flags, void* param) {
    app.mouseEvent(event, x, y, flags, param);
}

void run_grabcut(cv::Mat im, cv::Mat mask, const std::string& save_path,
                 const std::string& save_path_gc) {
    app.setImage(im, mask.clone());
    app.showImage();
    while (true) {
        char c = cv::waitKey(0);
        if (c >= 'A' && c <= 'Z') c = std::tolower(c);
        switch (c) {
            case 27:
                return;
            case 'h':
                help();
                break;
            case '1':
            case '2':
            case '3':
            case '4':
                app.image_type = GrabCut::ImageType(c - '1');
                app.showImage();
                break;
            case 'r':
                app.setImage(im, mask.clone());
                app.showImage();
                break;
            case 'c':
                app.view = cv::Rect2f(0, 0, mask.cols, mask.rows);
                app.showImage();
                break;
            case 'u':
                app.clear();
                app.showImage();
                break;
            case 's':
                std::cout << "Saving..\n";
                cv::imwrite(save_path_gc, app.getMask());
                cv::imwrite(save_path, app.getMaskBinary());
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

        if (endsWith(path, "_mask.png") || endsWith(path, "_mask_orig.png")) {
            continue;
        }

        std::cout << path << "\n";
        cv::Mat im = cv::imread(path);
        if (im.empty()) {
            std::cerr << "WARNING: empty image, skipped\n";
            continue;
        }

        std::string path_no_ext = removeExt(path);
        std::string mask_path = path_no_ext + "_mask.png";
        std::string mask_gc_path = path_no_ext + "_mask_gc.png";

        // First try to load _mask_gc.png, which should save the
        // GrabCut state
        cv::Mat mask = cv::imread(mask_gc_path);
        if (mask.empty()) {
            mask = cv::imread(mask_path);
            if (mask.empty() ||
                (mask.type() != CV_8U && mask.type() != CV_8UC3 &&
                 mask.type() != CV_8UC4)) {
                std::cerr << "WARNING: expected grayscale mask " << mask_path
                          << ", skipped\n";
                continue;
            }

            thread_local cv::Mat ele = getStructuringElement(
                cv::MORPH_ELLIPSE, Size(2 * MORPH_SZ + 1, 2 * MORPH_SZ + 1),
                Point(MORPH_SZ, MORPH_SZ));
            cv::Mat maskDil, maskEro;
            cv::dilate(mask, maskDil, ele);
            cv::erode(mask, maskEro, ele);
            cv::Mat gcMask = (maskDil - mask) / 255 * 2;
            gcMask += (mask - maskEro) / 255 * 3;
            gcMask += maskEro / 255;
            mask = gcMask;
        }
        if (mask.type() != CV_8U) {
            cv::extractChannel(mask, mask, mask.channels() - 1);
        }

        // Backup
        cv::imwrite(path_no_ext + "_mask_orig.png", mask);
        run_grabcut(im, mask, mask_path, mask_gc_path);
    }

    cv::destroyAllWindows();
}
