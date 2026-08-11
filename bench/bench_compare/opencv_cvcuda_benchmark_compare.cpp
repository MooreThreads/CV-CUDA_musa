#include "BenchUtils.hpp"

#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <cvcuda/OpAverageBlur.hpp>
#include <cvcuda/OpBilateralFilter.hpp>
#include <cvcuda/OpCenterCrop.hpp>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/OpHistogram.hpp>
#include <cvcuda/OpHistogramEq.hpp>
#include <cvcuda/OpLaplacian.hpp>
#include <cvcuda/OpMedianBlur.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpRemap.hpp>
#include <cvcuda/OpRotate.hpp>
#include <cvcuda/OpThreshold.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/OpWarpPerspective.hpp>

#include <nvcv/Tensor.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>

#ifdef NVCV_USE_MUSA
#    ifndef cudaEventCreate
#        define cudaEventCreate musaEventCreate
#    endif
#    ifndef cudaEventElapsedTime
#        define cudaEventElapsedTime musaEventElapsedTime
#    endif
#    ifndef cudaSetDevice
#        define cudaSetDevice musaSetDevice
#    endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result
{
    std::string benchmark;
    std::string backend;
    std::string device;
    std::string dtype;
    std::string shape;
    std::string params;
    int         samples{};
    double      cpu_sec{};
    double      gpu_sec{};
    double      stdev_sec{};
    std::string status{"success"};
    std::string notes;
};

#define CUDA_CHECK(expr)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err__ = (expr);                                                                      \
        if (err__ != cudaSuccess)                                                                        \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));          \
    } while (0)

double stdev(const std::vector<double> &values, double mean)
{
    if (values.size() < 2)
        return 0.0;
    double accum = 0.0;
    for (double v : values)
        accum += (v - mean) * (v - mean);
    return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

template<typename F>
double time_once(F &&fn)
{
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

template<typename F>
Result measure_opencv(const std::string &benchmark, const std::string &dtype, const std::string &shape,
                      const std::string &params, int warmup, int samples, F &&fn)
{
    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> times;
    times.reserve(samples);
    for (int i = 0; i < samples; ++i)
        times.push_back(time_once(fn));

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    return {benchmark, "opencv_cpu", "CPU", dtype, shape, params, samples, mean, 0.0, stdev(times, mean),
            "success", "OpenCV backend: direct cv::* CPU call"};
}

template<typename F>
Result measure_cvcuda(const std::string &benchmark, const std::string &dtype, const std::string &shape,
                      const std::string &params, int warmup, int samples, F &&fn)
{
#ifdef NVCV_USE_MUSA
    std::cerr << "START cvcuda " << benchmark << " " << dtype << " " << params << std::endl;
#endif

    cudaStream_t stream{};
    cudaEvent_t  start{}, stop{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        fn(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> gpu_times;
    std::vector<double> cpu_submit_times;
    gpu_times.reserve(samples);
    cpu_submit_times.reserve(samples);

    for (int i = 0; i < samples; ++i)
    {
        auto cpu_start = Clock::now();
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        auto cpu_end = Clock::now();

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_times.push_back(static_cast<double>(ms) / 1000.0);
        cpu_submit_times.push_back(std::chrono::duration<double>(cpu_end - cpu_start).count());
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));

    double gpu_mean = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / static_cast<double>(gpu_times.size());
    double cpu_mean = std::accumulate(cpu_submit_times.begin(), cpu_submit_times.end(), 0.0)
                    / static_cast<double>(cpu_submit_times.size());
#ifdef NVCV_USE_MUSA
    const char *device = "MUSA";
#else
    const char *device = "CUDA";
#endif
    return {benchmark, "cvcuda_gpu", device, dtype, shape, params, samples, cpu_mean, gpu_mean,
            stdev(gpu_times, gpu_mean), "success", "CV-CUDA backend: direct cvcuda::* API call with CUDA event timing"};
}

void write_csv(const std::string &path, const std::vector<Result> &results)
{
    std::ofstream out(path);
    out << "Benchmark,Backend,Device,InOutDataType,shape,params,Samples,CPU Time (sec),GPU Time (sec),StdDev (sec),Status,Notes\n";
    out << std::setprecision(17);
    for (const auto &r : results)
    {
        out << r.benchmark << ',' << r.backend << ',' << r.device << ',' << r.dtype << ',' << r.shape << ','
            << r.params << ',' << r.samples << ',' << r.cpu_sec << ',' << r.gpu_sec << ',' << r.stdev_sec << ','
            << r.status << ',' << r.notes << '\n';
    }
}

template<typename T>
cv::Mat make_input(int rows, int cols)
{
    cv::Mat src(rows, cols, cv::DataType<T>::type);
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(std::is_same_v<T, uint8_t> ? 255 : 1));
    return src;
}

cv::Mat make_input_u8c3(int rows, int cols)
{
    cv::Mat src(rows, cols, CV_8UC3);
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));
    return src;
}

cv::Mat make_input_u8c4(int rows, int cols)
{
    cv::Mat src(rows, cols, CV_8UC4);
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));
    return src;
}

template<typename T>
std::string dtype_name()
{
    return std::is_same_v<T, uint8_t> ? "U8" : "F32";
}

template<typename T>
nvcv::Tensor make_tensor(long n, long h, long w)
{
    nvcv::Tensor tensor({{n, h, w, 1}, "NHWC"}, benchutils::GetDataType<T>());
    benchutils::FillTensor<T>(tensor, benchutils::RandomValues<T>());
    return tensor;
}

nvcv::Tensor make_tensor_u8(long n, long h, long w, const nvcv::ImageFormat &format)
{
    nvcv::Tensor tensor(n, {static_cast<int>(w), static_cast<int>(h)}, format);
    benchutils::FillTensor<uint8_t>(tensor, benchutils::RandomValues<uint8_t>());
    return tensor;
}

template<typename T>
nvcv::Tensor make_scalar_tensor(long n, double value)
{
    nvcv::Tensor tensor({{n}, "N"}, nvcv::TYPE_F64);
    benchutils::FillTensor<double>(tensor, [value](const long4_16a &) { return value; });
    return tensor;
}

nvcv::Tensor make_dense_remap(long n, long h, long w)
{
    nvcv::Tensor map({{n, h, w, 1}, "NHWC"}, nvcv::TYPE_2F32);
    benchutils::FillTensor<float2>(map, [h, w](const long4_16a &c) {
        const float x = w > 1 ? (static_cast<float>(c.z) * 2.0f / static_cast<float>(w - 1) - 1.0f) : 0.0f;
        const float y = h > 1 ? (static_cast<float>(c.y) * 2.0f / static_cast<float>(h - 1) - 1.0f) : 0.0f;
        return float2{x, y};
    });
    return map;
}

cv::Mat make_opencv_remap(int rows, int cols)
{
    cv::Mat map(rows, cols, CV_32FC2);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
            map.at<cv::Vec2f>(y, x) = cv::Vec2f(static_cast<float>(x), static_cast<float>(y));
    }
    return map;
}

template<typename T>
void add_opencv_cases(std::vector<Result> &results, int warmup, int samples)
{
    constexpr int rows = 1080;
    constexpr int cols = 1920;
    const std::string dtype = dtype_name<T>();
    const std::string shape = "1x1080x1920";

    cv::Mat src = make_input<T>(rows, cols);
    cv::Mat dst;

    results.push_back(measure_opencv("Resize", dtype, shape, "resizeType=EXPAND;interpolation=LINEAR",
                                     warmup, samples, [&] {
                                         cv::resize(src, dst, cv::Size(cols * 2, rows * 2), 0.0, 0.0,
                                                    cv::INTER_LINEAR);
                                     }));

    results.push_back(measure_opencv("Flip", dtype, shape, "flipType=BOTH", warmup, samples, [&] {
        cv::flip(src, dst, -1);
    }));

    results.push_back(measure_opencv("Flip", dtype, shape, "flipType=HORIZONTAL", warmup, samples, [&] {
        cv::flip(src, dst, 1);
    }));

    results.push_back(measure_opencv("Flip", dtype, shape, "flipType=VERTICAL", warmup, samples, [&] {
        cv::flip(src, dst, 0);
    }));

    const int gaussian_kernel = static_cast<int>(std::round(1.2 * (std::is_same_v<T, uint8_t> ? 3 : 4) * 2 + 1)) | 1;
    results.push_back(measure_opencv("Gaussian", dtype, shape, "sigma=1.2;border=REFLECT", warmup, samples, [&] {
        cv::GaussianBlur(src, dst, cv::Size(gaussian_kernel, gaussian_kernel), 1.2, 1.2, cv::BORDER_REFLECT);
    }));

    results.push_back(measure_opencv("AverageBlur", dtype, shape, "kernel=7x7;border=REPLICATE", warmup, samples, [&] {
        cv::blur(src, dst, cv::Size(7, 7), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    }));

    results.push_back(measure_opencv("MedianBlur", dtype, shape, "kernel=5x5", warmup, samples, [&] {
        cv::medianBlur(src, dst, 5);
    }));

    const double thresh = std::is_same_v<T, uint8_t> ? 127.0 : 0.5;
    const double maxval = std::is_same_v<T, uint8_t> ? 255.0 : 1.0;
    auto add_threshold_case = [&](const std::string &params, int threshold_type, double threshold_value,
                                  double threshold_maxval) {
        results.push_back(measure_opencv("Threshold", dtype, shape, params, warmup, samples, [&] {
            cv::threshold(src, dst, threshold_value, threshold_maxval, threshold_type);
        }));
    };

    add_threshold_case("type=BINARY;thresh=auto;maxval=auto", cv::THRESH_BINARY, thresh, maxval);
    add_threshold_case("type=BINARY_INV;thresh=auto;maxval=auto", cv::THRESH_BINARY_INV, thresh, maxval);
    add_threshold_case("type=TRUNC;thresh=auto", cv::THRESH_TRUNC, thresh, maxval);
    add_threshold_case("type=TOZERO;thresh=auto", cv::THRESH_TOZERO, thresh, maxval);
    add_threshold_case("type=TOZERO_INV;thresh=auto", cv::THRESH_TOZERO_INV, thresh, maxval);

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        add_threshold_case("type=BINARY_OTSU;thresh=0;maxval=auto", cv::THRESH_BINARY | cv::THRESH_OTSU, 0.0, maxval);
    }

    results.push_back(measure_opencv("Laplacian", dtype, shape, "ksize=1;scale=1;border=REFLECT101", warmup, samples,
                                     [&] {
                                         cv::Laplacian(src, dst, -1, 1, 1.0, 0.0, cv::BORDER_REFLECT101);
                                     }));

    results.push_back(measure_opencv("CopyMakeBorder", dtype, shape, "top=srcH/2;left=srcW/2;border=REFLECT101", warmup,
                                     samples, [&] {
                                         cv::copyMakeBorder(src, dst, rows / 2, 0, cols / 2, 0, cv::BORDER_REFLECT101,
                                                            cv::Scalar::all(0));
                                     }));

    results.push_back(measure_opencv("BilateralFilter", dtype, shape,
                                     "d=5;sigmaColor=5;sigmaSpace=5;border=REFLECT", warmup, samples, [&] {
                                         cv::bilateralFilter(src, dst, 5, 5.0, 5.0, cv::BORDER_REFLECT);
                                     }));

    results.push_back(measure_opencv("ConvertTo", dtype, shape, "alpha=0.75;beta=0.125", warmup, samples, [&] {
        src.convertTo(dst, cv::DataType<T>::type, 0.75, 0.125);
    }));

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        {
            cv::Mat src_thresh = make_input<uint8_t>(rows, cols);
            cv::Mat dst_thresh;
            results.push_back(measure_opencv("AdaptiveThreshold", dtype, shape,
                                             "method=GAUSSIAN;type=BINARY;blockSize=7;C=-2.3", warmup, samples,
                                             [&] {
                                                 cv::adaptiveThreshold(src_thresh, dst_thresh, 123.0,
                                                                       cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                       cv::THRESH_BINARY, 7, -2.3);
                                             }));
        }

        {
            cv::Mat src_hist = make_input<uint8_t>(rows, cols);
            cv::Mat hist;
            int channels[] = {0};
            int hist_size[] = {256};
            float range[] = {0.0f, 256.0f};
            const float *ranges[] = {range};
            results.push_back(measure_opencv("CalcHist1d", dtype, shape, "histSize=256;range=0..256", warmup, samples,
                                             [&] {
                                                 cv::calcHist(&src_hist, 1, channels, cv::Mat(), hist, 1, hist_size,
                                                              ranges);
                                             }));
        }

        {
            cv::Mat src_eq = make_input<uint8_t>(rows, cols);
            cv::Mat dst_eq;
            results.push_back(measure_opencv("EqualizeHist", dtype, shape, "channels=1", warmup, samples, [&] {
                cv::equalizeHist(src_eq, dst_eq);
            }));
        }

        {
            cv::Mat src_rgb = make_input_u8c3(rows, cols);
            cv::Mat dst_bgr;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGB2BGR", warmup, samples, [&] {
                cv::cvtColor(src_rgb, dst_bgr, cv::COLOR_RGB2BGR);
            }));
        }

        {
            cv::Mat src_rgb = make_input_u8c3(rows, cols);
            cv::Mat dst_rgba;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGB2RGBA", warmup, samples, [&] {
                cv::cvtColor(src_rgb, dst_rgba, cv::COLOR_RGB2RGBA);
            }));
        }

        {
            cv::Mat src_rgba = make_input_u8c4(rows, cols);
            cv::Mat dst_rgb;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGBA2RGB", warmup, samples, [&] {
                cv::cvtColor(src_rgba, dst_rgb, cv::COLOR_RGBA2RGB);
            }));
        }

        {
            cv::Mat src_rgb = make_input_u8c3(rows, cols);
            cv::Mat dst_gray;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGB2GRAY", warmup, samples, [&] {
                cv::cvtColor(src_rgb, dst_gray, cv::COLOR_RGB2GRAY);
            }));
        }

        {
            cv::Mat src_gray = make_input<uint8_t>(rows, cols);
            cv::Mat dst_rgb;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=GRAY2RGB", warmup, samples, [&] {
                cv::cvtColor(src_gray, dst_rgb, cv::COLOR_GRAY2RGB);
            }));
        }

        {
            cv::Mat src_rgb = make_input_u8c3(rows, cols);
            cv::Mat dst_hsv;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGB2HSV", warmup, samples, [&] {
                cv::cvtColor(src_rgb, dst_hsv, cv::COLOR_RGB2HSV);
            }));
        }

        {
            cv::Mat src_hsv = make_input_u8c3(rows, cols);
            cv::Mat dst_rgb;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=HSV2RGB", warmup, samples, [&] {
                cv::cvtColor(src_hsv, dst_rgb, cv::COLOR_HSV2RGB);
            }));
        }

        {
            cv::Mat src_rgb = make_input_u8c3(rows, cols);
            cv::Mat dst_yuv;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=RGB2YUV", warmup, samples, [&] {
                cv::cvtColor(src_rgb, dst_yuv, cv::COLOR_RGB2YUV);
            }));
        }

        {
            cv::Mat src_yuv = make_input_u8c3(rows, cols);
            cv::Mat dst_rgb;
            results.push_back(measure_opencv("CvtColor", dtype, shape, "code=YUV2RGB", warmup, samples, [&] {
                cv::cvtColor(src_yuv, dst_rgb, cv::COLOR_YUV2RGB);
            }));
        }
    }

    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    results.push_back(measure_opencv("MorphologyErode", dtype, shape, "type=ERODE;kernel=3x3;iter=1;border=REPLICATE",
                                     warmup, samples, [&] {
                                         cv::erode(src, dst, morph_kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
                                     }));

    results.push_back(measure_opencv("MorphologyDilate", dtype, shape, "type=DILATE;kernel=3x3;iter=1;border=REPLICATE",
                                     warmup, samples, [&] {
                                         cv::dilate(src, dst, morph_kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
                                     }));

    results.push_back(measure_opencv("MorphologyOpen", dtype, shape, "type=OPEN;kernel=3x3;iter=1;border=REPLICATE",
                                     warmup, samples, [&] {
                                         cv::morphologyEx(src, dst, cv::MORPH_OPEN, morph_kernel, cv::Point(-1, -1), 1,
                                                          cv::BORDER_REPLICATE);
                                     }));

    results.push_back(measure_opencv("MorphologyClose", dtype, shape, "type=CLOSE;kernel=3x3;iter=1;border=REPLICATE",
                                     warmup, samples, [&] {
                                         cv::morphologyEx(src, dst, cv::MORPH_CLOSE, morph_kernel, cv::Point(-1, -1), 1,
                                                          cv::BORDER_REPLICATE);
                                     }));

    const double angle_rad = 123.456 * CV_PI / 180.0;
    const double alpha = std::cos(angle_rad);
    const double beta = std::sin(angle_rad);
    const double cx = cols / 2.0;
    const double cy = rows / 2.0;
    cv::Mat rotate_m = (cv::Mat_<double>(2, 3) << alpha, beta, (1.0 - alpha) * cx - beta * cy + 12.34,
                         -beta, alpha, beta * cx + (1.0 - alpha) * cy + 12.34);
    results.push_back(measure_opencv("Rotate", dtype, shape, "angle=123.456;shift=12.34;interpolation=CUBIC", warmup,
                                     samples, [&] {
                                         cv::warpAffine(src, dst, rotate_m, cv::Size(cols, rows), cv::INTER_CUBIC,
                                                        cv::BORDER_CONSTANT, cv::Scalar::all(0));
                                     }));

    cv::Mat warp_affine_m = (cv::Mat_<double>(2, 3) << 2.0, 2.0, 0.0, 3.0, 1.0, 0.0);
    results.push_back(measure_opencv("WarpAffine", dtype, shape,
                                     "matrix=bench-default;inverseMap=Y;interpolation=CUBIC;border=REFLECT", warmup,
                                     samples, [&] {
                                         cv::warpAffine(src, dst, warp_affine_m, cv::Size(cols, rows),
                                                        cv::INTER_CUBIC | cv::WARP_INVERSE_MAP, cv::BORDER_REFLECT,
                                                        cv::Scalar::all(0));
                                     }));

    cv::Mat warp_perspective_m = (cv::Mat_<double>(3, 3) << 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00);
    results.push_back(measure_opencv("WarpPerspective", dtype, shape,
                                     "matrix=bench-default;inverseMap=Y;interpolation=CUBIC;border=REFLECT", warmup,
                                     samples, [&] {
                                         cv::warpPerspective(src, dst, warp_perspective_m, cv::Size(cols, rows),
                                                             cv::INTER_CUBIC | cv::WARP_INVERSE_MAP,
                                                             cv::BORDER_REFLECT, cv::Scalar::all(0));
                                     }));

    cv::Mat remap_map = make_opencv_remap(rows, cols);
    results.push_back(measure_opencv("Remap", dtype, shape,
                                     "map=DENSE_IDENTITY;interpolation=NEAREST;border=CONSTANT", warmup, samples, [&] {
                                         cv::remap(src, dst, remap_map, cv::Mat(), cv::INTER_NEAREST,
                                                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
                                     }));

    results.push_back(measure_opencv("Remap", dtype, shape,
                                     "map=DENSE_IDENTITY;interpolation=LINEAR;border=CONSTANT", warmup, samples, [&] {
                                         cv::remap(src, dst, remap_map, cv::Mat(), cv::INTER_LINEAR,
                                                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
                                     }));

    results.push_back(measure_opencv("CenterCrop", dtype, shape, "crop=quarter", warmup, samples, [&] {
        const int crop_rows = rows / 2;
        const int crop_cols = cols / 2;
        const int y = (rows - crop_rows) / 2;
        const int x = (cols - crop_cols) / 2;
        dst = src(cv::Rect(x, y, crop_cols, crop_rows)).clone();
    }));
}

template<typename T>
void add_cvcuda_cases(std::vector<Result> &results, int warmup, int samples)
{
    const std::string dtype = dtype_name<T>();
    const std::string shape = "1x1080x1920";

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 2160, 3840, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Resize op;
        results.push_back(measure_cvcuda("Resize", dtype, shape, "resizeType=EXPAND;interpolation=LINEAR",
                                         warmup, samples, [&](cudaStream_t stream) {
                                             op(stream, src, dst, NVCV_INTERP_LINEAR);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Flip op;
        results.push_back(measure_cvcuda("Flip", dtype, shape, "flipType=BOTH", warmup, samples,
                                         [&](cudaStream_t stream) { op(stream, src, dst, -1); }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Flip op;
        results.push_back(measure_cvcuda("Flip", dtype, shape, "flipType=HORIZONTAL", warmup, samples,
                                         [&](cudaStream_t stream) { op(stream, src, dst, 0); }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Flip op;
        results.push_back(measure_cvcuda("Flip", dtype, shape, "flipType=VERTICAL", warmup, samples,
                                         [&](cudaStream_t stream) { op(stream, src, dst, 1); }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        const int gaussian_kernel = static_cast<int>(std::round(1.2 * (std::is_same_v<T, uint8_t> ? 3 : 4) * 2 + 1)) | 1;
        nvcv::Size2D kernel{gaussian_kernel, gaussian_kernel};
        double2 sigma{1.2, 1.2};
        cvcuda::Gaussian op(kernel, 1);
        results.push_back(measure_cvcuda("Gaussian", dtype, shape, "sigma=1.2;border=REFLECT", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, kernel, sigma, NVCV_BORDER_REFLECT);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Size2D kernel{7, 7};
        int2 kernel_anchor{-1, -1};
        cvcuda::AverageBlur op(kernel, 1);
        results.push_back(measure_cvcuda("AverageBlur", dtype, shape, "kernel=7x7;border=REPLICATE", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, kernel, kernel_anchor, NVCV_BORDER_REPLICATE);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Size2D kernel{5, 5};
        cvcuda::MedianBlur op(1);
        results.push_back(measure_cvcuda("MedianBlur", dtype, shape, "kernel=5x5", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, kernel);
                                         }));
    }

    {
        const double thresh = std::is_same_v<T, uint8_t> ? 127.0 : 0.5;
        const double maxval = std::is_same_v<T, uint8_t> ? 255.0 : 1.0;
        nvcv::Tensor thresh_t = make_scalar_tensor<T>(1, thresh);
        nvcv::Tensor maxval_t = make_scalar_tensor<T>(1, maxval);

        auto add_threshold_case = [&](const std::string &params, uint32_t threshold_type) {
            nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
            nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
            cvcuda::Threshold op(threshold_type, 1);
            results.push_back(measure_cvcuda("Threshold", dtype, shape, params, warmup, samples,
                                             [&](cudaStream_t stream) {
                                                 op(stream, src, dst, thresh_t, maxval_t);
                                             }));
        };

        add_threshold_case("type=BINARY;thresh=auto;maxval=auto", NVCV_THRESH_BINARY);
        add_threshold_case("type=BINARY_INV;thresh=auto;maxval=auto", NVCV_THRESH_BINARY_INV);
        add_threshold_case("type=TRUNC;thresh=auto", NVCV_THRESH_TRUNC);
        add_threshold_case("type=TOZERO;thresh=auto", NVCV_THRESH_TOZERO);
        add_threshold_case("type=TOZERO_INV;thresh=auto", NVCV_THRESH_TOZERO_INV);

        if constexpr (std::is_same_v<T, uint8_t>)
        {
            nvcv::Tensor otsu_thresh = make_scalar_tensor<T>(1, 0.0);
            nvcv::Tensor otsu_maxval = make_scalar_tensor<T>(1, maxval);
            nvcv::Tensor src = make_tensor<uint8_t>(1, 1080, 1920);
            nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<uint8_t>());
            cvcuda::Threshold op(NVCV_THRESH_BINARY | NVCV_THRESH_OTSU, 1);
            results.push_back(measure_cvcuda("Threshold", dtype, shape, "type=BINARY_OTSU;thresh=0;maxval=auto", warmup,
                                             samples, [&](cudaStream_t stream) {
                                                 op(stream, src, dst, otsu_thresh, otsu_maxval);
                                             }));
        }
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Laplacian op;
        results.push_back(measure_cvcuda("Laplacian", dtype, shape, "ksize=1;scale=1;border=REFLECT101", warmup,
                                         samples, [&](cudaStream_t stream) {
                                             op(stream, src, dst, 1, 1.0f, NVCV_BORDER_REFLECT101);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1620, 2880, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::CopyMakeBorder op;
        results.push_back(measure_cvcuda("CopyMakeBorder", dtype, shape,
                                         "top=srcH/2;left=srcW/2;border=REFLECT101", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, 540, 960, NVCV_BORDER_REFLECT101,
                                                float4{0.f, 0.f, 0.f, 0.f});
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::BilateralFilter op;
        results.push_back(measure_cvcuda("BilateralFilter", dtype, shape,
                                         "d=5;sigmaColor=5;sigmaSpace=5;border=REFLECT", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, 5, 5.0f, 5.0f, NVCV_BORDER_REFLECT);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::ConvertTo op;
        results.push_back(measure_cvcuda("ConvertTo", dtype, shape, "alpha=0.75;beta=0.125", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, 0.75, 0.125);
                                         }));
    }

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        {
            nvcv::Tensor src = make_tensor<uint8_t>(1, 1080, 1920);
            nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<uint8_t>());
            cvcuda::AdaptiveThreshold op(7, 1);
            results.push_back(measure_cvcuda("AdaptiveThreshold", dtype, shape,
                                             "method=GAUSSIAN;type=BINARY;blockSize=7;C=-2.3", warmup, samples,
                                             [&](cudaStream_t stream) {
                                                 op(stream, src, dst, 123.0, NVCV_ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    NVCV_THRESH_BINARY, 7, -2.3);
                                             }));
        }

        {
            nvcv::Tensor src = make_tensor<uint8_t>(1, 1080, 1920);
            nvcv::Tensor hist({{1, 256, 1}, "HWC"}, nvcv::TYPE_S32);
            nvcv::Tensor mask{nullptr};
            cvcuda::Histogram op;
            results.push_back(measure_cvcuda("CalcHist1d", dtype, shape, "histSize=256;range=0..256", warmup,
                                             samples, [&](cudaStream_t stream) {
                                                 op(stream, src, mask, hist);
                                             }));
        }

        {
            nvcv::Tensor src = make_tensor<uint8_t>(1, 1080, 1920);
            nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<uint8_t>());
            cvcuda::HistogramEq op(1);
            results.push_back(measure_cvcuda("EqualizeHist", dtype, shape, "channels=1", warmup, samples,
                                             [&](cudaStream_t stream) {
                                                 op(stream, src, dst);
                                             }));
        }

        auto add_cvt_color_case = [&](const std::string &params, NVCVImageFormat in_format,
                                      NVCVImageFormat out_format, NVCVColorConversionCode code) {
            nvcv::Tensor src = make_tensor_u8(1, 1080, 1920, nvcv::ImageFormat{in_format});
            nvcv::Tensor dst = make_tensor_u8(1, 1080, 1920, nvcv::ImageFormat{out_format});
            cvcuda::CvtColor op;
            results.push_back(measure_cvcuda("CvtColor", dtype, shape, params, warmup, samples,
                                             [&](cudaStream_t stream) {
                                                 op(stream, src, dst, code);
                                             }));
        };

        add_cvt_color_case("code=RGB2BGR", NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8, NVCV_COLOR_RGB2BGR);
        add_cvt_color_case("code=RGB2RGBA", NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGBA8, NVCV_COLOR_RGB2RGBA);
        add_cvt_color_case("code=RGBA2RGB", NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_RGBA2RGB);
        add_cvt_color_case("code=RGB2GRAY", NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_Y8, NVCV_COLOR_RGB2GRAY);
        add_cvt_color_case("code=GRAY2RGB", NVCV_IMAGE_FORMAT_Y8, NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_GRAY2RGB);
        add_cvt_color_case("code=RGB2HSV", NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_HSV8, NVCV_COLOR_RGB2HSV);
        add_cvt_color_case("code=HSV2RGB", NVCV_IMAGE_FORMAT_HSV8, NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_HSV2RGB);
        add_cvt_color_case("code=RGB2YUV", NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_YUV8, NVCV_COLOR_RGB2YUV);
        add_cvt_color_case("code=YUV2RGB", NVCV_IMAGE_FORMAT_YUV8, NVCV_IMAGE_FORMAT_RGB8, NVCV_COLOR_YUV2RGB);
    }

    auto add_morphology_case = [&](const std::string &benchmark, const std::string &params,
                                   NVCVMorphologyType morph_type) {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor workspace{nullptr};
        if (morph_type == NVCV_OPEN || morph_type == NVCV_CLOSE)
            workspace = nvcv::Tensor({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Size2D mask{3, 3};
        int2 anchor{-1, -1};
        cvcuda::Morphology op;
        results.push_back(measure_cvcuda(benchmark, dtype, shape, params, warmup, samples, [&](cudaStream_t stream) {
            op(stream, src, dst, workspace, morph_type, mask, anchor, 1, NVCV_BORDER_REPLICATE);
        }));
    };

    add_morphology_case("MorphologyErode", "type=ERODE;kernel=3x3;iter=1;border=REPLICATE", NVCV_ERODE);
    add_morphology_case("MorphologyDilate", "type=DILATE;kernel=3x3;iter=1;border=REPLICATE", NVCV_DILATE);
    add_morphology_case("MorphologyOpen", "type=OPEN;kernel=3x3;iter=1;border=REPLICATE", NVCV_OPEN);
    add_morphology_case("MorphologyClose", "type=CLOSE;kernel=3x3;iter=1;border=REPLICATE", NVCV_CLOSE);

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::Rotate op(1);
        double2 shift{12.34, 12.34};
        results.push_back(measure_cvcuda("Rotate", dtype, shape, "angle=123.456;shift=12.34;interpolation=CUBIC", warmup,
                                         samples, [&](cudaStream_t stream) {
                                             op(stream, src, dst, 123.456, shift, NVCV_INTERP_CUBIC);
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        NVCVAffineTransform xform{2.f, 2.f, 0.f, 3.f, 1.f, 0.f};
        int flags = NVCV_INTERP_CUBIC | NVCV_WARP_INVERSE_MAP;
        cvcuda::WarpAffine op(1);
        results.push_back(measure_cvcuda("WarpAffine", dtype, shape,
                                         "matrix=bench-default;inverseMap=Y;interpolation=CUBIC;border=REFLECT", warmup,
                                         samples, [&](cudaStream_t stream) {
                                             op(stream, src, dst, xform, flags, NVCV_BORDER_REFLECT,
                                                float4{0.f, 0.f, 0.f, 0.f});
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        NVCVPerspectiveTransform xform{0.27f, 0.16f, 0.00f, -0.11f, 0.61f, 0.65f, -0.09f, 0.06f, 1.00f};
        int flags = NVCV_INTERP_CUBIC | NVCV_WARP_INVERSE_MAP;
        cvcuda::WarpPerspective op(1);
        results.push_back(measure_cvcuda("WarpPerspective", dtype, shape,
                                         "matrix=bench-default;inverseMap=Y;interpolation=CUBIC;border=REFLECT", warmup,
                                         samples, [&](cudaStream_t stream) {
                                             op(stream, src, dst, xform, flags, NVCV_BORDER_REFLECT,
                                                float4{0.f, 0.f, 0.f, 0.f});
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor map = make_dense_remap(1, 1080, 1920);
        cvcuda::Remap op;
        results.push_back(measure_cvcuda("Remap", dtype, shape,
                                         "map=DENSE_IDENTITY;interpolation=NEAREST;border=CONSTANT", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, map, NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST,
                                                NVCV_REMAP_ABSOLUTE_NORMALIZED, true, NVCV_BORDER_CONSTANT,
                                                float4{0.f, 0.f, 0.f, 0.f});
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080, 1920, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor map = make_dense_remap(1, 1080, 1920);
        cvcuda::Remap op;
        results.push_back(measure_cvcuda("Remap", dtype, shape,
                                         "map=DENSE_IDENTITY;interpolation=LINEAR;border=CONSTANT", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, map, NVCV_INTERP_LINEAR, NVCV_INTERP_LINEAR,
                                                NVCV_REMAP_ABSOLUTE_NORMALIZED, true, NVCV_BORDER_CONSTANT,
                                                float4{0.f, 0.f, 0.f, 0.f});
                                         }));
    }

    {
        nvcv::Tensor src = make_tensor<T>(1, 1080, 1920);
        nvcv::Tensor dst({{1, 1080 / 2, 1920 / 2, 1}, "NHWC"}, benchutils::GetDataType<T>());
        cvcuda::CenterCrop op;
        nvcv::Size2D crop{1920 / 2, 1080 / 2};
        results.push_back(measure_cvcuda("CenterCrop", dtype, shape, "crop=quarter", warmup, samples,
                                         [&](cudaStream_t stream) {
                                             op(stream, src, dst, crop);
                                         }));
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::string output = argc > 1 ? argv[1] : "backend_compare_raw.csv";
    int samples = argc > 2 ? std::max(1, std::stoi(argv[2])) : 100;
    int warmup = argc > 3 ? std::max(0, std::stoi(argv[3])) : 10;

    cv::setNumThreads(1);
    CUDA_CHECK(cudaSetDevice(0));

    std::vector<Result> results;
    add_opencv_cases<uint8_t>(results, warmup, samples);
    add_opencv_cases<float>(results, warmup, samples);
    add_cvcuda_cases<uint8_t>(results, warmup, samples);
    add_cvcuda_cases<float>(results, warmup, samples);

    write_csv(output, results);
    std::cout << "Wrote " << output << " with " << results.size() << " rows\n";
    return 0;
}
