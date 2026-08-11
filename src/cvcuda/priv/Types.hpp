/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CVCUDA_TYPES_HPP
#define CVCUDA_TYPES_HPP

#include <cvcuda/Types.h>

#include <algorithm>
#include <string>
#include <vector>

namespace cvcuda::priv {

#define checkERR(call) check_error(call, #call, __LINE__, __FILE__)

#if defined(NVCV_USE_MUSA)
// musify: MUSA runtime API mappings from /usr/local/musa/tools/general.json.
using DeviceError_t = musaError_t;
using DeviceMemcpyKind_t = musaMemcpyKind;
inline static constexpr DeviceError_t kDeviceSuccess = musaSuccess;
inline static const char *DeviceGetErrorString(DeviceError_t e)
{
    return musaGetErrorString(e);
}
inline static const char *DeviceGetErrorName(DeviceError_t e)
{
    return musaGetErrorName(e);
}
template<typename T>
inline static DeviceError_t DeviceMalloc(T **ptr, size_t size)
{
    return musaMalloc(reinterpret_cast<void **>(ptr), size);
}
inline static DeviceError_t DeviceMemcpy(void *dst, const void *src, size_t count, DeviceMemcpyKind_t kind)
{
    return musaMemcpy(dst, src, count, kind);
}
inline static DeviceError_t DeviceFree(void *ptr)
{
    return musaFree(ptr);
}
inline static constexpr auto kDeviceMemcpyHostToDevice   = musaMemcpyHostToDevice;
inline static constexpr auto kDeviceMemcpyDeviceToDevice = musaMemcpyDeviceToDevice;
#else
using DeviceError_t = cudaError_t;
using DeviceMemcpyKind_t = cudaMemcpyKind;
inline static constexpr DeviceError_t kDeviceSuccess = cudaSuccess;
inline static const char *DeviceGetErrorString(DeviceError_t e)
{
    return cudaGetErrorString(e);
}
inline static const char *DeviceGetErrorName(DeviceError_t e)
{
    return cudaGetErrorName(e);
}
template<typename T>
inline static DeviceError_t DeviceMalloc(T **ptr, size_t size)
{
    return cudaMalloc(reinterpret_cast<void **>(ptr), size);
}
inline static DeviceError_t DeviceMemcpy(void *dst, const void *src, size_t count, DeviceMemcpyKind_t kind)
{
    return cudaMemcpy(dst, src, count, kind);
}
inline static DeviceError_t DeviceFree(void *ptr)
{
    return cudaFree(ptr);
}
inline static constexpr auto kDeviceMemcpyHostToDevice   = cudaMemcpyHostToDevice;
inline static constexpr auto kDeviceMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
#endif

inline static bool check_error(DeviceError_t e, const char *call, int line, const char *file)
{
    if (e != kDeviceSuccess)
    {
        fprintf(stderr, "Device Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, DeviceGetErrorString(e),
                DeviceGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

// Default font, user can install via below command:
//      sudo apt-get update
//      sudo apt-get install ttf-dejavu fonts-dejavu
#define DEFAULT_OSD_FONT "DejaVuSansMono"

class NVCVText
{
public:
    const char   *utf8Text = nullptr; // Text to draw in utf8 format.
    int32_t       fontSize;           // Font size for the text.
    const char   *fontName = nullptr; // Font name for the text.
    NVCVPointI    tlPos;              // Top-left corner point for label text, \ref NVCVPointI.
    NVCVColorRGBA fontColor;          // Font color of the text.
    NVCVColorRGBA bgColor;            // Background color of text box.

    NVCVText(const char *_utf8Text, int32_t _fontSize, const char *_fontName, NVCVPointI _tlPos,
             NVCVColorRGBA _fontColor, NVCVColorRGBA _bgColor)
        : fontSize(_fontSize)
        , tlPos(_tlPos)
        , fontColor(_fontColor)
        , bgColor(_bgColor)
    {
        size_t len          = std::char_traits<char>::length(_utf8Text);
        char  *tmp_utf8Text = (char *)malloc(len + 1);
        std::copy_n(_utf8Text, len + 1, tmp_utf8Text);
        len                = std::char_traits<char>::length(_fontName);
        char *tmp_fontName = (char *)malloc(len + 1);
        std::copy_n(_fontName, len + 1, tmp_fontName);
        utf8Text = tmp_utf8Text;
        fontName = tmp_fontName;
    }

    NVCVText(const NVCVText &text)
        : fontSize(text.fontSize)
        , tlPos(text.tlPos)
        , fontColor(text.fontColor)
        , bgColor(text.bgColor)
    {
        size_t len          = std::char_traits<char>::length(text.utf8Text);
        char  *tmp_utf8Text = (char *)malloc(len + 1);
        std::copy_n(text.utf8Text, len + 1, tmp_utf8Text);
        len                = std::char_traits<char>::length(text.fontName);
        char *tmp_fontName = (char *)malloc(len + 1);
        std::copy_n(text.fontName, len + 1, tmp_fontName);
        utf8Text = tmp_utf8Text;
        fontName = tmp_fontName;
    }

    NVCVText &operator=(const NVCVText &text)
    {
        if (this != &text)
        {
            if (utf8Text != nullptr)
            {
                free((void *)utf8Text);
                utf8Text = nullptr;
            }
            if (fontName != nullptr)
            {
                free((void *)fontName);
                fontName = nullptr;
            }
            *this = NVCVText(text);
        }
        return *this;
    };

    ~NVCVText()
    {
        if (utf8Text != nullptr)
        {
            free((void *)utf8Text);
            utf8Text = nullptr;
        }
        if (fontName != nullptr)
        {
            free((void *)fontName);
            fontName = nullptr;
        }
    };
};

class NVCVSegment
{
public:
    NVCVBoxI      box;            // Bounding box of segment, \ref NVCVBoxI.
    int32_t       thickness;      // Line thickness of segment outter rect.
    float        *dSeg = nullptr; // Device pointer for segment mask, cannot be nullptr.
                                  // Array length: segWidth * segHeight
                                  // Format:
                                  //      Score_00, Score_01, ..., Score_0k, ...
                                  //      Score_10, Score_11, ..., Score_kk, ...
                                  //          ... ,     ... , ...,     ... , ...
    int32_t       segWidth;       // Segment mask width.
    int32_t       segHeight;      // Segment mask height.
    float         segThreshold;   // Segment threshold.
    NVCVColorRGBA borderColor;    // Line color of segment outter rect.
    NVCVColorRGBA segColor;       // Segment mask color.

    NVCVSegment(NVCVBoxI _box, int32_t _thickness, float *_hSeg, int32_t _segWidth, int32_t _segHeight,
                float _segThreshold, NVCVColorRGBA _borderColor, NVCVColorRGBA _segColor)
        : box(_box)
        , thickness(_thickness)
        , segWidth(_segWidth)
        , segHeight(_segHeight)
        , segThreshold(_segThreshold)
        , borderColor(_borderColor)
        , segColor(_segColor)
    {
        checkERR(DeviceMalloc(&dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float)));
        checkERR(
            DeviceMemcpy(dSeg, _hSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float), kDeviceMemcpyHostToDevice));
    }

    NVCVSegment(const NVCVSegment &segment)
        : box(segment.box)
        , thickness(segment.thickness)
        , segWidth(segment.segWidth)
        , segHeight(segment.segHeight)
        , segThreshold(segment.segThreshold)
        , borderColor(segment.borderColor)
        , segColor(segment.segColor)
    {
        checkERR(DeviceMalloc(&dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float)));
        checkERR(DeviceMemcpy(dSeg, segment.dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float),
                            kDeviceMemcpyDeviceToDevice));
    }

    NVCVSegment &operator=(const NVCVSegment &) = delete;

    ~NVCVSegment()
    {
        if (dSeg != nullptr)
        {
            checkERR(DeviceFree(dSeg));
            dSeg = nullptr;
        }
    };
};

class NVCVPolyLine
{
public:
    int32_t      *hPoints = nullptr; // Host pointer for polyline points' xy, cannot be nullptr.
                                     // Array length: 2 * numPoints.
                                     // Format : X0, Y0, X1, Y1, ..., Xk, Yk, ...
    int32_t      *dPoints = nullptr; // Device pointer for polyline points' xy.
                                     // Can be nullptr only if fillColor.a == 0.
                                     // Array length: 2 * numPoints.
                                     // Format: X0, Y0, X1, Y1, ..., Xk, Yk, ...
    int32_t       numPoints;         // Number of polyline points.
    int32_t       thickness;         // Polyline thickness.
    bool          isClosed;          // Connect p(0) to p(n-1) or not.
    NVCVColorRGBA borderColor;       // Line color of polyline border.
    NVCVColorRGBA fillColor;         // Fill color of poly fill area.
    bool          interpolation;     // Default: true

    NVCVPolyLine(int32_t *_hPoints, int32_t _numPoints, int32_t _thickness, bool _isClosed, NVCVColorRGBA _borderColor,
                 NVCVColorRGBA _fillColor, bool _interpolation)
        : numPoints(_numPoints)
        , thickness(_thickness)
        , isClosed(_isClosed)
        , borderColor(_borderColor)
        , fillColor(_fillColor)
        , interpolation(_interpolation)
    {
        hPoints = (int *)malloc(numPoints * 2 * sizeof(int));
        checkERR(DeviceMalloc(&dPoints, 2 * numPoints * sizeof(int)));

        std::copy_n(_hPoints, 2 * numPoints, hPoints);
        checkERR(DeviceMemcpy(dPoints, _hPoints, 2 * numPoints * sizeof(int), kDeviceMemcpyHostToDevice));
    }

    NVCVPolyLine(const NVCVPolyLine &pl)
        : numPoints(pl.numPoints)
        , thickness(pl.thickness)
        , isClosed(pl.isClosed)
        , borderColor(pl.borderColor)
        , fillColor(pl.fillColor)
        , interpolation(pl.interpolation)
    {
        hPoints = (int *)malloc(numPoints * 2 * sizeof(int));
        checkERR(DeviceMalloc(&dPoints, 2 * numPoints * sizeof(int)));

        std::copy_n(pl.hPoints, 2 * numPoints, hPoints);
        checkERR(DeviceMemcpy(dPoints, pl.dPoints, 2 * numPoints * sizeof(int), kDeviceMemcpyDeviceToDevice));
    }

    NVCVPolyLine &operator=(const NVCVPolyLine &) = delete;

    ~NVCVPolyLine()
    {
        if (hPoints != nullptr)
        {
            free(hPoints);
            hPoints = nullptr;
        }
        if (dPoints != nullptr)
        {
            checkERR(DeviceFree(dPoints));
            dPoints = nullptr;
        }
    };
};

class NVCVClock
{
public:
    NVCVClockFormat clockFormat;    // Pre-defined clock format.
    long            time;           // Clock time.
    int32_t         fontSize;       // Font size.
    const char     *font = nullptr; // Font name.
    NVCVPointI      tlPos;          // Top-left corner point, \ref NVCVPointI.
    NVCVColorRGBA   fontColor;      // Font color of the text.
    NVCVColorRGBA   bgColor;        // Background color of text box.

    NVCVClock(NVCVClockFormat _clockFormat, long _time, int32_t _fontSize, const char *_font, NVCVPointI _tlPos,
              NVCVColorRGBA _fontColor, NVCVColorRGBA _bgColor)
        : clockFormat(_clockFormat)
        , time(_time)
        , fontSize(_fontSize)
        , tlPos(_tlPos)
        , fontColor(_fontColor)
        , bgColor(_bgColor)
    {
        const size_t len      = std::char_traits<char>::length(_font);
        char        *tmp_font = (char *)malloc(len + 1);
        std::copy_n(_font, len + 1, tmp_font);
        font = tmp_font;
    }

    NVCVClock(const NVCVClock &clock)
        : clockFormat(clock.clockFormat)
        , time(clock.time)
        , fontSize(clock.fontSize)
        , tlPos(clock.tlPos)
        , fontColor(clock.fontColor)
        , bgColor(clock.bgColor)
    {
        const size_t len      = std::char_traits<char>::length(clock.font);
        char        *tmp_font = (char *)malloc(len + 1);
        std::copy_n(clock.font, len + 1, tmp_font);
        font = tmp_font;
    }

    NVCVClock &operator=(const NVCVClock &clock)
    {
        if (this != &clock)
        {
            if (font != nullptr)
            {
                free((void *)font);
                font = nullptr;
            }
            *this = NVCVClock(clock);
        }
        return *this;
    };

    ~NVCVClock()
    {
        if (font != nullptr)
        {
            free((void *)font);
            font = nullptr;
        }
    };
};

class NVCVElement
{
public:
    NVCVElement(NVCVOSDType osd_type, const void *src);
    NVCVElement(const NVCVElement &)            = delete;
    NVCVElement &operator=(const NVCVElement &) = delete;
    ~NVCVElement();

    NVCVOSDType type();
    void       *ptr();
    // void assign(const void* src);

private:
    /*
        *  type:
        *      NVCV_OSD_RECT           -   \ref NVCVBndBoxI.
        *      NVCV_OSD_TEXT           -   \ref NVCVText.
        *      NVCV_OSD_SEGMENT        -   \ref NVCVSegment.
        *      NVCV_OSD_POINT          -   \ref NVCVPoint.
        *      NVCV_OSD_LINE           -   \ref NVCVLine.
        *      NVCV_OSD_POLYLINE       -   \ref NVCVPolyLine.
        *      NVCV_OSD_ROTATED_RECT   -   \ref NVCVRotatedBox.
        *      NVCV_OSD_CIRCLE         -   \ref NVCVCircle.
        *      NVCV_OSD_ARROW          -   \ref NVCVArrow.
        *      NVCV_OSD_CLOCK          -   \ref NVCVClock.
        */
    NVCVOSDType m_type; // OSD element type to draw.
    void       *m_data; // OSD element data pointer.
};

inline NVCVElement::NVCVElement(NVCVOSDType osd_type, const void *src)
    : m_type(osd_type)
{
    switch (m_type)
    {
    case NVCVOSDType::NVCV_OSD_RECT:
    {
        auto rect = NVCVBndBoxI(*(NVCVBndBoxI *)src);
        m_data    = new NVCVBndBoxI(rect);
        break;
    }
    case NVCVOSDType::NVCV_OSD_TEXT:
    {
        auto text = NVCVText(*(NVCVText *)src);
        m_data    = new NVCVText(text);
        break;
    }
    case NVCVOSDType::NVCV_OSD_SEGMENT:
    {
        auto segment = NVCVSegment(*(NVCVSegment *)src);
        m_data       = new NVCVSegment(segment);
        break;
    }
    case NVCVOSDType::NVCV_OSD_POINT:
    {
        auto point = NVCVPoint(*(NVCVPoint *)src);
        m_data     = new NVCVPoint(point);
        break;
    }
    case NVCVOSDType::NVCV_OSD_LINE:
    {
        auto line = NVCVLine(*(NVCVLine *)src);
        m_data    = new NVCVLine(line);
        break;
    }
    case NVCVOSDType::NVCV_OSD_POLYLINE:
    {
        auto pl = NVCVPolyLine(*(NVCVPolyLine *)src);
        m_data  = new NVCVPolyLine(pl);
        break;
    }
    case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
    {
        auto rb = NVCVRotatedBox(*(NVCVRotatedBox *)src);
        m_data  = new NVCVRotatedBox(rb);
        break;
    }
    case NVCVOSDType::NVCV_OSD_CIRCLE:
    {
        auto circle = NVCVCircle(*(NVCVCircle *)src);
        m_data      = new NVCVCircle(circle);
        break;
    }
    case NVCVOSDType::NVCV_OSD_ARROW:
    {
        auto arrow = NVCVArrow(*(NVCVArrow *)src);
        m_data     = new NVCVArrow(arrow);
        break;
    }
    case NVCVOSDType::NVCV_OSD_CLOCK:
    {
        auto clock = NVCVClock(*(NVCVClock *)src);
        m_data     = new NVCVClock(clock);
        break;
    }
    default:
        break;
    }
}

inline NVCVElement::~NVCVElement()
{
    switch (m_type)
    {
    case NVCVOSDType::NVCV_OSD_RECT:
    {
        NVCVBndBoxI *bndBox = (NVCVBndBoxI *)m_data;
        if (bndBox != nullptr)
        {
            delete (bndBox);
            bndBox = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_TEXT:
    {
        NVCVText *label = (NVCVText *)m_data;
        if (label != nullptr)
        {
            delete (label);
            label = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_SEGMENT:
    {
        NVCVSegment *segment = (NVCVSegment *)m_data;
        if (segment != nullptr)
        {
            delete (segment);
            segment = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_POINT:
    {
        NVCVPoint *point = (NVCVPoint *)m_data;
        if (point != nullptr)
        {
            delete (point);
            point = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_LINE:
    {
        NVCVLine *line = (NVCVLine *)m_data;
        if (line != nullptr)
        {
            delete (line);
            line = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_POLYLINE:
    {
        NVCVPolyLine *pl = (NVCVPolyLine *)m_data;
        if (pl != nullptr)
        {
            delete (pl);
            pl = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
    {
        NVCVRotatedBox *rb = (NVCVRotatedBox *)m_data;
        if (rb != nullptr)
        {
            delete (rb);
            rb = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_CIRCLE:
    {
        NVCVCircle *circle = (NVCVCircle *)m_data;
        if (circle != nullptr)
        {
            delete (circle);
            circle = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_ARROW:
    {
        NVCVArrow *arrow = (NVCVArrow *)m_data;
        if (arrow != nullptr)
        {
            delete (arrow);
            arrow = nullptr;
        }
        break;
    }
    case NVCVOSDType::NVCV_OSD_CLOCK:
    {
        NVCVClock *clock = (NVCVClock *)m_data;
        if (clock != nullptr)
        {
            delete (clock);
            clock = nullptr;
        }
        break;
    }
    default:
        break;
    }
}

inline NVCVOSDType NVCVElement::type()
{
    return m_type;
}

inline void *NVCVElement::ptr()
{
    return m_data;
}

class NVCVBlurBoxesImpl
{
public:
    NVCVBlurBoxesImpl(const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec);
    NVCVBlurBoxesImpl(const NVCVBlurBoxesImpl &)            = delete;
    NVCVBlurBoxesImpl &operator=(const NVCVBlurBoxesImpl &) = delete;
    ~NVCVBlurBoxesImpl();

    int32_t      batch() const;
    int32_t      numBoxesAt(int32_t b) const;
    NVCVBlurBoxI boxAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<NVCVBlurBoxI>> m_blurboxes_vec;
};

inline NVCVBlurBoxesImpl::NVCVBlurBoxesImpl(const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec)
{
    m_blurboxes_vec = blurboxes_vec;
}

inline NVCVBlurBoxesImpl::~NVCVBlurBoxesImpl()
{
    std::vector<std::vector<NVCVBlurBoxI>> tmp;
    m_blurboxes_vec.swap(tmp);
}

inline int32_t NVCVBlurBoxesImpl::batch() const
{
    return m_blurboxes_vec.size();
}

inline int32_t NVCVBlurBoxesImpl::numBoxesAt(int32_t b) const
{
    return m_blurboxes_vec[b].size();
}

inline NVCVBlurBoxI NVCVBlurBoxesImpl::boxAt(int32_t b, int32_t i) const
{
    return m_blurboxes_vec[b][i];
}

class NVCVBndBoxesImpl
{
public:
    NVCVBndBoxesImpl(const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec);
    NVCVBndBoxesImpl(const NVCVBndBoxesImpl &)            = delete;
    NVCVBndBoxesImpl &operator=(const NVCVBndBoxesImpl &) = delete;
    ~NVCVBndBoxesImpl();

    int32_t     batch() const;
    int32_t     numBoxesAt(int32_t b) const;
    NVCVBndBoxI boxAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<NVCVBndBoxI>> m_bndboxes_vec;
};

inline NVCVBndBoxesImpl::NVCVBndBoxesImpl(const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec)
{
    m_bndboxes_vec = bndboxes_vec;
}

inline NVCVBndBoxesImpl::~NVCVBndBoxesImpl()
{
    std::vector<std::vector<NVCVBndBoxI>> tmp;
    m_bndboxes_vec.swap(tmp);
}

inline int32_t NVCVBndBoxesImpl::batch() const
{
    return m_bndboxes_vec.size();
}

inline int32_t NVCVBndBoxesImpl::numBoxesAt(int32_t b) const
{
    return m_bndboxes_vec[b].size();
}

inline NVCVBndBoxI NVCVBndBoxesImpl::boxAt(int32_t b, int32_t i) const
{
    return m_bndboxes_vec[b][i];
}

class NVCVElementsImpl
{
public:
    NVCVElementsImpl(const std::vector<std::vector<std::shared_ptr<NVCVElement>>> &elements_vec);
    NVCVElementsImpl(const NVCVElementsImpl &)            = delete;
    NVCVElementsImpl &operator=(const NVCVElementsImpl &) = delete;
    ~NVCVElementsImpl();

    int32_t                      batch() const;
    int32_t                      numElementsAt(int32_t b) const;
    std::shared_ptr<NVCVElement> elementAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> m_elements_vec;
};

inline NVCVElementsImpl::NVCVElementsImpl(const std::vector<std::vector<std::shared_ptr<NVCVElement>>> &elements_vec)
{
    m_elements_vec = elements_vec;
}

inline NVCVElementsImpl::~NVCVElementsImpl()
{
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> tmp;
    m_elements_vec.swap(tmp);
}

inline int32_t NVCVElementsImpl::batch() const
{
    return m_elements_vec.size();
}

inline int32_t NVCVElementsImpl::numElementsAt(int32_t b) const
{
    return m_elements_vec[b].size();
}

inline std::shared_ptr<NVCVElement> NVCVElementsImpl::elementAt(int32_t b, int32_t i) const
{
    return m_elements_vec[b][i];
}

} // namespace cvcuda::priv

#endif // CVCUDA_TYPES_HPP
