// RealSense capture for the live example. Color and depth are aligned to the
// color frame; depth pixels are the camera's native z16 units (see depth_scale).

#include <librealsense2/rs.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

struct Camera {
    rs2::pipeline pipe;
    rs2::align align;
    float depth_scale;
    int width;
    int height;

    Camera() : align(RS2_STREAM_COLOR), depth_scale(0.001f), width(0), height(0) {}
};

static void set_err(char* err, int err_len, const char* message) {
    if (err && err_len > 0) std::snprintf(err, static_cast<size_t>(err_len), "%s", message);
}

static void copy_plane(const void* src, int stride, int width, int height, int bytes_per_pixel, void* dst) {
    const auto* in = static_cast<const uint8_t*>(src);
    auto* out = static_cast<uint8_t*>(dst);
    const int row = width * bytes_per_pixel;
    for (int y = 0; y < height; ++y) {
        std::memcpy(out + static_cast<size_t>(y) * row, in + static_cast<size_t>(y) * stride, row);
    }
}

extern "C" {

Camera* d2d_cam_open(int width, int height, int fps, float* depth_scale, char* err, int err_len) {
    Camera* cam = new Camera();
    try {
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8, fps);
        rs2::pipeline_profile profile = cam->pipe.start(cfg);
        cam->depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
        auto color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        cam->width = color.width();
        cam->height = color.height();
        *depth_scale = cam->depth_scale;
        for (int i = 0; i < 15; ++i) cam->pipe.wait_for_frames(5000);
        return cam;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        delete cam;
        return nullptr;
    }
}

int d2d_cam_width(const Camera* cam) { return cam->width; }
int d2d_cam_height(const Camera* cam) { return cam->height; }

int d2d_cam_grab(Camera* cam, uint8_t* rgb, uint16_t* depth, char* err, int err_len) {
    try {
        rs2::frameset frames = cam->align.process(cam->pipe.wait_for_frames(1000));
        rs2::video_frame color = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        if (!color || !depth_frame || color.get_width() != cam->width || color.get_height() != cam->height
            || depth_frame.get_width() != cam->width || depth_frame.get_height() != cam->height) {
            set_err(err, err_len, "aligned color and depth sizes do not match");
            return 1;
        }
        copy_plane(color.get_data(), color.get_stride_in_bytes(), cam->width, cam->height, 3, rgb);
        copy_plane(depth_frame.get_data(), depth_frame.get_stride_in_bytes(), cam->width, cam->height, 2, depth);
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        // A late frame is recoverable; anything else is not.
        return std::strstr(e.what(), "didn't arrive") ? 2 : 1;
    }
}

void d2d_cam_close(Camera* cam) {
    if (!cam) return;
    try {
        cam->pipe.stop();
    } catch (const std::exception&) {
    }
    delete cam;
}

}
