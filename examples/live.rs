//! Live three-panel view: RGB, RealSense depth, and the fused depth.
//!
//!   cargo run --release --features cuda,live --example live
//!
//! Weights default to weights/dinov2_vits14.safetensors and weights/da2_head_vits.safetensors.
//! Close the window or press Esc to quit. Needs a RealSense and, for the cuda feature,
//! CUDA_COMPUTE_CAP set to the GPU's architecture at build time (110 on Thor).

use std::ffi::{c_char, c_void, CStr};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle::{DType, Device};
use depth2depth::{Config, Depth2Depth};
use minifb::{Key, Window, WindowOptions};

const FAR_M: f32 = 6.0;
const NEAR_M: f32 = 0.3;

extern "C" {
    fn d2d_cam_open(width: i32, height: i32, fps: i32, depth_scale: *mut f32, err: *mut c_char, err_len: i32) -> *mut c_void;
    fn d2d_cam_width(cam: *const c_void) -> i32;
    fn d2d_cam_height(cam: *const c_void) -> i32;
    fn d2d_cam_grab(cam: *mut c_void, rgb: *mut u8, depth: *mut u16, err: *mut c_char, err_len: i32) -> i32;
    fn d2d_cam_close(cam: *mut c_void);
}

struct Camera {
    raw: *mut c_void,
    width: usize,
    height: usize,
    depth_scale: f32,
}

impl Camera {
    fn open(width: i32, height: i32, fps: i32) -> Result<Self> {
        let mut err = [0 as c_char; 512];
        let mut depth_scale = 0f32;
        let raw = unsafe { d2d_cam_open(width, height, fps, &mut depth_scale, err.as_mut_ptr(), err.len() as i32) };
        if raw.is_null() {
            bail!("{}", c_message(&err));
        }
        let (width, height) = unsafe { (d2d_cam_width(raw) as usize, d2d_cam_height(raw) as usize) };
        Ok(Self { raw, width, height, depth_scale })
    }

    /// `Ok(false)` when the camera missed its deadline; the caller should try again.
    fn grab(&mut self, rgb: &mut [u8], depth: &mut [u16]) -> Result<bool> {
        let mut err = [0 as c_char; 512];
        let code = unsafe { d2d_cam_grab(self.raw, rgb.as_mut_ptr(), depth.as_mut_ptr(), err.as_mut_ptr(), err.len() as i32) };
        if code == 2 {
            return Ok(false);
        }
        if code != 0 {
            bail!("{}", c_message(&err));
        }
        Ok(true)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe { d2d_cam_close(self.raw) }
    }
}

fn c_message(err: &[c_char]) -> String {
    let text = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
    if text.is_empty() { "RealSense call failed".into() } else { text.into_owned() }
}

// Polynomial approximation of the turbo colormap (Google AI blog, 2019).
fn turbo(t: f32) -> [u8; 3] {
    let x = t.clamp(0.0, 1.0);
    let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn pack(rgb: [u8; 3]) -> u32 {
    (rgb[0] as u32) << 16 | (rgb[1] as u32) << 8 | rgb[2] as u32
}

fn depth_color(z: f32) -> u32 {
    if !(NEAR_M..=FAR_M).contains(&z) { 0 } else { pack(turbo(z / FAR_M)) }
}

// 5-wide, 7-tall glyphs, bit 4 is the left pixel.
fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        b'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        _ => [0; 7],
    }
}

fn draw_label(buf: &mut [u32], dst_w: usize, dst_h: usize, x0: usize, text: &[u8]) {
    let scale = 2usize;
    let origin_x = x0 + 8;
    let origin_y = 8;
    for (i, &ch) in text.iter().enumerate() {
        let g = glyph(ch);
        for row in 0..7 {
            for col in 0..5 {
                if g[row] & (1 << (4 - col)) == 0 {
                    continue;
                }
                for sy in 0..scale {
                    for sx in 0..scale {
                        let x = origin_x + i * 6 * scale + col * scale + sx;
                        let y = origin_y + row * scale + sy;
                        if x + 1 < dst_w && y + 1 < dst_h {
                            buf[(y + 1) * dst_w + x + 1] = 0;
                            buf[y * dst_w + x] = 0x00ff_ffff;
                        }
                    }
                }
            }
        }
    }
}

fn blit_color(dst: &mut [u32], dst_w: usize, origin_x: usize, panel_w: usize, panel_h: usize, src: &[u8], src_w: usize, src_h: usize) {
    for y in 0..panel_h {
        let sy = y * src_h / panel_h;
        let row = &mut dst[y * dst_w + origin_x..y * dst_w + origin_x + panel_w];
        for (x, px) in row.iter_mut().enumerate() {
            let sx = x * src_w / panel_w;
            let i = (sy * src_w + sx) * 3;
            *px = pack([src[i], src[i + 1], src[i + 2]]);
        }
    }
}

fn blit_depth(dst: &mut [u32], dst_w: usize, origin_x: usize, panel_w: usize, panel_h: usize, src: &[f32], src_w: usize, src_h: usize, color: impl Fn(f32) -> u32) {
    for y in 0..panel_h {
        let sy = y * src_h / panel_h;
        let row = &mut dst[y * dst_w + origin_x..y * dst_w + origin_x + panel_w];
        for (x, px) in row.iter_mut().enumerate() {
            let sx = x * src_w / panel_w;
            *px = color(src[sy * src_w + sx]);
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (dino, head) = match args.len() {
        1 => ("weights/dinov2_vits14.safetensors".to_string(), "weights/da2_head_vits.safetensors".to_string()),
        3 => (args[1].clone(), args[2].clone()),
        _ => bail!("usage: live [dinov2.safetensors da2_head.safetensors]"),
    };

    let device = Device::cuda_if_available(0)?;
    let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
    println!("loading model on {device:?} {dtype:?}");
    let mut d2d = Depth2Depth::new(&dino, &head, device, dtype, Config::default())?;

    println!("opening RealSense");
    let mut camera = Camera::open(848, 480, 30).context("RealSense")?;
    let (w, h) = (camera.width, camera.height);
    println!("stream {w}x{h}  depth scale {} m/unit", camera.depth_scale);

    let panel_w = 640;
    let panel_h = panel_w * h / w;
    let dst_w = panel_w * 3;
    let dst_h = panel_h;
    let mut window = Window::new(
        "depth2depth    RGB  |  RealSense  |  enhanced",
        dst_w,
        dst_h,
        WindowOptions { topmost: true, resize: false, ..WindowOptions::default() },
    )?;
    let mut pixels = vec![0u32; dst_w * dst_h];
    let mut rgb = vec![0u8; w * h * 3];
    let mut depth_raw = vec![0u16; w * h];
    let mut depth_m = vec![0f32; w * h];

    let mut frames = 0u32;
    let mut stamp = Instant::now();
    println!("viewer open — Esc closes it");
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if !camera.grab(&mut rgb, &mut depth_raw)? {
            println!("camera frame late; waiting");
            window.update();
            continue;
        }
        let scale = camera.depth_scale;
        for (dst, &raw) in depth_m.iter_mut().zip(&depth_raw) {
            *dst = raw as f32 * scale;
        }
        let t0 = Instant::now();
        let fusion = d2d.fuse(&rgb, &depth_m, h, w)?;
        let fuse_ms = t0.elapsed().as_secs_f64() * 1000.0;

        blit_color(&mut pixels, dst_w, 0, panel_w, panel_h, &rgb, w, h);
        blit_depth(&mut pixels, dst_w, panel_w, panel_w, panel_h, &depth_m, w, h, depth_color);
        blit_depth(&mut pixels, dst_w, 2 * panel_w, panel_w, panel_h, &fusion.fused, w, h, |z| pack(turbo(z / FAR_M)));
        for y in 0..dst_h {
            pixels[y * dst_w + panel_w - 1] = 0x00ff_ffff;
            pixels[y * dst_w + 2 * panel_w - 1] = 0x00ff_ffff;
        }
        draw_label(&mut pixels, dst_w, dst_h, 0, b"RGB");
        draw_label(&mut pixels, dst_w, dst_h, panel_w, b"RAW");
        draw_label(&mut pixels, dst_w, dst_h, 2 * panel_w, b"FUSED");

        frames += 1;
        if stamp.elapsed().as_secs_f32() >= 1.0 {
            let fps = frames as f64 / stamp.elapsed().as_secs_f64();
            frames = 0;
            stamp = Instant::now();
            let title = format!("depth2depth    RGB | RealSense | enhanced     {fps:.1} fps   fuse {fuse_ms:.0} ms   a {:.3} b {:+.3}", fusion.a, fusion.b);
            window.set_title(&title);
            println!("{fps:.1} fps   fuse {fuse_ms:.0} ms   a {:.3} b {:+.3}", fusion.a, fusion.b);
        }
        window.update_with_buffer(&pixels, dst_w, dst_h)?;
    }
    Ok(())
}
