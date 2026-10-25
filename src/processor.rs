use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn default_black_point() -> f32 { 0.22 }
fn default_pre_contrast() -> f32 { 1.35 }
fn default_color_boost() -> f32 { 1.25 }
fn default_exposure() -> f32 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessOptions {
    /// Sensitivity to detect faint strokes (0.0 to 1.0, default: 0.52)
    #[serde(default = "default_sensitivity")]
    pub sensitivity: f32,
    /// Line contrast / blackness boost (1.0 to 3.0, default: 1.45)
    #[serde(default = "default_contrast")]
    pub contrast: f32,
    /// Paper illumination flattening strength (0.0 to 1.0, default: 0.85)
    #[serde(default = "default_shadow")]
    pub shadow_removal: f32,
    /// Suppress faint grid or quad-rule paper lines (0.0 to 1.0, default: 0.0)
    #[serde(default = "default_zero")]
    pub grid_suppression: f32,
    /// Despeckle radius (default: 0 to prevent line erosion)
    #[serde(default = "default_u32_zero")]
    pub despeckle: u32,
    /// Output mode: "transparent", "white", "blueprint", "inverted_transparent"
    #[serde(default = "default_output_mode")]
    pub output_mode: String,
    /// Line color mode: "original", "black", "graphite", "white"
    #[serde(default = "default_line_color_mode")]
    pub line_color_mode: String,
    /// Pre-removal black point depth / darkening
    #[serde(default = "default_black_point")]
    pub black_point: f32,
    /// Pre-removal contrast boost
    #[serde(default = "default_pre_contrast")]
    pub pre_contrast: f32,
    /// Pre-removal color saturation & vibrancy boost
    #[serde(default = "default_color_boost")]
    pub color_boost: f32,
    /// Pre-removal exposure adjustment
    #[serde(default = "default_exposure")]
    pub exposure: f32,
}

fn default_sensitivity() -> f32 { 0.52 }
fn default_contrast() -> f32 { 1.45 }
fn default_shadow() -> f32 { 0.85 }
fn default_zero() -> f32 { 0.0 }
fn default_u32_zero() -> u32 { 0 }
fn default_output_mode() -> String { "transparent".to_string() }
fn default_line_color_mode() -> String { "black".to_string() }

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            sensitivity: default_sensitivity(),
            contrast: default_contrast(),
            shadow_removal: default_shadow(),
            grid_suppression: default_zero(),
            despeckle: default_u32_zero(),
            output_mode: default_output_mode(),
            line_color_mode: default_line_color_mode(),
            black_point: default_black_point(),
            pre_contrast: default_pre_contrast(),
            color_boost: default_color_boost(),
            exposure: default_exposure(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct ImageProcessor;

impl ImageProcessor {
    /// Computes the bounding box of non-zero active strokes
    pub fn compute_bbox(img: &RgbaImage) -> Option<BoundingBox> {
        let (w, h) = img.dimensions();
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut found = false;

        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y);
                if p[3] > 15 {
                    found = true;
                    if x < min_x { min_x = x; }
                    if x > max_x { max_x = x; }
                    if y < min_y { min_y = y; }
                    if y > max_y { max_y = y; }
                }
            }
        }

        if found && max_x >= min_x && max_y >= min_y {
            Some(BoundingBox {
                min_x,
                min_y,
                max_x,
                max_y,
                width: max_x - min_x + 1,
                height: max_y - min_y + 1,
            })
        } else {
            None
        }
    }

    /// Processes an engineering drawing image with adaptive paper background removal.
    /// Internally normalizes paper exposure and deepens black point so that 100% of lines,
    /// dimensions, arrowheads, and annotations are mapped without breakage.
    pub fn process(image: &DynamicImage, options: &ProcessOptions) -> RgbaImage {
        let (width, height) = image.dimensions();
        let rgba_orig = image.to_rgba8();

        // 1. Calculate the estimated paper background illumination field
        let background = Self::estimate_background(&rgba_orig, width, height);

        // 2. Compute normalized ink density and alpha map
        let mut processed = RgbaImage::new(width, height);
        let raw_pixels: Vec<u8> = rgba_orig.into_raw();
        let bg_pixels = background.into_raw();
        let out_pixels = processed.as_mut();

        // Baseline calibrated thresholds for complete, unbroken line & dimension mapping
        let sens_offset = (options.sensitivity - 0.52) * 0.03;
        let t_low = (0.038 - sens_offset).clamp(0.015, 0.08);
        let t_high = (0.18 - sens_offset).clamp(0.08, 0.35);

        // Process row by row in parallel with rayon
        let row_stride = (width * 4) as usize;
        let rows: Vec<(usize, usize)> = (0..height as usize)
            .map(|y| (y * row_stride, (y + 1) * row_stride))
            .collect();

        let out_chunks: Vec<Vec<u8>> = rows
            .par_iter()
            .map(|&(start, end)| {
                let mut row_out = vec![0u8; end - start];
                for i in (0..(end - start)).step_by(4) {
                    let idx = start + i;
                    let r = raw_pixels[idx] as f32;
                    let g = raw_pixels[idx + 1] as f32;
                    let b = raw_pixels[idx + 2] as f32;

                    let bg_r = (bg_pixels[idx] as f32).max(1.0);
                    let bg_g = (bg_pixels[idx + 1] as f32).max(1.0);
                    let bg_b = (bg_pixels[idx + 2] as f32).max(1.0);

                    // Illumination normalized RGB (1.0 = clean background paper, automatically exposed)
                    let norm_r = (r / bg_r).min(1.0);
                    let norm_g = (g / bg_g).min(1.0);
                    let norm_b = (b / bg_b).min(1.0);

                    // Perceived luminance using standard photometric weights
                    let norm_lum = 0.299 * norm_r + 0.587 * norm_g + 0.114 * norm_b;

                    // Chromatic difference for colored ink/annotations
                    let chroma = (norm_r - norm_g).abs().max((norm_g - norm_b).abs()).max((norm_b - norm_r).abs());

                    // Total ink density: 0.0 = pure paper, 1.0 = pitch black ink
                    let mut ink_density = (1.0 - norm_lum).max(chroma * 0.75);

                    // Optional grid / faint paper artifact suppression
                    if options.grid_suppression > 0.0 {
                        let faint_cutoff = options.grid_suppression * 0.12;
                        if ink_density < faint_cutoff {
                            ink_density = (ink_density - faint_cutoff * 0.5).max(0.0) / (1.0 - faint_cutoff * 0.5);
                        }
                    }

                    // Compute smooth alpha with internal black point mapping
                    // Strict zero-cutoff: any density below t_low is strictly paper background (0 alpha, 0 color)
                    let (alpha_byte, final_alpha) = if ink_density <= t_low {
                        (0u8, 0.0f32)
                    } else if ink_density >= t_high {
                        (255u8, 1.0f32)
                    } else {
                        let t = (ink_density - t_low) / (t_high - t_low);
                        // Smoothstep curve for natural pencil / pen edge falloff
                        let a = (t * t * (3.0 - 2.0 * t) * options.contrast.max(1.0)).min(1.0);
                        if a < 0.03 {
                            (0u8, 0.0f32)
                        } else {
                            let byte = (a * 255.0).round().min(255.0) as u8;
                            (byte, a)
                        }
                    };

                    // Determine stroke color with internal black point deepening
                    let (stroke_r, stroke_g, stroke_b) = match options.line_color_mode.as_str() {
                        "original" => {
                            // "original": Keep original stroke color but deepen graphite into rich black
                            let boost = (1.0 - final_alpha * 0.85).max(0.0);
                            (
                                ((norm_r * boost * 255.0).clamp(0.0, 255.0)) as u8,
                                ((norm_g * boost * 255.0).clamp(0.0, 255.0)) as u8,
                                ((norm_b * boost * 255.0).clamp(0.0, 255.0)) as u8,
                            )
                        }
                        "graphite" => (45, 48, 52), // 2B mechanical pencil graphite
                        "white" => (245, 248, 252), // Blueprint CAD white
                        _ => (15, 18, 22), // "black": Deep technical drawing ink (default)
                    };

                    // Compose output pixel based on output mode
                    match options.output_mode.as_str() {
                        "white" => {
                            // Pure white background, high-contrast engineering print
                            let a = final_alpha;
                            let out_r = (stroke_r as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                            let out_g = (stroke_g as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                            let out_b = (stroke_b as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                            row_out[i] = out_r;
                            row_out[i + 1] = out_g;
                            row_out[i + 2] = out_b;
                            row_out[i + 3] = 255;
                        }
                        "blueprint" => {
                            // Classic engineering blueprint blue background: #0c2b54
                            let bp_r = 12.0;
                            let bp_g = 43.0;
                            let bp_b = 84.0;
                            let a = final_alpha;
                            let line_r = 230.0;
                            let line_g = 245.0;
                            let line_b = 255.0;
                            let out_r = (line_r * a + bp_r * (1.0 - a)).round() as u8;
                            let out_g = (line_g * a + bp_g * (1.0 - a)).round() as u8;
                            let out_b = (line_b * a + bp_b * (1.0 - a)).round() as u8;
                            row_out[i] = out_r;
                            row_out[i + 1] = out_g;
                            row_out[i + 2] = out_b;
                            row_out[i + 3] = 255;
                        }
                        "inverted_transparent" => {
                            // Crisp white strokes on 100% transparent background (Alpha 0, 0 Color)
                            if alpha_byte == 0 {
                                row_out[i] = 0;
                                row_out[i + 1] = 0;
                                row_out[i + 2] = 0;
                                row_out[i + 3] = 0;
                            } else {
                                row_out[i] = 255;
                                row_out[i + 1] = 255;
                                row_out[i + 2] = 255;
                                row_out[i + 3] = alpha_byte;
                            }
                        }
                        _ => {
                            // "transparent": Transparent background with STRICT 0 COLOR (RGBA: 0, 0, 0, 0)
                            if alpha_byte == 0 {
                                row_out[i] = 0;
                                row_out[i + 1] = 0;
                                row_out[i + 2] = 0;
                                row_out[i + 3] = 0;
                            } else {
                                row_out[i] = stroke_r;
                                row_out[i + 1] = stroke_g;
                                row_out[i + 2] = stroke_b;
                                row_out[i + 3] = alpha_byte;
                            }
                        }
                    }
                }
                row_out
            })
            .collect();

        // Flatten chunk results into processed buffer
        for (row_idx, chunk) in out_chunks.into_iter().enumerate() {
            let start = row_idx * row_stride;
            let end = start + chunk.len();
            out_pixels[start..end].copy_from_slice(&chunk);
        }

        processed
    }

    /// Fast, robust estimation of paper background illumination.
    /// Uses area-maximum downscaling + dilation to model paper lighting,
    /// shadows, and yellowing without allowing black strokes to contaminate the background.
    fn estimate_background(
        orig: &RgbaImage,
        width: u32,
        height: u32,
    ) -> RgbaImage {
        let scale = 8.max((width.min(height) / 30).min(16));
        let ds_w = (width / scale).max(16);
        let ds_h = (height / scale).max(16);

        // Area-max downscaling: each downscaled pixel samples the MAXIMUM (brightest paper)
        // in its neighborhood, effectively wiping out thin lines and text
        let mut ds_img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(ds_w, ds_h);
        for dy in 0..ds_h {
            let sy_start = dy * scale;
            let sy_end = (sy_start + scale).min(height);
            for dx in 0..ds_w {
                let sx_start = dx * scale;
                let sx_end = (sx_start + scale).min(width);

                let mut max_r = 0u8;
                let mut max_g = 0u8;
                let mut max_b = 0u8;

                for py in sy_start..sy_end {
                    for px in sx_start..sx_end {
                        let p = orig.get_pixel(px, py);
                        if p[0] > max_r { max_r = p[0]; }
                        if p[1] > max_g { max_g = p[1]; }
                        if p[2] > max_b { max_b = p[2]; }
                    }
                }
                ds_img.put_pixel(dx, dy, Rgba([max_r, max_g, max_b, 255]));
            }
        }

        // Morphological dilation (radius 3 = 7x7 max filter on downscaled grid)
        let radius = 3;
        let mut dilated = ds_img.clone();
        for y in 0..ds_h {
            let y_min = y.saturating_sub(radius);
            let y_max = (y + radius).min(ds_h - 1);
            for x in 0..ds_w {
                let x_min = x.saturating_sub(radius);
                let x_max = (x + radius).min(ds_w - 1);

                let mut max_r = 0u8;
                let mut max_g = 0u8;
                let mut max_b = 0u8;

                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        let p = ds_img.get_pixel(nx, ny);
                        if p[0] > max_r { max_r = p[0]; }
                        if p[1] > max_g { max_g = p[1]; }
                        if p[2] > max_b { max_b = p[2]; }
                    }
                }
                dilated.put_pixel(x, y, Rgba([max_r, max_g, max_b, 255]));
            }
        }

        // Bilinear interpolation back to full resolution in parallel
        let mut full_bg = RgbaImage::new(width, height);
        let ds_raw = dilated.into_raw();
        let row_stride = (width * 4) as usize;

        full_bg
            .as_mut()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let sy = (y as f32 / scale as f32).min((ds_h - 1) as f32);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min((ds_h - 1) as usize);
                let fy = sy - y0 as f32;

                for x in 0..(width as usize) {
                    let sx = (x as f32 / scale as f32).min((ds_w - 1) as f32);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min((ds_w - 1) as usize);
                    let fx = sx - x0 as f32;

                    let idx00 = (y0 * ds_w as usize + x0) * 4;
                    let idx10 = (y0 * ds_w as usize + x1) * 4;
                    let idx01 = (y1 * ds_w as usize + x0) * 4;
                    let idx11 = (y1 * ds_w as usize + x1) * 4;

                    let px_idx = x * 4;
                    for channel in 0..3 {
                        let v00 = ds_raw[idx00 + channel] as f32;
                        let v10 = ds_raw[idx10 + channel] as f32;
                        let v01 = ds_raw[idx01 + channel] as f32;
                        let v11 = ds_raw[idx11 + channel] as f32;

                        let top = v00 * (1.0 - fx) + v10 * fx;
                        let bot = v01 * (1.0 - fx) + v11 * fx;
                        let interp = top * (1.0 - fy) + bot * fy;

                        let val = interp.clamp(1.0, 255.0);
                        row[px_idx + channel] = val.round() as u8;
                    }
                    row[px_idx + 3] = 255;
                }
            });

        full_bg
    }
}
