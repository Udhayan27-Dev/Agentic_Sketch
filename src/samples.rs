use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
use std::io::Cursor;

pub fn get_sample_drawing(name: &str) -> Option<(Vec<u8>, &'static str)> {
    match name {
        "gear_sketch" => {
            let img = generate_mechanical_gear_sketch();
            let mut bytes = Vec::new();
            let mut cursor = Cursor::new(&mut bytes);
            DynamicImage::ImageRgba8(img)
                .write_to(&mut cursor, ImageFormat::Png)
                .ok()?;
            Some((bytes, "image/png"))
        }
        "bracket_draft" => {
            let img = generate_flange_bracket_sketch();
            let mut bytes = Vec::new();
            let mut cursor = Cursor::new(&mut bytes);
            DynamicImage::ImageRgba8(img)
                .write_to(&mut cursor, ImageFormat::Png)
                .ok()?;
            Some((bytes, "image/png"))
        }
        _ => None,
    }
}

/// Generates a realistic hand-drawn mechanical spur gear & shaft drawing on paper with uneven lighting
fn generate_mechanical_gear_sketch() -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let w = 900u32;
    let h = 650u32;
    let mut img = ImageBuffer::new(w, h);

    // 1. Generate realistic uneven paper background (cream/aged paper with phone camera gradient shadow)
    for y in 0..h {
        let yf = y as f32 / h as f32;
        for x in 0..w {
            let xf = x as f32 / w as f32;

            // Gradient shadow from top-right to bottom-left + slight radial vignette
            let dist_corner = ((xf - 0.2).powi(2) + (yf - 0.2).powi(2)).sqrt();
            let shadow = (dist_corner * 45.0).min(50.0);
            let noise = ((x * 17 + y * 31) % 7) as f32;

            // Warm paper base: 242, 238, 226
            let r = (242.0 - shadow - noise).clamp(170.0, 255.0) as u8;
            let g = (238.0 - shadow * 1.05 - noise).clamp(165.0, 255.0) as u8;
            let b = (226.0 - shadow * 1.15 - noise).clamp(150.0, 255.0) as u8;

            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // Helper for drawing pencil lines with graphite texture
    let draw_pencil_line = |buffer: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
                            x0: f32,
                            y0: f32,
                            x1: f32,
                            y1: f32,
                            thickness: f32,
                            darkness: f32| {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1.0 {
            return;
        }
        let steps = (len * 2.0) as usize;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;

            let rad = (thickness * 1.5).ceil() as i32;
            for oy in -rad..=rad {
                for ox in -rad..=rad {
                    let px = (cx + ox as f32).round() as i32;
                    let py = (cy + oy as f32).round() as i32;
                    if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                        let dist = (ox as f32 * ox as f32 + oy as f32 * oy as f32).sqrt();
                        if dist <= thickness {
                            let edge_falloff = 1.0 - (dist / thickness).min(1.0);
                            let grain = 0.85 + (((px * 13 + py * 29) % 5) as f32 * 0.05);
                            let alpha = edge_falloff * darkness * grain * 0.8;

                            let cur = buffer.get_pixel(px as u32, py as u32);
                            // Graphite color ~ 40, 42, 45
                            let new_r = (cur[0] as f32 * (1.0 - alpha) + 40.0 * alpha) as u8;
                            let new_g = (cur[1] as f32 * (1.0 - alpha) + 42.0 * alpha) as u8;
                            let new_b = (cur[2] as f32 * (1.0 - alpha) + 45.0 * alpha) as u8;
                            buffer.put_pixel(px as u32, py as u32, Rgba([new_r, new_g, new_b, 255]));
                        }
                    }
                }
            }
        }
    };

    let draw_circle = |buffer: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
                       cx: f32,
                       cy: f32,
                       radius: f32,
                       thickness: f32,
                       darkness: f32| {
        let steps = (radius * 6.28) as usize;
        for s in 0..steps {
            let th = s as f32 * 6.28318 / steps as f32;
            let x = cx + radius * th.cos();
            let y = cy + radius * th.sin();
            let next_th = (s + 1) as f32 * 6.28318 / steps as f32;
            let nx = cx + radius * next_th.cos();
            let ny = cy + radius * next_th.sin();
            draw_pencil_line(buffer, x, y, nx, ny, thickness, darkness);
        }
    };

    // Mechanical Drawing Features:
    // Left: Spur Gear Front View
    let gcx = 280.0;
    let gcy = 320.0;

    // Centerlines (dash-dot)
    draw_pencil_line(&mut img, gcx - 180.0, gcy, gcx + 180.0, gcy, 0.9, 0.45);
    draw_pencil_line(&mut img, gcx, gcy - 180.0, gcx, gcy + 180.0, 0.9, 0.45);

    // Gear Pitch circle, Root circle, Outer circle
    draw_circle(&mut img, gcx, gcy, 140.0, 1.8, 0.9); // Tip circle
    draw_circle(&mut img, gcx, gcy, 125.0, 1.0, 0.5); // Pitch circle (fainter)
    draw_circle(&mut img, gcx, gcy, 105.0, 1.6, 0.85); // Root circle
    draw_circle(&mut img, gcx, gcy, 60.0, 1.8, 0.9);  // Hub
    draw_circle(&mut img, gcx, gcy, 35.0, 2.0, 0.95); // Bore hole

    // Gear teeth profile indications
    let num_teeth = 18;
    for i in 0..num_teeth {
        let angle = i as f32 * 6.28318 / num_teeth as f32;
        let r_in = 105.0;
        let r_out = 140.0;
        let x0 = gcx + r_in * angle.cos();
        let y0 = gcy + r_in * angle.sin();
        let x1 = gcx + r_out * angle.cos();
        let y1 = gcy + r_out * angle.sin();
        draw_pencil_line(&mut img, x0, y0, x1, y1, 1.7, 0.85);
    }

    // Keyway slot in bore
    draw_pencil_line(&mut img, gcx - 6.0, gcy - 35.0, gcx - 6.0, gcy - 44.0, 1.8, 0.9);
    draw_pencil_line(&mut img, gcx - 6.0, gcy - 44.0, gcx + 6.0, gcy - 44.0, 1.8, 0.9);
    draw_pencil_line(&mut img, gcx + 6.0, gcy - 44.0, gcx + 6.0, gcy - 35.0, 1.8, 0.9);

    // Right: Stepped Shaft Side Section View
    let sx = 520.0;
    let sy = 320.0;

    // Shaft centerline
    draw_pencil_line(&mut img, sx - 20.0, sy, sx + 320.0, sy, 0.9, 0.45);

    // Stepped Shaft Segments
    // Segment 1 (Bearing Seat): 540 to 600, diam 60 (y -30 to +30)
    draw_pencil_line(&mut img, sx, sy - 30.0, sx + 70.0, sy - 30.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx, sy + 30.0, sx + 70.0, sy + 30.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx, sy - 30.0, sx, sy + 30.0, 2.0, 0.9);

    // Segment 2 (Gear Seat): 610 to 710, diam 90 (y -45 to +45)
    draw_pencil_line(&mut img, sx + 70.0, sy - 45.0, sx + 180.0, sy - 45.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 70.0, sy + 45.0, sx + 180.0, sy + 45.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 70.0, sy - 45.0, sx + 70.0, sy - 30.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 70.0, sy + 30.0, sx + 70.0, sy + 45.0, 2.0, 0.9);

    // Segment 3 (Coupling End): 720 to 800, diam 45 (y -22 to +22)
    draw_pencil_line(&mut img, sx + 180.0, sy - 22.0, sx + 280.0, sy - 22.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 180.0, sy + 22.0, sx + 280.0, sy + 22.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 180.0, sy - 45.0, sx + 180.0, sy - 22.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 180.0, sy + 22.0, sx + 180.0, sy + 45.0, 2.0, 0.9);
    draw_pencil_line(&mut img, sx + 280.0, sy - 22.0, sx + 280.0, sy + 22.0, 2.0, 0.9);

    // Cross-hatching for section (45 degree lines)
    for hx in (0..100).step_by(12) {
        let x_start = sx + 72.0 + hx as f32;
        let x_end = x_start + 40.0;
        if x_start < sx + 178.0 {
            draw_pencil_line(&mut img, x_start, sy - 44.0, x_end.min(sx + 178.0), sy, 1.1, 0.55);
            draw_pencil_line(&mut img, x_start, sy, x_end.min(sx + 178.0), sy + 44.0, 1.1, 0.55);
        }
    }

    // Mechanical Dimension Lines & Leaders & Arrows
    // Dimension 1: Bore Diameter Ø35 H7
    draw_pencil_line(&mut img, gcx, gcy - 35.0, gcx + 80.0, gcy - 90.0, 1.2, 0.7);
    draw_pencil_line(&mut img, gcx + 80.0, gcy - 90.0, gcx + 170.0, gcy - 90.0, 1.2, 0.7);
    // Arrowhead
    draw_pencil_line(&mut img, gcx, gcy - 35.0, gcx + 12.0, gcy - 44.0, 1.5, 0.85);
    draw_pencil_line(&mut img, gcx, gcy - 35.0, gcx + 4.0, gcy - 48.0, 1.5, 0.85);

    // Dimension 2: Shaft Length 110 mm
    let dim_y = sy + 90.0;
    draw_pencil_line(&mut img, sx + 70.0, sy + 48.0, sx + 70.0, dim_y + 10.0, 1.0, 0.6);
    draw_pencil_line(&mut img, sx + 180.0, sy + 48.0, sx + 180.0, dim_y + 10.0, 1.0, 0.6);
    draw_pencil_line(&mut img, sx + 70.0, dim_y, sx + 180.0, dim_y, 1.2, 0.75);
    // Arrows at ends
    draw_pencil_line(&mut img, sx + 70.0, dim_y, sx + 80.0, dim_y - 4.0, 1.4, 0.8);
    draw_pencil_line(&mut img, sx + 70.0, dim_y, sx + 80.0, dim_y + 4.0, 1.4, 0.8);
    draw_pencil_line(&mut img, sx + 180.0, dim_y, sx + 170.0, dim_y - 4.0, 1.4, 0.8);
    draw_pencil_line(&mut img, sx + 180.0, dim_y, sx + 170.0, dim_y + 4.0, 1.4, 0.8);

    // Title Block boundary (bottom right)
    draw_pencil_line(&mut img, w as f32 - 260.0, h as f32 - 80.0, w as f32 - 30.0, h as f32 - 80.0, 1.8, 0.85);
    draw_pencil_line(&mut img, w as f32 - 260.0, h as f32 - 80.0, w as f32 - 260.0, h as f32 - 30.0, 1.8, 0.85);
    draw_pencil_line(&mut img, w as f32 - 260.0, h as f32 - 30.0, w as f32 - 30.0, h as f32 - 30.0, 1.8, 0.85);
    draw_pencil_line(&mut img, w as f32 - 30.0, h as f32 - 80.0, w as f32 - 30.0, h as f32 - 30.0, 1.8, 0.85);

    img
}

/// Generates a second mechanical sketch: Bracket mount with quad grid paper
fn generate_flange_bracket_sketch() -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let w = 850u32;
    let h = 600u32;
    let mut img = ImageBuffer::new(w, h);

    // 1. Grid Paper Background (Millimeter / 5mm Quad Grid with uneven lighting)
    for y in 0..h {
        let yf = y as f32 / h as f32;
        for x in 0..w {
            let xf = x as f32 / w as f32;
            let corner_shadow = (xf * 0.4 + yf * 0.6) * 35.0;

            let mut r = 248.0 - corner_shadow;
            let mut g = 246.0 - corner_shadow;
            let mut b = 240.0 - corner_shadow;

            // Faint quad paper grid lines every 25px
            let is_grid_x = x % 25 == 0;
            let is_grid_y = y % 25 == 0;
            if is_grid_x || is_grid_y {
                // Light cyan-grey grid ink
                r -= 22.0;
                g -= 14.0;
                b += 2.0;
            }

            img.put_pixel(x, y, Rgba([r.clamp(140.0, 255.0) as u8, g.clamp(140.0, 255.0) as u8, b.clamp(140.0, 255.0) as u8, 255]));
        }
    }

    let draw_pen_line = |buffer: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
                         x0: f32,
                         y0: f32,
                         x1: f32,
                         y1: f32,
                         thickness: f32,
                         darkness: f32| {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1.0 { return; }
        let steps = (len * 2.0) as usize;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let rad = (thickness * 1.5).ceil() as i32;
            for oy in -rad..=rad {
                for ox in -rad..=rad {
                    let px = (cx + ox as f32).round() as i32;
                    let py = (cy + oy as f32).round() as i32;
                    if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                        let dist = (ox as f32 * ox as f32 + oy as f32 * oy as f32).sqrt();
                        if dist <= thickness {
                            let falloff = 1.0 - (dist / thickness).min(1.0);
                            let alpha = falloff * darkness * 0.9;
                            let cur = buffer.get_pixel(px as u32, py as u32);
                            let new_r = (cur[0] as f32 * (1.0 - alpha) + 20.0 * alpha) as u8;
                            let new_g = (cur[1] as f32 * (1.0 - alpha) + 24.0 * alpha) as u8;
                            let new_b = (cur[2] as f32 * (1.0 - alpha) + 30.0 * alpha) as u8;
                            buffer.put_pixel(px as u32, py as u32, Rgba([new_r, new_g, new_b, 255]));
                        }
                    }
                }
            }
        }
    };

    // Draw L-bracket with gusset reinforcement
    let bx = 180.0;
    let by = 150.0;

    // Bracket profile
    draw_pen_line(&mut img, bx, by, bx + 450.0, by, 2.2, 0.95);
    draw_pen_line(&mut img, bx + 450.0, by, bx + 450.0, by + 60.0, 2.2, 0.95);
    draw_pen_line(&mut img, bx + 450.0, by + 60.0, bx + 70.0, by + 60.0, 2.2, 0.95);
    draw_pen_line(&mut img, bx + 70.0, by + 60.0, bx + 70.0, by + 320.0, 2.2, 0.95);
    draw_pen_line(&mut img, bx + 70.0, by + 320.0, bx, by + 320.0, 2.2, 0.95);
    draw_pen_line(&mut img, bx, by + 320.0, bx, by, 2.2, 0.95);

    // Gusset rib
    draw_pen_line(&mut img, bx + 70.0, by + 220.0, bx + 280.0, by + 60.0, 2.0, 0.9);

    // Mounting holes (4 bolt circle)
    let holes = [(bx + 35.0, by + 120.0), (bx + 35.0, by + 260.0), (bx + 180.0, by + 30.0), (bx + 380.0, by + 30.0)];
    for &(hx, hy) in &holes {
        // Bolt circle
        let r = 16.0;
        let steps = 40;
        for s in 0..steps {
            let th = s as f32 * 6.28318 / steps as f32;
            let nth = (s + 1) as f32 * 6.28318 / steps as f32;
            draw_pen_line(&mut img, hx + r * th.cos(), hy + r * th.sin(), hx + r * nth.cos(), hy + r * nth.sin(), 1.8, 0.9);
        }
        // Center mark
        draw_pen_line(&mut img, hx - 24.0, hy, hx + 24.0, hy, 0.9, 0.5);
        draw_pen_line(&mut img, hx, hy - 24.0, hx, hy + 24.0, 0.9, 0.5);
    }

    // Dimension lines
    draw_pen_line(&mut img, bx, by - 40.0, bx + 450.0, by - 40.0, 1.1, 0.7);
    draw_pen_line(&mut img, bx, by - 50.0, bx, by - 5.0, 0.9, 0.5);
    draw_pen_line(&mut img, bx + 450.0, by - 50.0, bx + 450.0, by - 5.0, 0.9, 0.5);

    img
}
