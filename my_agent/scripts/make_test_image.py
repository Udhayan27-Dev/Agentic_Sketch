import os
import sys
from PIL import Image, ImageDraw, ImageFont


def draw_arrow(draw, p1, p2, arrow_len=8, fill="black", width=1):
    """Draw a line between p1 and p2 with arrowheads at both ends."""
    draw.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2

    if y1 == y2:  # Horizontal
        # Left arrowhead at p1 pointing left
        draw.polygon([(x1, y1), (x1 + arrow_len, y1 - 4), (x1 + arrow_len, y1 + 4)], fill=fill)
        # Right arrowhead at p2 pointing right
        draw.polygon([(x2, y2), (x2 - arrow_len, y2 - 4), (x2 - arrow_len, y2 + 4)], fill=fill)
    elif x1 == x2:  # Vertical
        # Top arrowhead at p1 pointing up
        draw.polygon([(x1, y1), (x1 - 4, y1 + arrow_len), (x1 + 4, y1 + arrow_len)], fill=fill)
        # Bottom arrowhead at p2 pointing down
        draw.polygon([(x2, y2), (x2 - 4, y2 - arrow_len), (x2 + 4, y2 - arrow_len)], fill=fill)


def make_test_image(output_path="test_l_shape.png"):
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Scale: 15 pixels per mm
    # 24 mm wide = 360 px
    # 12 mm tall = 180 px
    # 6 mm wall = 90 px
    scale = 15
    ox, oy = 220, 450  # Origin (0,0) in canvas coords

    # L-Shape vertices in mm (CAD coords: origin bottom-left, y up)
    pts_mm = [
        (0.0, 0.0),
        (24.0, 0.0),
        (24.0, 6.0),
        (6.0, 6.0),
        (6.0, 12.0),
        (0.0, 12.0),
    ]

    # Convert to canvas coords (y down)
    pts_px = [(ox + int(x * scale), oy - int(y * scale)) for x, y in pts_mm]

    # Draw closed outline
    for i in range(len(pts_px)):
        p_start = pts_px[i]
        p_end = pts_px[(i + 1) % len(pts_px)]
        draw.line([p_start, p_end], fill="black", width=4)

    # Try to load default font
    try:
        font = ImageFont.load_default(size=16)
        title_font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Title
    draw.text((200, 40), "2D Technical Drawing: L-Shape", fill="black", font=title_font)
    draw.text((200, 70), "All dimensions are in mm", fill="#555555", font=font)

    # Dimension 1: 24 mm bottom total width
    dim_y = oy + 45
    draw.line([(pts_px[0][0], oy), (pts_px[0][0], dim_y + 10)], fill="#888888", width=1)
    draw.line([(pts_px[1][0], pts_px[1][1]), (pts_px[1][0], dim_y + 10)], fill="#888888", width=1)
    draw_arrow(draw, (pts_px[0][0], dim_y), (pts_px[1][0], dim_y), fill="#222222", width=2)
    draw.text((ox + 150, dim_y + 5), "24 mm", fill="black", font=font)

    # Dimension 2: 12 mm left total height
    dim_x = ox - 45
    draw.line([(ox, pts_px[0][1]), (dim_x - 10, pts_px[0][1])], fill="#888888", width=1)
    draw.line([(ox, pts_px[5][1]), (dim_x - 10, pts_px[5][1])], fill="#888888", width=1)
    draw_arrow(draw, (dim_x, pts_px[5][1]), (dim_x, pts_px[0][1]), fill="#222222", width=2)
    draw.text((dim_x - 60, oy - 100), "12 mm", fill="black", font=font)

    # Dimension 3: 6 mm step height (right side)
    dim_rx = pts_px[1][0] + 35
    draw.line([(pts_px[1][0], pts_px[1][1]), (dim_rx + 10, pts_px[1][1])], fill="#888888", width=1)
    draw.line([(pts_px[2][0], pts_px[2][1]), (dim_rx + 10, pts_px[2][1])], fill="#888888", width=1)
    draw_arrow(draw, (dim_rx, pts_px[2][1]), (dim_rx, pts_px[1][1]), fill="#222222", width=2)
    draw.text((dim_rx + 15, oy - 55), "6 mm", fill="black", font=font)

    # Dimension 4: 6 mm wall thickness (top horizontal)
    dim_ty = pts_px[5][1] - 35
    draw.line([(pts_px[5][0], pts_px[5][1]), (pts_px[5][0], dim_ty - 10)], fill="#888888", width=1)
    draw.line([(pts_px[4][0], pts_px[4][1]), (pts_px[4][0], dim_ty - 10)], fill="#888888", width=1)
    draw_arrow(draw, (pts_px[5][0], dim_ty), (pts_px[4][0], dim_ty), fill="#222222", width=2)
    draw.text((ox + 20, dim_ty - 22), "6 mm", fill="black", font=font)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Created test image at: {output_path}")
    return output_path


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "test_l_shape.png"
    make_test_image(out_file)
