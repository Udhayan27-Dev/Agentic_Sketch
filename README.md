# Engineering Drawing Background Remover - Autodesk CAD Environment

High-performance, multithreaded Rust background remover designed specifically for hand-drawn mechanical drawings, architectural floor plans, technical sketches, and engineering schematics. Renders cleaned drawings directly onto an **Autodesk model space CAD grid** environment.

## Key Features

- **Adaptive Paper Illumination Normalization**: Eliminates phone camera shadows, folds, and paper tint in ~15 ms.
- **Strict 0-Color Alpha Matting**: Background pixels are output as `RGBA: (0, 0, 0, 0)` with no white or hazy residue.
- **Autodesk CAD Model Space Grid**:
  - Continuous major and minor grid divisions identical to Autodesk AutoCAD / Fusion 360 drafting canvas.
  - Minimal Autodesk UCS Coordinate Triad (Red $+X$ and Green $+Y$ axes) in the corner.
  - Zero measurement clutter: no dimension arrows, rulers, or bounding boxes—strictly the clean CAD workspace grid.
  - Quick grid toggle with `F7` keyboard shortcut or toolbar button.
  - Theme toggles: Autodesk Light Grid (default), Autodesk Dark Grid, and Transparency Checkerboard.
- **Interactive Result Viewer**:
  - CAD Grid view, Split-screen comparison wipe slider, and Side-by-side mode.
  - Smooth pan and zoom (drag anywhere to pan, scroll wheel to zoom).
- **Download Options**:
  - **Download Cleaned Drawing (PNG)**: 100% 0-color transparent drawing (`RGBA: 0, 0, 0, 0`).
  - **Download on Autodesk Grid (PNG)**: Drawing rendered with the Autodesk CAD grid backdrop.

## Running the Application

```bash
cargo run --release
```

The server runs at:
```
http://127.0.0.1:8080
```
