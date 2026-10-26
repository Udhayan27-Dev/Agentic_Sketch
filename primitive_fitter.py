import cv2
import numpy as np
import math
import time
from skimage.morphology import skeletonize

class LightweightPrimitiveFitter:
    def __init__(self):
        pass

    def fit_circle_kasa(self, pts):
        """Algebraic circle fitting on Nx2 points."""
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack((x, y, np.ones_like(x)))
        b = x**2 + y**2
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = float(c[0] / 2.0)
            cy = float(c[1] / 2.0)
            val = c[2] + cx**2 + cy**2
            if val <= 0:
                return None
            r = float(np.sqrt(val))
            dists = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)
            return cx, cy, r, float(np.mean(dists))
        except Exception:
            return None

    def fit_line_pca(self, pts):
        """Fit 2D line using Principal Component Analysis."""
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, -1]
        normal = np.array([-direction[1], direction[0]])
        dists = np.abs(np.dot(centered, normal))
        return mean, direction, float(np.mean(dists))

    def detect_arrowheads(self, stroke_mask):
        """Locate arrowheads (small solid acute components)."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stroke_mask)
        arrow_centroids = []
        arrow_mask = np.zeros_like(stroke_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # Arrows are small, solid, and slightly elongated or triangular
            if 20 <= area <= 320 and 5 <= w <= 45 and 5 <= h <= 45:
                solidity = area / max(w * h, 1)
                aspect = max(w, h) / max(min(w, h), 1)
                if solidity > 0.40 and aspect >= 1.15:
                    arrow_centroids.append(centroids[i])
                    arrow_mask[labels == i] = 255
        return arrow_centroids, arrow_mask

    def is_touching_arrows(self, pts, arrow_centroids, dist_thresh=20.0):
        """Check if any endpoint or point of the segment touches an arrowhead."""
        if not arrow_centroids or len(pts) == 0:
            return False
        # Check endpoints
        p_start = pts[0]
        p_end = pts[-1]
        for ac in arrow_centroids:
            d1 = math.hypot(p_start[0] - ac[0], p_start[1] - ac[1])
            d2 = math.hypot(p_end[0] - ac[0], p_end[1] - ac[1])
            if d1 < dist_thresh or d2 < dist_thresh:
                return True
        return False

    def process(self, img_bgr, stroke_color=(0, 0, 0), stroke_thickness=2):
        t0 = time.time()
        h, w = img_bgr.shape[:2]

        # 1. Extract clean stroke mask
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        diff = cv2.subtract(bg, gray)
        _, strokes = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(strokes) < 100:
            strokes = (gray < 140).astype(np.uint8) * 255

        # 2. Detect Arrowheads
        arrow_centroids, arrow_mask = self.detect_arrowheads(strokes)

        # 3. Skeletonize to 1-pixel centerlines
        skel = skeletonize(strokes > 0).astype(np.uint8) * 255

        # 4. Decompose skeleton at junctions (degree >= 3)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel)
        junctions = ((neighbors >= 13) & (skel > 0)).astype(np.uint8) * 255
        junctions_dilated = cv2.dilate(junctions, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        branch_free_skel = cv2.subtract(skel, junctions_dilated)

        num_segs, seg_labels, seg_stats, _ = cv2.connectedComponentsWithStats(branch_free_skel)

        # 5. Classify segments into Arcs and Lines
        arcs = []
        lines = []

        for i in range(1, num_segs):
            length = seg_stats[i, cv2.CC_STAT_AREA]
            if length < 15:
                continue

            ys, xs = np.where(seg_labels == i)
            pts = np.column_stack((xs, ys)).astype(np.float64)

            # Sort points sequentially along the path
            if len(pts) > 2:
                # Approximate path ordering by starting from one extreme
                p0 = pts[0]
                dists_from_p0 = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
                start_idx = np.argmax(dists_from_p0)
                # Sort by distance from start
                dists = np.hypot(pts[:, 0] - pts[start_idx, 0], pts[:, 1] - pts[start_idx, 1])
                order = np.argsort(dists)
                pts = pts[order]

            # Discard if it touches an arrowhead (dimension line / leader line)
            if self.is_touching_arrows(pts, arrow_centroids):
                continue

            cfit = self.fit_circle_kasa(pts)
            lfit = self.fit_line_pca(pts)

            # Arc criteria: low circle RMSE (<0.9) and reasonable radius
            if cfit and cfit[3] < 0.9 and 25 <= cfit[2] <= 1500 and (lfit is None or cfit[3] < lfit[2] * 0.8):
                arcs.append({
                    'pts': pts,
                    'cx': cfit[0],
                    'cy': cfit[1],
                    'r': cfit[2],
                    'rmse': cfit[3],
                    'len': length
                })
            elif lfit and lfit[2] < 1.0:
                lines.append({
                    'pts': pts,
                    'mean': lfit[0],
                    'dir': lfit[1],
                    'rmse': lfit[2],
                    'len': length,
                    'p_start': pts[0],
                    'p_end': pts[-1]
                })

        # 6. Cluster Arcs by Center & Radius (Bridge broken segments across dimensions)
        circle_clusters = []
        for arc in arcs:
            cx, cy, r = arc['cx'], arc['cy'], arc['r']
            matched = False
            for cl in circle_clusters:
                ccx, ccy, cr = cl['center_r']
                if math.hypot(cx - ccx, cy - ccy) < 20.0 and abs(r - cr) < 20.0:
                    cl['arcs'].append(arc)
                    cl['pts'].append(arc['pts'])
                    # Refit with all pooled points
                    all_pts = np.vstack(cl['pts'])
                    refit = self.fit_circle_kasa(all_pts)
                    if refit:
                        cl['center_r'] = (refit[0], refit[1], refit[2])
                        cl['rmse'] = refit[3]
                    matched = True
                    break
            if not matched:
                circle_clusters.append({
                    'center_r': (cx, cy, r),
                    'rmse': arc['rmse'],
                    'arcs': [arc],
                    'pts': [arc['pts']]
                })

        # Filter circles: Require significant support
        valid_circles = []
        for cl in circle_clusters:
            total_len = sum(a['len'] for a in cl['arcs'])
            cx, cy, r = cl['center_r']
            # Compute angular coverage of pooled points
            all_pts = np.vstack(cl['pts'])
            angles = np.arctan2(all_pts[:, 1] - cy, all_pts[:, 0] - cx)
            angles_deg = np.rad2deg(angles) % 360
            hist, _ = np.histogram(angles_deg, bins=36, range=(0, 360))
            coverage_deg = np.count_nonzero(hist > 0) * 10

            # Keep if total points > 80 or coverage > 90 deg or multiple broken arcs reconnected
            if total_len > 80 or coverage_deg >= 90 or len(cl['arcs']) >= 2:
                valid_circles.append({
                    'cx': cx,
                    'cy': cy,
                    'r': r,
                    'coverage_deg': coverage_deg,
                    'total_len': total_len,
                    'num_segments': len(cl['arcs']),
                    'rmse': cl['rmse'],
                    'is_full_circle': coverage_deg >= 180 or len(cl['arcs']) >= 2
                })

        # 7. Merge Collinear Lines
        valid_lines = []
        for l in lines:
            # Reject radial lines pointing at circle centers
            is_radial = False
            for vc in valid_circles:
                d_start = math.hypot(l['p_start'][0] - vc['cx'], l['p_start'][1] - vc['cy'])
                d_end = math.hypot(l['p_end'][0] - vc['cx'], l['p_end'][1] - vc['cy'])
                # If line is radial or terminates on the circle
                if (abs(d_start - vc['r']) < 8 or abs(d_end - vc['r']) < 8) and l['len'] < 60:
                    is_radial = True
                    break
            if not is_radial and l['len'] >= 25:
                valid_lines.append(l)

        # 8. Render mathematically continuous CAD primitives onto 0-color transparent canvas
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        color_rgba = (stroke_color[0], stroke_color[1], stroke_color[2], 255)

        for c in valid_circles:
            icx = int(round(c['cx']))
            icy = int(round(c['cy']))
            ir = int(round(c['r']))
            cv2.circle(canvas, (icx, icy), ir, color_rgba, stroke_thickness, cv2.LINE_AA)

        for l in valid_lines:
            pt1 = (int(round(l['p_start'][0])), int(round(l['p_start'][1])))
            pt2 = (int(round(l['p_end'][0])), int(round(l['p_end'][1])))
            cv2.line(canvas, pt1, pt2, color_rgba, stroke_thickness, cv2.LINE_AA)

        dt_ms = int((time.time() - t0) * 1000)

        stats = {
            'circles_reconstructed': len(valid_circles),
            'lines_reconstructed': len(valid_lines),
            'arrowheads_cleared': len(arrow_centroids),
            'processing_time_ms': dt_ms,
            'circles_details': [
                {'cx': round(c['cx'], 1), 'cy': round(c['cy'], 1), 'r': round(c['r'], 1),
                 'segments_joined': c['num_segments'], 'coverage_deg': c['coverage_deg']}
                for c in valid_circles
            ]
        }

        return canvas, stats

if __name__ == '__main__':
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else '/home/udhayan/.gemini/antigravity/brain/b58b2d10-872d-48a3-ac04-7530ea8d1f79/.user_uploaded/media_1792960188410_c5f175cd.png'
    img = cv2.imread(test_path)
    fitter = LightweightPrimitiveFitter()
    canvas, stats = fitter.process(img)
    cv2.imwrite('fitter_clean_output.png', canvas)
    print('Fitter results:', stats)
