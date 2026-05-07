# report_extractor/preprocessing/image_preprocessor.py

import cv2
import numpy as np
from PIL import Image, ExifTags
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality metrics for the input image."""
    sharpness: float        # Laplacian variance — higher = sharper
    contrast: float         # Standard deviation of pixel intensities
    brightness: float       # Mean pixel intensity
    is_blurry: bool
    is_low_contrast: bool
    is_overexposed: bool
    is_underexposed: bool
    noise_level: float
    overall_quality: str    # 'excellent', 'good', 'poor', 'unusable'
    needs_preprocessing: bool
    has_readable_text: bool
    is_document: bool


class ImagePreprocessor:
    """
    State-of-the-art preprocessing pipeline for medical report images.
    Handles both scanned documents and hand-taken phone photos.

    Pipeline:
       1. Quality assessment (sharpness, contrast, brightness, noise, exposure)
       2. Smart decision making (skip processing for excellent images)
       3. EXIF orientation correction (phone photos)
       4. Scanner border / dark edge removal
       5. Document detection & perspective correction (4-point transform)
       6. Robust deskewing (Hough line median angle)
       7. White balance correction (gray-world normalisation)
       8. Shadow & uneven lighting removal (morphological background subtraction)
       9. Adaptive denoising (noise-level aware, edge-preserving)
      10. Contrast enhancement in LAB colour space (CLAHE on L channel)
      11. Unsharp-mask text sharpening (only when blurry)
      12. Adaptive binarisation fallback (severely degraded scans)
      13. Smart resize to target resolution

    Outputs RGB (for VLM) by default; set output_grayscale=True for OCR.
    """

    def __init__(
        self,
        max_dimension: int = 2048,
        clahe_clip: float = 1.5,  # lowered from 2.0 for less aggressive enhancement
        clahe_tile: int = 16,     # increased from 8 for larger tiles = less local artifacts
        clahe_blend: float = 0.7, # blend factor (0-1): higher = more enhancement
        sharpness_threshold: float = 80.0,
        contrast_threshold: float = 35.0,
        deskew_angle_threshold: float = 1.5,   # balanced — avoids false rotations on straight documents
        output_grayscale: bool = False,
        skip_excellent_images: bool = True,
        raise_on_unusable: bool = False,
        adaptive_binarise_fallback: bool = False,  # enable for OCR on severely degraded scans
    ):
        self.max_dimension = max_dimension
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.clahe_blend = max(0.0, min(1.0, clahe_blend))  # clamp to [0, 1]
        self.sharpness_threshold = sharpness_threshold
        self.contrast_threshold = contrast_threshold
        self.deskew_angle_threshold = deskew_angle_threshold
        self.output_grayscale = output_grayscale
        self.skip_excellent_images = skip_excellent_images
        self.raise_on_unusable = raise_on_unusable
        self.adaptive_binarise_fallback = adaptive_binarise_fallback

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Run the full preprocessing pipeline and return a clean PIL image."""
        return self._run_pipeline(image)[0]

    def smart_preprocess(self, image: Image.Image) -> Tuple[Image.Image, QualityReport]:
        """
        Preprocess with quality-aware decision making.
        Returns both the processed image and the quality report.
        """
        return self._run_pipeline(image)

    def assess_quality(self, image: Image.Image) -> QualityReport:
        """Public helper — return quality metrics without preprocessing."""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self._assess_quality(cv_image)

    # ------------------------------------------------------------------ #
    #  CORE PIPELINE (single pass — avoids duplicate quality assessment)   #
    # ------------------------------------------------------------------ #

    def _run_pipeline(self, image: Image.Image) -> Tuple[Image.Image, QualityReport]:
        """Internal unified pipeline. Returns (processed_image, quality_report)."""

        # 0. EXIF orientation correction (phone photos)
        image = self._correct_exif_orientation(image)

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 1. Quality assessment
        quality = self._assess_quality(cv_image)
        logger.info(
            f"Quality — overall: {quality.overall_quality}, "
            f"sharpness: {quality.sharpness:.1f}, "
            f"contrast: {quality.contrast:.1f}, "
            f"brightness: {quality.brightness:.1f}, "
            f"noise: {quality.noise_level:.1f}, "
            f"blurry: {quality.is_blurry}, "
            f"low_contrast: {quality.is_low_contrast}, "
            f"overexposed: {quality.is_overexposed}, "
            f"underexposed: {quality.is_underexposed}, "
            f"has_text: {quality.has_readable_text}, "
            f"is_document: {quality.is_document}"
        )

        # Handle unusable images
        if quality.overall_quality == 'unusable':
            if self.raise_on_unusable:
                raise ValueError(
                    f"Image quality insufficient for processing. "
                    f"Sharpness: {quality.sharpness:.1f}, "
                    f"Contrast: {quality.contrast:.1f}, "
                    f"Brightness: {quality.brightness:.1f}"
                )
            logger.warning("Image quality unusable — returning original image")
            return self._convert_output(cv_image), quality

        # Skip most preprocessing for excellent images, but always resize
        if quality.overall_quality == 'excellent' and self.skip_excellent_images:
            logger.info("Image quality excellent — skipping preprocessing but applying resize")
            cv_image = self._resize(cv_image)
            return self._convert_output(cv_image), quality

        # 2. Scanner border / dark edge removal
        cv_image = self._remove_borders(cv_image)

        # 3. Document detection & perspective correction
        cv_image = self._detect_and_correct_perspective(cv_image)

        # 4. Deskew (before shadow removal for cleaner illumination estimate)
        cv_image = self._deskew(cv_image)

        # 5. White balance correction (normalise colour cast from phone flash / lighting)
        cv_image = self._white_balance(cv_image)

        # 6. Shadow / uneven-lighting removal
        cv_image = self._remove_shadows(cv_image)

        # 7. Adaptive denoise (strength scaled by measured noise)
        cv_image = self._denoise(cv_image, noise_level=quality.noise_level)

        # 8. Contrast enhancement in LAB colour space (only for poor quality)
        cv_image = self._enhance_contrast_lab(cv_image, quality=quality)

        # 9. Text sharpening (unsharp mask — only when blurry)
        if quality.is_blurry:
            cv_image = self._sharpen(cv_image)

        # 10. Adaptive binarisation fallback for severely degraded scans
        if self.adaptive_binarise_fallback and quality.overall_quality == 'poor':
            cv_image = self._adaptive_binarise(cv_image)

        # 11. Resize
        cv_image = self._resize(cv_image)

        return self._convert_output(cv_image), quality

    # ------------------------------------------------------------------ #
    #  HELPER — Output conversion                                         #
    # ------------------------------------------------------------------ #

    def _convert_output(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL with appropriate color space."""
        if self.output_grayscale:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            return Image.fromarray(gray)
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------ #
    #  STEP 0 — EXIF orientation correction (phone photos)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _correct_exif_orientation(image: Image.Image) -> Image.Image:
        """
        Apply EXIF orientation tag so the image is right-side-up.
        Phone cameras often store the image in sensor orientation and
        rely on the EXIF tag to display it correctly.  Without this,
        the rest of the pipeline may process a rotated document.
        """
        try:
            exif = image.getexif()
            if not exif:
                return image

            # Find the orientation tag id
            orientation_key = None
            for tag_id, tag_name in ExifTags.TAGS.items():
                if tag_name == 'Orientation':
                    orientation_key = tag_id
                    break

            if orientation_key is None or orientation_key not in exif:
                return image

            orientation = exif[orientation_key]

            transforms = {
                2: Image.FLIP_LEFT_RIGHT,
                3: Image.ROTATE_180,
                4: Image.FLIP_TOP_BOTTOM,
                5: Image.TRANSPOSE,
                6: Image.ROTATE_270,
                7: Image.TRANSVERSE,
                8: Image.ROTATE_90,
            }

            if orientation in transforms:
                image = image.transpose(transforms[orientation])
                logger.info(f"EXIF orientation corrected (tag={orientation})")
        except Exception as e:
            logger.debug(f"EXIF orientation check skipped: {e}")

        return image

    # ------------------------------------------------------------------ #
    #  STEP 1 — Quality assessment                                        #
    # ------------------------------------------------------------------ #

    def _assess_quality(self, cv_image: np.ndarray) -> QualityReport:
        """Comprehensive quality assessment of the input image."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Basic metrics
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))

        # Exposure detection
        is_overexposed = brightness > 230 or np.percentile(gray, 95) > 250
        is_underexposed = brightness < 50 or np.percentile(gray, 5) < 10

        # Noise estimation
        noise_level = self._estimate_noise(gray)

        # Text and document detection
        has_readable_text = self._has_readable_text(gray)
        is_document = self._is_document(cv_image)

        # Overall quality score
        overall_quality = self._calculate_quality_score(
            sharpness, contrast, brightness, is_overexposed, is_underexposed, noise_level
        )

        return QualityReport(
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            is_blurry=sharpness < self.sharpness_threshold,
            is_low_contrast=contrast < self.contrast_threshold,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            noise_level=noise_level,
            overall_quality=overall_quality,
            needs_preprocessing=overall_quality in ['poor', 'good'],
            has_readable_text=has_readable_text,
            is_document=is_document,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation method.
        Based on J. Immerkaer's "Fast Noise Variance Estimation" (1996).
        """
        h, w = gray.shape
        M = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])

        filtered = cv2.filter2D(gray.astype(np.float64), -1, M)
        # Exclude 1-pixel border (affected by filter padding, per Immerkaer's paper)
        filtered = filtered[1:-1, 1:-1]
        sigma = np.sum(np.abs(filtered))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
        return float(sigma)

    def _has_readable_text(self, gray: np.ndarray) -> bool:
        """
        Quick check if image contains readable text.
        Uses MSER (Maximally Stable Extremal Regions) for text detection.
        """
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            # Threshold for "enough text" — documents typically have 100+ regions
            return len(regions) > 100
        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return True  # Assume text is present on error

    def _is_document(self, cv_image: np.ndarray) -> bool:
        """
        Detect if image is a document vs an arbitrary photo.
        Checks for rectangular boundary OR high text density (scanned pages
        without visible borders still count as documents).
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_area = cv_image.shape[0] * cv_image.shape[1]

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Check largest contour for rectangular shape
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= img_area * 0.25:
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                if len(approx) == 4:
                    return True

        # Fallback: high text density ⇒ likely a full-page scan with no visible border
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            if len(regions) > 200:
                return True
        except Exception:
            pass

        return False

    def _calculate_quality_score(
        self,
        sharpness: float,
        contrast: float,
        brightness: float,
        overexposed: bool,
        underexposed: bool,
        noise: float,
    ) -> str:
        """
        Determine overall quality category using a penalty-point system
        so that a single moderate issue doesn't immediately condemn the image.

        Returns: 'excellent', 'good', 'poor', or 'unusable'
        """
        penalty = 0

        # --- Sharpness ---
        if sharpness < 10:
            penalty += 4          # completely unreadable
        elif sharpness < 30:
            penalty += 3
        elif sharpness < 50:
            penalty += 2
        elif sharpness < 100:
            penalty += 1

        # --- Contrast ---
        if contrast < 10:
            penalty += 4
        elif contrast < 20:
            penalty += 3
        elif contrast < 30:
            penalty += 2
        elif contrast < 40:
            penalty += 1

        # --- Exposure (overexposed / underexposed photos are often still salvageable) ---
        if overexposed:
            if brightness > 240:
                penalty += 3      # nearly white-out
            else:
                penalty += 2      # bright but recoverable
        if underexposed:
            if brightness < 30:
                penalty += 3      # nearly black
            else:
                penalty += 2      # dark but recoverable

        # --- Noise ---
        if noise > 30:
            penalty += 3
        elif noise > 15:
            penalty += 2
        elif noise > 8:
            penalty += 1

        # --- Brightness extremes (when not already flagged as exposure issue) ---
        if not overexposed and not underexposed:
            if brightness < 60 or brightness > 250:
                penalty += 1
            elif brightness < 90 or brightness > 210:
                penalty += 0.5

        # Map penalty → category
        if penalty >= 5:
            return 'unusable'
        if penalty >= 3:
            return 'poor'
        if penalty >= 1:
            return 'good'
        return 'excellent'

    # ------------------------------------------------------------------ #
    #  STEP 2 — Scanner border / dark edge removal                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remove_borders(cv_image: np.ndarray) -> np.ndarray:
        """
        Remove dark borders or scanner artifacts from edges of the image.
        Works by detecting the largest bright region (the actual page).
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Threshold to find page content (light areas)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours of bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv_image

        # Get bounding rectangle of the largest bright area
        largest = max(contours, key=cv2.contourArea)
        img_area = cv_image.shape[0] * cv_image.shape[1]

        # Only crop if the bright region is a meaningful portion of the image
        # but smaller than the full image (i.e., there IS a border to remove)
        contour_area = cv2.contourArea(largest)
        if contour_area < img_area * 0.3 or contour_area > img_area * 0.98:
            return cv_image

        x, y, w, h = cv2.boundingRect(largest)

        # Add small margin to avoid clipping content
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(cv_image.shape[1] - x, w + 2 * margin)
        h = min(cv_image.shape[0] - y, h + 2 * margin)

        cropped = cv_image[y:y + h, x:x + w]

        # Sanity: don't return a tiny crop
        if cropped.shape[0] < 100 or cropped.shape[1] < 100:
            return cv_image

        logger.info(f"Border removal: cropped from {cv_image.shape[:2]} to {cropped.shape[:2]}")
        return cropped

    # ------------------------------------------------------------------ #
    #  STEP 3 — Document detection & perspective correction                #
    # ------------------------------------------------------------------ #

    def _detect_and_correct_perspective(self, cv_image: np.ndarray) -> np.ndarray:
        """Find the document quadrilateral and apply a 4-point perspective warp."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv_image

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        img_area = cv_image.shape[0] * cv_image.shape[1]

        # Only proceed if the contour covers a significant portion of the image
        if cv2.contourArea(largest) < img_area * 0.25:
            return cv_image

        # Try multiple epsilon values for polygon approximation
        peri = cv2.arcLength(largest, True)
        for eps_factor in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
            if len(approx) == 4:
                # Reject non-convex quads (likely false detections from partial borders)
                if not cv2.isContourConvex(approx):
                    continue
                pts = approx.reshape(4, 2).astype(np.float32)
                logger.info(f"Perspective correction: document quad detected (eps={eps_factor})")
                return self._four_point_transform(cv_image, pts)

        logger.debug("Perspective correction: no document quadrilateral found — skipped")
        return cv_image

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Warp a quadrilateral defined by pts into a top-down rectangle."""
        # Order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        (tl, tr, br, bl) = rect
        width_bottom = np.linalg.norm(br - bl)
        width_top = np.linalg.norm(tr - tl)
        max_w = int(max(width_bottom, width_top))

        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        max_h = int(max(height_right, height_left))

        if max_w == 0 or max_h == 0:
            return image

        # Reject degenerate quads with extreme aspect ratios
        aspect = max_w / max_h
        if aspect < 0.3 or aspect > 3.0:
            logger.debug(f"Perspective correction: rejected degenerate quad (aspect={aspect:.2f})")
            return image

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_CUBIC)

    # ------------------------------------------------------------------ #
    #  STEP 4 — Deskew (Hough-line median angle)                          #
    # ------------------------------------------------------------------ #

    def _deskew(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Robust deskewing using probabilistic Hough lines.
        Uses the median angle of near-horizontal and near-vertical lines.
        Only rotates when the measured skew exceeds the threshold.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Probabilistic Hough — returns line segments, more precise
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=cv_image.shape[1] // 8,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            return cv_image

        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1:     # near-vertical → skip for angle calc
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            # Keep only near-horizontal lines (within 45° of horizontal)
            if abs(angle_deg) < 45:
                angles.append(angle_deg)

        if not angles:
            return cv_image

        median_angle = float(np.median(angles))

        if abs(median_angle) < self.deskew_angle_threshold:
            return cv_image

        logger.info(f"Deskew: rotating by {median_angle:.2f}°")
        h, w = cv_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Compute new bounding box so nothing is clipped
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        return cv2.warpAffine(
            cv_image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ------------------------------------------------------------------ #
    #  STEP 5 — White balance correction (gray-world)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _white_balance(cv_image: np.ndarray) -> np.ndarray:
        """
        Gray-world white balance.  Assumes the average colour of a well-lit
        document page should be neutral grey.  Corrects colour casts from
        warm / cool / fluorescent lighting and phone flash.
        """
        b, g, r = cv2.split(cv_image.astype(np.float64))

        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b == 0 or avg_g == 0 or avg_r == 0:
            logger.debug("White balance: skipped (zero-mean channel)")
            return cv_image

        b = np.clip(b * (avg_all / avg_b), 0, 255)
        g = np.clip(g * (avg_all / avg_g), 0, 255)
        r = np.clip(r * (avg_all / avg_r), 0, 255)

        logger.info(
            f"White balance: gray-world correction applied "
            f"(R={avg_r:.0f}, G={avg_g:.0f}, B={avg_b:.0f} → target={avg_all:.0f})"
        )
        return cv2.merge([b, g, r]).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  STEP 6 — Shadow & uneven-lighting removal                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remove_shadows(cv_image: np.ndarray) -> np.ndarray:
        """
        Gentle shadow removal that preserves document texture.
        Uses a large Gaussian blur (not morphological closing) to estimate
        the background illumination, then only corrects areas where genuine
        shadow gradients exist.  The result is blended with the original
        to avoid the washed-out / smudged look.
        """
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Estimate background illumination with a large Gaussian blur
        # (softer than morphological closing — preserves local paper texture)
        ksize = max(l_channel.shape[0], l_channel.shape[1]) // 6
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(ksize, 51)
        bg = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)

        # Compute difference between background and foreground
        # Only correct where there's a meaningful shadow (bg significantly brighter)
        shadow_mask = bg.astype(np.float32) - l_channel.astype(np.float32)
        shadow_mask = np.clip(shadow_mask, 0, 255)

        # If there's barely any shadow variation, skip entirely
        mean_shadow = float(np.mean(shadow_mask))
        if mean_shadow < 8:
            logger.debug("Shadow removal: no significant shadows detected — skipped")
            return cv_image

        # Scale correction strength by shadow severity (gentler for mild shadows)
        correction_strength = min(0.6, mean_shadow / 30.0)

        # Correct by adding back the shadow difference (additive, not divisive)
        corrected = np.clip(l_channel.astype(np.float32) + shadow_mask * correction_strength, 0, 255).astype(np.uint8)

        # Blend with original — less blending for mild corrections
        blend = min(0.5, correction_strength)
        l_result = cv2.addWeighted(corrected, blend, l_channel, 1.0 - blend, 0)

        lab_corrected = cv2.merge([l_result, a, b])
        logger.info(f"Shadow removal: correction applied (mean_shadow={mean_shadow:.1f}, strength={correction_strength:.2f}, kernel={ksize})")
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ #
    #  STEP 7 — Adaptive denoising (noise-level aware)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _denoise(cv_image: np.ndarray, noise_level: float = 5.0) -> np.ndarray:
        """
        Noise-adaptive denoising.
        - Low noise  (< 5):  light bilateral filter (fast, preserves all detail)
        - Medium noise (5-15): fastNlMeans (better text preservation)
        - High noise (> 15):  stronger fastNlMeans
        """
        if noise_level < 2:
            logger.debug(f"Denoise: skipped — negligible noise (noise_level={noise_level:.1f})")
            return cv_image

        if noise_level < 5:
            logger.info(f"Denoise: light bilateral filter (noise_level={noise_level:.1f})")
            return cv2.bilateralFilter(cv_image, d=5, sigmaColor=40, sigmaSpace=40)

        # fastNlMeansDenoisingColored is slower but dramatically better for
        # preserving text edges compared to bilateral at higher noise levels.
        h_param = min(int(noise_level * 0.8), 15)   # strength scales with noise
        template_window = 7
        search_window = 21
        logger.info(
            f"Denoise: fastNlMeansDenoising applied "
            f"(noise_level={noise_level:.1f}, h={h_param})"
        )
        return cv2.fastNlMeansDenoisingColored(
            cv_image, None, h_param, h_param, template_window, search_window,
        )

    # ------------------------------------------------------------------ #
    #  STEP 8 — Contrast enhancement in LAB colour space                  #
    # ------------------------------------------------------------------ #

    def _enhance_contrast_lab(self, cv_image: np.ndarray, quality: Optional[QualityReport] = None) -> np.ndarray:
        """
        Apply CLAHE to the L channel in LAB space (only for poor quality images).
        Skipped for excellent/good quality to avoid over-processing.
        """
        # Skip CLAHE for excellent quality and good quality without contrast issues
        if quality and quality.overall_quality == 'excellent':
            logger.debug(f"Contrast enhancement: skipped ({quality.overall_quality} quality)")
            return cv_image
        if quality and quality.overall_quality == 'good' and not quality.is_low_contrast:
            logger.debug(f"Contrast enhancement: skipped ({quality.overall_quality} quality, adequate contrast)")
            return cv_image
        
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l_channel_orig, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_tile, self.clahe_tile),
        )
        l_channel_enhanced = clahe.apply(l_channel_orig)

        # Blend enhanced with original to prevent over-processing
        if self.clahe_blend < 1.0:
            l_channel = cv2.addWeighted(
                l_channel_enhanced, self.clahe_blend,
                l_channel_orig, 1.0 - self.clahe_blend,
                0
            )
        else:
            l_channel = l_channel_enhanced

        quality_str = quality.overall_quality if quality else 'unknown'
        logger.info(
            f"Contrast enhancement: CLAHE applied in LAB space ({quality_str} quality) "
            f"(clip={self.clahe_clip}, tile={self.clahe_tile}x{self.clahe_tile}, "
            f"blend={self.clahe_blend:.1%})"
        )
        return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ #
    #  STEP 9 — Text sharpening (unsharp mask)                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sharpen(cv_image: np.ndarray) -> np.ndarray:
        """Unsharp-mask: sharpen text without amplifying noise too much."""
        logger.info("Sharpening: unsharp mask applied (amount=1.5, sigma=3)")
        blurred = cv2.GaussianBlur(cv_image, (0, 0), sigmaX=3)
        return cv2.addWeighted(cv_image, 1.5, blurred, -0.5, 0)

    # ------------------------------------------------------------------ #
    #  STEP 10 — Adaptive binarisation (severely degraded scans)           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adaptive_binarise(cv_image: np.ndarray) -> np.ndarray:
        """
        Sauvola-style adaptive thresholding for severely degraded documents
        (faded fax copies, thermal prints).  Converts to binary then back
        to BGR so downstream code doesn't break.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Sauvola thresholding approximation via adaptive Gaussian
        # blockSize must be odd and large enough to cover text stroke width
        block_size = max(gray.shape[0], gray.shape[1]) // 40
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(block_size, 11)  # minimum useful size

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 10,
        )

        logger.info(f"Adaptive binarisation: Gaussian threshold applied (block_size={block_size})")
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    #  STEP 11 — Smart resize                                             #
    # ------------------------------------------------------------------ #

    def _resize(self, cv_image: np.ndarray) -> np.ndarray:
        h, w = cv_image.shape[:2]
        max_side = max(h, w)
        if max_side <= self.max_dimension:
            logger.debug(f"Resize: not needed ({w}x{h} within {self.max_dimension}px limit)")
            return cv_image

        scale = self.max_dimension / max_side
        new_size = (int(w * scale), int(h * scale))
        logger.info(f"Resize: {w}x{h} → {new_size[0]}x{new_size[1]} (Lanczos4)")
        return cv2.resize(cv_image, new_size, interpolation=cv2.INTER_LANCZOS4)