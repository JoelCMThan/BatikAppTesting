import os
import io
import re
import uuid
import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, url_for
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATIK_PROMPT = """
Convert the supplied hand-drawn sketch into a SIMPLE BATIK WAX OUTLINE TEMPLATE.

Preserve the main subject, recognizable shape, pose, proportions, and important
features, but simplify the drawing substantially.

STYLE:
- Add Malaysian Floral patterns
- Beautify the original drawing, keep the original drawing alive
- clean black outlines on a pure white background
- thick, smooth, consistent lines
- simple coloring-book / batik tracing template
- flowing organic curves suitable for a batik canting tool
- prefer long continuous strokes over many short segments
- simplify complicated shapes into clear recognizable forms
- remove tiny details
- convert rough or shaky lines into smooth intentional curves
- close shapes wherever practical
- maintain clear separation between major regions
- minimal internal details
- generous spacing between neighbouring lines
- approximately 15-30 major enclosed regions where appropriate

STRICTLY AVOID:
- color
- grayscale shading
- shadows
- hatching
- texture
- gradients
- photorealism
- 3D effects
- background scenery
- tiny decorative details
- extremely thin lines
- disconnected sketch marks
- unnecessary double lines

TECHNICAL OUTPUT:
Create a flat, high-contrast outline suitable for later contour extraction and
conversion to XY G-code. Prioritize continuous geometry, smooth curves, closed
contours, and clear separation between lines. White background, black linework.
"""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_canvas_data(data_url: str) -> Path:
    match = re.match(r"data:image/(png|jpeg|jpg|webp);base64,(.+)", data_url, re.S)
    if not match:
        raise ValueError("Invalid drawing data.")
    raw = base64.b64decode(match.group(2))
    path = UPLOAD_FOLDER / f"{uuid.uuid4()}_drawing.png"
    Image.open(io.BytesIO(raw)).convert("RGB").save(path, "PNG")
    return path


def normalize_uploaded_image(file_storage) -> Path:
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    raw_path = UPLOAD_FOLDER / f"{uuid.uuid4()}_upload.{ext}"
    file_storage.save(raw_path)

    img = Image.open(raw_path).convert("RGB")
    png_path = UPLOAD_FOLDER / f"{uuid.uuid4()}_upload.png"
    img.save(png_path, "PNG")
    return png_path


def force_black_white(input_path: Path, output_path: Path, threshold: int = 205):
    """Convert generated art into a strict black/white PNG."""
    img = Image.open(input_path).convert("L")
    img = ImageOps.autocontrast(img)
    bw = img.point(lambda p: 255 if p >= threshold else 0, mode="1")
    bw.save(output_path, "PNG")


def image_to_gcode(
    image_path: Path,
    gcode_path: Path,
    drawing_width_mm: float = 150.0,
    drawing_height_mm: float = 150.0,
    z_draw: float = 0.2,
    z_travel: float = 2.0,
    feed_draw: float = 1200.0,
    feed_travel: float = 3000.0,
    min_contour_area: float = 20.0,
    simplify_epsilon: float = 1.5,
):
    """
    Convert a strict black-on-white outline into XY toolpath G-code.

    Black pixels become drawable contours.
    Z is raised during travel and lowered for drawing.

    IMPORTANT:
    Test with the tool removed / machine powered safely before real use.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the generated outline image.")

    # Black lines -> white foreground for OpenCV contour detection.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find all relevant contours.
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid image dimensions.")

    # Preserve aspect ratio within the requested drawing area.
    scale = min(drawing_width_mm / w, drawing_height_mm / h)
    actual_w = w * scale
    actual_h = h * scale

    # Center inside requested workspace.
    offset_x = (drawing_width_mm - actual_w) / 2.0
    offset_y = (drawing_height_mm - actual_h) / 2.0

    usable = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        length = cv2.arcLength(contour, True)

        # Area alone can reject thin open-ish lines, so allow sufficiently long paths.
        if area >= min_contour_area or length >= 30:
            approx = cv2.approxPolyDP(contour, simplify_epsilon, True)
            if len(approx) >= 2:
                usable.append(approx)

    # Prefer larger paths first.
    usable.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)

    lines = [
        "; Batik outline G-code",
        "; Generated by Batik Outline Flask App",
        "G21 ; units in millimeters",
        "G90 ; absolute positioning",
        f"G0 Z{z_travel:.3f} F{feed_travel:.0f}",
    ]

    path_count = 0
    point_count = 0

    for contour in usable:
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        # Convert image coordinates to machine coordinates.
        # Flip Y so the image appears upright in Cartesian coordinates.
        converted = []
        for px, py in pts:
            x = offset_x + px * scale
            y = offset_y + (h - 1 - py) * scale
            converted.append((x, y))

        x0, y0 = converted[0]

        # Travel with tool raised.
        lines.append(f"G0 Z{z_travel:.3f} F{feed_travel:.0f}")
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{feed_travel:.0f}")

        # Lower tool for drawing.
        lines.append(f"G1 Z{z_draw:.3f} F{feed_draw:.0f}")

        for x, y in converted[1:]:
            lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed_draw:.0f}")
            point_count += 1

        # Close contour explicitly.
        lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{feed_draw:.0f}")
        point_count += 1

        # Lift tool.
        lines.append(f"G0 Z{z_travel:.3f} F{feed_travel:.0f}")
        path_count += 1

    lines.extend([
        "G0 Z5.000 F3000",
        "G0 X0 Y0 F3000",
        "M2 ; end",
        "",
    ])

    gcode_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "paths": path_count,
        "points": point_count,
        "width_mm": actual_w,
        "height_mm": actual_h,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    input_image = None
    output_image = None
    raw_output_image = None
    gcode_file = None
    gcode_stats = None
    error = None

    if request.method == "POST":
        try:
            canvas_data = request.form.get("canvas_data", "").strip()
            upload = request.files.get("image")

            if canvas_data:
                source_path = save_canvas_data(canvas_data)
            elif upload and upload.filename:
                if not allowed_file(upload.filename):
                    raise ValueError("Only PNG, JPG, JPEG and WEBP files are supported.")
                source_path = normalize_uploaded_image(upload)
            else:
                raise ValueError("Draw something on the canvas or upload an image first.")

            input_image = url_for(
                "static",
                filename=f"uploads/{source_path.name}"
            )

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key."
                )

            with open(source_path, "rb") as image_file:
                result = client.images.edit(
                    model="gpt-image-2",
                    image=image_file,
                    prompt=BATIK_PROMPT,
                    size="1024x1024"
                )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            raw_name = f"{uuid.uuid4()}_batik_raw.png"
            raw_path = OUTPUT_FOLDER / raw_name
            raw_path.write_bytes(image_bytes)

            clean_name = f"{uuid.uuid4()}_batik_bw.png"
            clean_path = OUTPUT_FOLDER / clean_name
            force_black_white(raw_path, clean_path)

            # User-adjustable G-code parameters.
            drawing_width = float(request.form.get("drawing_width", 150))
            drawing_height = float(request.form.get("drawing_height", 150))
            z_draw = float(request.form.get("z_draw", 0.2))
            z_travel = float(request.form.get("z_travel", 2.0))
            feed_draw = float(request.form.get("feed_draw", 1200))
            feed_travel = float(request.form.get("feed_travel", 3000))
            simplify = float(request.form.get("simplify", 1.5))

            # Basic sanity limits.
            drawing_width = max(10, min(drawing_width, 1000))
            drawing_height = max(10, min(drawing_height, 1000))
            z_draw = max(-20, min(z_draw, 100))
            z_travel = max(-20, min(z_travel, 100))
            feed_draw = max(10, min(feed_draw, 30000))
            feed_travel = max(10, min(feed_travel, 30000))
            simplify = max(0.1, min(simplify, 20))

            gcode_name = f"{uuid.uuid4()}_batik.gcode"
            gcode_path = OUTPUT_FOLDER / gcode_name

            gcode_stats = image_to_gcode(
                clean_path,
                gcode_path,
                drawing_width_mm=drawing_width,
                drawing_height_mm=drawing_height,
                z_draw=z_draw,
                z_travel=z_travel,
                feed_draw=feed_draw,
                feed_travel=feed_travel,
                simplify_epsilon=simplify,
            )

            raw_output_image = url_for(
                "static",
                filename=f"outputs/{raw_name}"
            )
            output_image = url_for(
                "static",
                filename=f"outputs/{clean_name}"
            )
            gcode_file = url_for(
                "static",
                filename=f"outputs/{gcode_name}"
            )

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        input_image=input_image,
        output_image=output_image,
        raw_output_image=raw_output_image,
        gcode_file=gcode_file,
        gcode_stats=gcode_stats,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
