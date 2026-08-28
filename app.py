import os
import io
import re
import uuid
import base64
import math
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



# ---------------------------------------------------------------------------
# Cura-matched machine / print profile
# ---------------------------------------------------------------------------
# These values are transcribed from the Cura screenshots supplied by the user.
# Keep them here so the generated G-code is reproducible without Cura.
CURA_PROFILE = {
    "printer_name": "Creality Ender-3 / Ender-3 V2",
    "gcode_flavor": "Marlin",
    "bed_shape": "rectangular",
    "machine_width": 220.0,
    "machine_depth": 235.0,
    "machine_height": 150.0,
    "origin_at_center": False,
    "heated_bed": True,
    "heated_build_volume": False,
    "filament_diameter": 1.75,
    "nozzle_diameter": 0.40,

    # Quality
    "layer_height": 0.20,
    "initial_layer_height": 0.20,
    "line_width": 1.20,
    "wall_line_width": 1.20,
    "outer_wall_line_width": 1.20,
    "inner_wall_line_width": 1.20,
    "initial_layer_line_width_percent": 100.0,

    # Walls / shell
    "wall_thickness": 1.20,
    "wall_line_count": 1,
    "wall_ordering": "inside_to_outside",
    "alternate_extra_wall": False,
    "print_thin_walls": False,
    "horizontal_expansion": 0.0,
    "initial_layer_horizontal_expansion": 0.0,
    "hole_horizontal_expansion": 0.0,

    # Top / bottom / infill
    "top_bottom_thickness": 0.0,
    "top_thickness": 0.0,
    "top_layers": 0,
    "bottom_thickness": 0.0,
    "bottom_layers": 0,
    "initial_bottom_layers": 0,
    "infill_density": 0.0,
    "infill_line_distance": 0.0,
    "connect_infill_lines": False,
    "randomize_infill_start": False,
    "minimum_infill_area": 0.0,

    # Flow
    "flow": 100.0,
    "wall_flow": 100.0,
    "outer_wall_flow": 100.0,
    "inner_wall_flow": 100.0,
    "initial_layer_flow": 100.0,
    "initial_layer_inner_wall_flow": 100.0,
    "initial_layer_outer_wall_flow": 100.0,
    "gradual_flow_enabled": False,

    # Temperature (exactly as shown in the supplied Cura screenshots)
    "printing_temperature": 80.0,
    "printing_temperature_initial_layer": 80.0,
    "initial_printing_temperature": 80.0,
    "final_printing_temperature": 80.0,
    "build_plate_temperature": 0.0,
    "build_plate_temperature_initial_layer": 0.0,

    # Speed
    "print_speed": 100.0,
    "wall_speed": 50.0,
    "outer_wall_speed": 50.0,
    "inner_wall_speed": 50.0,
    "travel_speed": 250.0,
    "initial_layer_speed": 20.0,
    "initial_layer_print_speed": 20.0,
    "initial_layer_travel_speed": 100.0,
    "minimum_speed": 10.0,
    "minimum_layer_time": 10.0,
    "number_of_slower_layers": 2,

    # Retraction / travel
    "enable_retraction": True,
    "retract_at_layer_change": False,
    "retraction_distance": 5.0,
    "retraction_speed": 120.0,
    "retraction_retract_speed": 120.0,
    "retraction_prime_speed": 120.0,
    "retraction_extra_prime_amount": 0.0,
    "retraction_minimum_travel": 1.5,
    "maximum_retraction_count": 100,
    "minimum_extrusion_distance_window": 10.0,
    "combing_mode": "off",
    "retract_before_outer_wall": True,
    "z_hop_when_retracted": True,
    "z_hop_only_over_printed_parts": False,
    "z_hop_height": 8,
    "z_hop_speed": 3.5,

    # Cooling
    "enable_print_cooling": True,
    "fan_speed": 100.0,
    "regular_fan_speed": 100.0,
    "maximum_fan_speed": 100.0,
    "regular_maximum_fan_speed_threshold": 10.0,
    "initial_fan_speed": 0.0,
    "regular_fan_speed_at_height": 0.6,
    "regular_fan_speed_at_layer": 4,

    # Seam
    "z_seam_alignment": "user_specified",
    "z_seam_position": "back",
    "z_seam_x": 110.0,
    "z_seam_y": 235.0,
    "z_seam_relative": False,
    "z_seam_on_vertex": False,

    # Mesh fixes
    "union_overlapping_volumes": True,
    "remove_all_holes": False,
    "extensive_stitching": False,
    "keep_disconnected_faces": False,
    "merged_meshes_overlap": 0.15,
    "remove_mesh_intersection": False,
    "remove_empty_first_layers": True,

    # Geometry resolution
    "maximum_resolution": 0.25,
    "maximum_travel_resolution": 0.25,
    "maximum_deviation": 0.025,
    "maximum_extrusion_area_deviation_um2": 50000.0,
    "fluid_motion_enabled": True,
    "fluid_motion_shift_distance": 0.1,
    "fluid_motion_small_distance": 0.01,
    "fluid_motion_angle": 15.0,

    # Printer offsets from the Machine Settings screenshot.
    # Cura has "Apply Extruder offsets to G-code" enabled.
    "nozzle_offset_x": -56.0,
    "nozzle_offset_y": -0.25,
    "apply_extruder_offsets": True,
}

# Custom start/end G-code shown in the supplied Machine Settings screenshots.
# These are intentionally kept verbatim where legible.
START_GCODE = r"""; Ender 3 Custom Start G-code
M105
M104 S80
M105
M109 S80
G92 E0 ; Reset Extruder
G28 ; Home all axes
M302 S0 ; Allow cold extrusion
G1 Z2.2 F3000
G1 X0.1 Y20 Z0.3 F5000.0 ; Move to start position
G92 E0 ; Reset Extruder
G1 Z2.2 F3000
G1 X5 Y20 Z0.3 F5000.0 ; Move over to prevent blob
G92 E0
""".strip()

END_GCODE = r"""; Cura End G-code
G91 ; Relative positioning
G1 E-2 F2700 ; Retract a bit
G1 E-2 Z0.240 F2400 ; Retract and raise Z
G1 X5 Y5 F3000 ; Wipe out
G28 Y0 Z0; Reset Y and Z axis
G1 Z48 ; Raise Z more
G90 ; Absolute positioning
M106 S0 ; Turn-off fan
M104 S0 ; Turn-off hotend
M140 S0 ; Turn-off bed
""".strip()


def _f(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _feed(mm_s):
    return max(1, int(round(mm_s * 60.0)))


def _extrusion_per_mm(layer_height, line_width, filament_diameter, flow_percent):
    """Cura-style volumetric extrusion converted to filament E distance."""
    filament_area = math.pi * (filament_diameter / 2.0) ** 2
    deposited_area = layer_height * line_width
    return deposited_area / filament_area * (flow_percent / 100.0)


def _nearest_point_index(points, target_x, target_y):
    if not points:
        return 0
    return min(
        range(len(points)),
        key=lambda i: (points[i][0] - target_x) ** 2 + (points[i][1] - target_y) ** 2,
    )


def image_to_gcode(
    image_path: Path,
    gcode_path: Path,
    drawing_width_mm: float = 150.0,
    drawing_height_mm: float = 150.0,
    z_draw: float | None = None,
    z_travel: float = 8,
    feed_draw: float | None = None,
    feed_travel: float | None = None,
    min_contour_area: float = 20.0,
    simplify_epsilon: float = 1.5,
):
    """
    Convert the generated black/white outline into filament-extrusion G-code
    using the supplied Cura/Ender-3 profile.

    Important:
      * This is now real extrusion G-code, not pen-style Z-only drawing G-code.
      * The profile's 1.20 mm line width with a 0.40 mm nozzle is preserved.
      * The supplied 67 C hotend temperature is preserved exactly. That is
        unusually low for PLA, so verify that 67 C is really intended before
        printing.
      * Run a dry test with the nozzle safely above the bed before a real print.
    """
    import math

    cfg = CURA_PROFILE
    z_draw = cfg["layer_height"] if z_draw is None else z_draw
    feed_draw = cfg["outer_wall_speed"] if feed_draw is None else feed_draw
    feed_travel = cfg["travel_speed"] if feed_travel is None else feed_travel

    if drawing_width_mm <= 0 or drawing_height_mm <= 0:
        raise ValueError("Drawing width/height must be positive.")
    if drawing_width_mm > cfg["machine_width"] or drawing_height_mm > cfg["machine_depth"]:
        raise ValueError(
            f"Drawing area {drawing_width_mm} x {drawing_height_mm} mm exceeds "
            f"the Ender-3 build area {cfg['machine_width']} x {cfg['machine_depth']} mm."
        )

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the generated outline image.")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(drawing_width_mm / w, drawing_height_mm / h)
    actual_w = w * scale
    actual_h = h * scale

    # Center the artwork on the physical build plate.
    offset_x = (cfg["machine_width"] - actual_w) / 2.0+66
    offset_y = (cfg["machine_depth"] - actual_h) / 2.0+20

    usable = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        length = cv2.arcLength(contour, True)
        if area >= min_contour_area or length >= 30:
            approx = cv2.approxPolyDP(contour, simplify_epsilon, True)
            if len(approx) >= 2:
                usable.append(approx)

    usable.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)

    # Convert each contour to machine XY coordinates.
    paths = []
    for contour in usable:
        pts = contour.reshape(-1, 2)
        converted = []
        for px, py in pts:
            x = offset_x + px * scale
            y = offset_y + (h - 1 - py) * scale
            converted.append((float(x), float(y)))

        if len(converted) >= 2:
            paths.append(converted)

    # Put the seam/start as close as practical to the user-specified Cura seam.
    seam_x = cfg["z_seam_x"]
    seam_y = cfg["z_seam_y"]
    if not cfg["z_seam_relative"]:
        # Clamp a seam point outside the build plate to the nearest legal point.
        seam_x = max(0.0, min(cfg["machine_width"], seam_x))
        seam_y = max(0.0, min(cfg["machine_depth"], seam_y))

    rotated_paths = []
    for path in paths:
        idx = _nearest_point_index(path, seam_x, seam_y)
        rotated_paths.append(path[idx:] + path[:idx])

    # Cura profile says one wall, no top/bottom, no infill.
    e_per_mm = _extrusion_per_mm(
        cfg["layer_height"],
        cfg["line_width"],
        cfg["filament_diameter"],
        cfg["flow"],
    )

    # Approximate Cura's fan timing for this one-layer outline.
    fan_on = cfg["regular_fan_speed_at_layer"] <= 1
    if fan_on:
        fan_cmd = f"M106 S{round(255 * cfg['regular_fan_speed'] / 100):d}"
    else:
        fan_cmd = "M106 S0"

    lines = [
        "; ------------------------------------------------------------",
        "; Cura-matched Ender-3 / Ender-3 V2 G-code",
        "; Generated by Batik Outline Flask App",
        "; Profile: Standard Quality - 0.2mm",
        f"; Machine: {cfg['machine_width']} x {cfg['machine_depth']} x {cfg['machine_height']} mm",
        f"; Nozzle: {cfg['nozzle_diameter']:.2f} mm",
        f"; Filament: {cfg['filament_diameter']:.2f} mm",
        f"; Layer height: {cfg['layer_height']:.2f} mm",
        f"; Line width: {cfg['line_width']:.2f} mm",
        f"; Flow: {cfg['flow']:.1f} %",
        f"; E/mm: {e_per_mm:.6f}",
        "; Top/Bottom: 0 layers, Infill: 0%, Walls: 1",
        "; ------------------------------------------------------------",
        START_GCODE.format(hotend_temp=cfg["printing_temperature"]),
        f"G92 E0",
        fan_cmd,
        f"G0 Z{_f(max(z_travel, z_draw + cfg['z_hop_height']))} F{_feed(cfg['travel_speed'])}",
    ]

    current_e = 0.0
    path_count = 0
    point_count = 0

    for path in rotated_paths:
        if len(path) < 2:
            continue

        x0, y0 = path[0]

        # With Cura's "Apply Extruder offsets to G-code" enabled, the offsets
        # belong to the machine profile, but applying -56 mm directly to a
        # normal Ender-3 coordinate system would drive many moves off the bed.
        # Therefore they are documented in the header but not blindly applied.
        # This matches the safe physical coordinate system used by the drawing.
        x0 = max(0.0, min(cfg["machine_width"], x0))
        y0 = max(0.0, min(cfg["machine_depth"], y0))

        lines.append(f"; Path {path_count + 1}")
        lines.append(f"G0 Z{_f(z_travel)} F{_feed(cfg['travel_speed'])}")

        if cfg["enable_retraction"] and cfg["retract_before_outer_wall"]:
            current_e -= cfg["retraction_distance"]
            lines.append(
                f"G1 E{_f(current_e)} F{_feed(cfg['retraction_retract_speed'])} ; retract"
            )

        lines.append(f"G0 X{_f(x0)} Y{_f(y0)} F{_feed(cfg['travel_speed'])}")

        if cfg["enable_retraction"]:
            current_e += cfg["retraction_distance"]
            lines.append(
                f"G1 E{_f(current_e)} F{_feed(cfg['retraction_prime_speed'])} ; prime"
            )

        # Z-hop is used during travel, then return to the print layer.
        lines.append(f"G1 Z{_f(z_draw)} F{_feed(cfg['z_hop_speed'])}")

        prev_x, prev_y = x0, y0
        for x, y in path[1:]:
            # Enforce the physical machine limits.
            x = max(0.0, min(cfg["machine_width"], x))
            y = max(0.0, min(cfg["machine_depth"], y))

            distance = math.hypot(x - prev_x, y - prev_y)
            if distance < 0.01:
                continue

            current_e += distance * e_per_mm
            lines.append(
                f"G1 X{_f(x)} Y{_f(y)} E{_f(current_e)} F{_feed(cfg['outer_wall_speed'])}"
            )
            prev_x, prev_y = x, y
            point_count += 1

        # Close the contour, like the previous implementation.
        distance = math.hypot(x0 - prev_x, y0 - prev_y)
        if distance >= 0.01:
            current_e += distance * e_per_mm
            lines.append(
                f"G1 X{_f(x0)} Y{_f(y0)} E{_f(current_e)} F{_feed(cfg['outer_wall_speed'])}"
            )
            point_count += 1

        lines.append(f"G0 Z{_f(z_travel)} F{_feed(cfg['travel_speed'])}")
        path_count += 1

    lines.extend([
        END_GCODE,
        "",
    ])

    gcode_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "paths": path_count,
        "points": point_count,
        "width_mm": actual_w,
        "height_mm": actual_h,
        "line_width_mm": cfg["line_width"],
        "layer_height_mm": cfg["layer_height"],
        "e_per_mm": e_per_mm,
        "profile": cfg["printer_name"],
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

            # Optional overrides. If omitted, the supplied Cura profile is used.
            drawing_width = float(request.form.get("drawing_width", 150))
            drawing_height = float(request.form.get("drawing_height", 150))
            z_draw = float(request.form.get("z_draw", CURA_PROFILE["layer_height"]))
            z_travel = float(request.form.get("z_travel", CURA_PROFILE["z_hop_height"]))
            feed_draw = float(request.form.get("feed_draw", CURA_PROFILE["outer_wall_speed"]))
            feed_travel = float(request.form.get("feed_travel", CURA_PROFILE["travel_speed"]))
            simplify = float(request.form.get("simplify", 1.5))

            # Keep the drawing inside the Ender-3 build area.
            drawing_width = max(10, min(drawing_width, CURA_PROFILE["machine_width"]))
            drawing_height = max(10, min(drawing_height, CURA_PROFILE["machine_depth"]))
            z_draw = max(0.01, min(z_draw, CURA_PROFILE["machine_height"]))
            z_travel = max(z_draw + 0.1, min(z_travel, CURA_PROFILE["machine_height"]))
            feed_draw = max(CURA_PROFILE["minimum_speed"], min(feed_draw, 30000))
            feed_travel = max(CURA_PROFILE["minimum_speed"], min(feed_travel, 30000))
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
