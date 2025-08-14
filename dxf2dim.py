import ezdxf

# Load DXF file
doc = ezdxf.readfile("Part1.dxf")
msp = doc.modelspace()

# Conversion factor: if DXF is in millimeters
MM_TO_CM = 0.1

# Variables for bounding box
max_x = float('-inf')
min_x = float('inf')
max_y = float('-inf')
min_y = float('inf')

# Store circle diameters
circle_diameters_cm = []

# Process entities
for entity in msp:
    etype = entity.dxftype()

    if etype in ["LINE", "LWPOLYLINE", "POLYLINE"]:
        # Get points from entity
        points = []
        if hasattr(entity, "get_points"):
            try:
                points = entity.get_points()
            except:
                pass
        if hasattr(entity, "vertices"):
            points += [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if hasattr(entity, "dxf") and hasattr(entity.dxf, "start"):
            points += [(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]

        for p in points:
            x, y = p[0], p[1]
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)

    elif etype == "CIRCLE":
        # Circle radius from DXF (in mm), convert to cm
        radius_mm = entity.dxf.radius
        diameter_cm = (radius_mm * 2) * MM_TO_CM
        circle_diameters_cm.append(diameter_cm)

        # Include circle bounds in bounding box
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        max_x = max(max_x, cx + entity.dxf.radius)
        min_x = min(min_x, cx - entity.dxf.radius)
        max_y = max(max_y, cy + entity.dxf.radius)
        min_y = min(min_y, cy - entity.dxf.radius)

# Calculate dimensions in cm
width_cm = (max_x - min_x) * MM_TO_CM
height_cm = (max_y - min_y) * MM_TO_CM

# Output results
print(f"Width of drawing: {width_cm:.2f} cm")
print(f"Height of drawing: {height_cm:.2f} cm")
print("Circle diameters (cm):")
for i, d in enumerate(circle_diameters_cm, 1):
    print(f"  Circle {i}: {d:.2f} cm")
