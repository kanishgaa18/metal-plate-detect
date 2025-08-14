import matplotlib.pyplot as plt

# Plate dimensions in cm
width_cm = 50
height_cm = 50

# Hole details in cm
holes_cm = [
    {"center": (40, 10), "diameter": 10},
    {"center": (40, 40), "diameter": 10},
    {"center": (10, 40), "diameter": 10},
    {"center": (10, 10), "diameter": 10},
]

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Draw plate
plate = plt.Rectangle((0, 0), width_cm, height_cm, fill=None, edgecolor='black', linewidth=2)
ax.add_patch(plate)

# Draw holes
for hole in holes_cm:
    center = hole["center"]
    radius = hole["diameter"] / 2
    circle = plt.Circle(center, radius, fill=None, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)

# Set limits and remove axes
ax.set_xlim(-5, width_cm + 5)
ax.set_ylim(-5, height_cm + 5)
ax.axis('off')

# Save image
plt.savefig("metal_plate_cm.png", bbox_inches='tight', dpi=300)
plt.show()
