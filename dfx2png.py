import ezdxf
import matplotlib.pyplot as plt

def dxf_to_png(dxf_path, output_png):
    """
    Converts a DXF file to a PNG image.
    
    Args:
        dxf_path (str): Path to the input DXF file.
        output_png (str): Path for the output PNG image.
    """
    try:
        doc = ezdxf.readfile(dxf_path)
    except IOError:
        print(f"Error: DXF file '{dxf_path}' not found.")
        return
    except ezdxf.DXFStructureError:
        print(f"Error: Invalid DXF file structure for '{dxf_path}'.")
        return

    msp = doc.modelspace()
    fig, ax = plt.subplots()

    for entity in msp:
        if entity.dxftype() == 'LINE':
            # Extract start and end points
            start = entity.dxf.start
            end = entity.dxf.end
            ax.plot([start.x, end.x], [start.y, end.y], 'k-')
        
        elif entity.dxftype() == 'CIRCLE':
            # Extract center and radius
            center = entity.dxf.center
            radius = entity.dxf.radius
            circle = plt.Circle((center.x, center.y), radius, fill=False)
            ax.add_artist(circle)
        
        # Add other entity types (e.g., ARC, LWPOLYLINE) as needed

    # Set plot limits and aspect ratio for a clean rendering
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    ax.set_title(f"Rendering of '{dxf_path}'")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    # Save the plot to a PNG file
    fig.savefig(output_png, dpi=300)
    print(f"DXF file converted and saved to '{output_png}' successfully.")
    
    plt.close(fig) # Close the plot to free up memory

if __name__ == "__main__":
    dxf_file = "Part1.DXF"
    png_file = "Part1.png"
    dxf_to_png(dxf_file, png_file)