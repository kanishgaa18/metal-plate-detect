import cv2
import numpy as np
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import tkinter as tk
from tkinter import filedialog
 
class DXFOverlay:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
       
        # DXF variables
        self.dxf_image = None
        self.scale = 1.0
        self.color = (0, 255, 0)  # Green in BGR
        self.position = (0, 0)
       
        # Create window
        cv2.namedWindow('DXF Overlay', cv2.WINDOW_NORMAL)
       
        # Add trackbars
        self.create_trackbars()
 
    def create_trackbars(self):
        cv2.createTrackbar('Scale', 'DXF Overlay', 100, 200, lambda x: None)
        cv2.createTrackbar('X Position', 'DXF Overlay', 0, 1000, lambda x: None)
        cv2.createTrackbar('Y Position', 'DXF Overlay', 0, 1000, lambda x: None)
 
    def browse_dxf_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select DXF File",
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")]
        )
        return file_path
 
    def render_dxf(self, dxf_path):
        try:
            # Load DXF file
            doc = ezdxf.readfile(dxf_path)
           
            # Create figure with transparent background
            fig = plt.figure(facecolor='none', edgecolor='none')
            ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
            ax.set_axis_off()
           
            # Render DXF
            context = RenderContext(doc)
            backend = MatplotlibBackend(ax)
            Frontend(context, backend).draw_layout(doc.modelspace(), finalize=True)
           
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
           
            # Convert to OpenCV image
            pil_img = Image.open(buf)
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
           
            # Apply color
            mask = img[:, :, 3] > 0
            for c in range(3):
                img[:, :, c] = np.where(mask, self.color[c], img[:, :, c])
               
            return img
           
        except Exception as e:
            print(f"Error rendering DXF: {e}")
            return None
 
    def update_overlay(self):
        # Get current trackbar values
        self.scale = cv2.getTrackbarPos('Scale', 'DXF Overlay') / 100.0
        self.position = (
            cv2.getTrackbarPos('X Position', 'DXF Overlay'),
            cv2.getTrackbarPos('Y Position', 'DXF Overlay')
        )
 
    def run(self):
        # Browse for DXF file
        dxf_path = self.browse_dxf_file()
        if not dxf_path:
            print("No file selected")
            return
       
        # Render DXF
        self.dxf_image = self.render_dxf(dxf_path)
        if self.dxf_image is None:
            print("Failed to render DXF file")
            return
       
        while True:
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                break
           
            # Convert to BGRA for transparency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
           
            # Update overlay parameters
            self.update_overlay()
           
            # Overlay DXF if available
            if self.dxf_image is not None:
                # Scale DXF image
                h, w = self.dxf_image.shape[:2]
                scaled_img = cv2.resize(
                    self.dxf_image,
                    (int(w * self.scale), int(h * self.scale)),
                    interpolation=cv2.INTER_AREA
                )
               
                # Position overlay
                x, y = self.position
                frame_h, frame_w = frame.shape[:2]
                overlay_h, overlay_w = scaled_img.shape[:2]
               
                # Calculate overlay region
                y1, y2 = max(0, y), min(frame_h, y + overlay_h)
                x1, x2 = max(0, x), min(frame_w, x + overlay_w)
               
                # Calculate source region
                src_y1 = max(0, -y)
                src_x1 = max(0, -x)
                src_y2 = src_y1 + (y2 - y1)
                src_x2 = src_x1 + (x2 - x1)
               
                # Blend images
                if src_y2 > src_y1 and src_x2 > src_x1:
                    alpha = scaled_img[src_y1:src_y2, src_x1:src_x2, 3] / 255.0
                    for c in range(3):
                        frame[y1:y2, x1:x2, c] = (
                            alpha * scaled_img[src_y1:src_y2, src_x1:src_x2, c] +
                            (1 - alpha) * frame[y1:y2, x1:x2, c]
                        )
           
            # Display result
            cv2.imshow('DXF Overlay', cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
           
            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break
       
        self.cap.release()
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    overlay = DXFOverlay()
    overlay.run()
 
 