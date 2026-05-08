import pystray
from PIL import Image, ImageDraw
from ghost_engine import GhostMouseEngine

# Initialize the Ghost Mouse Engine
engine = GhostMouseEngine()

def create_image():
    """Generates a simple icon for the system tray."""
    # Create a simple mouse/ghost-like icon
    image = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    # Draw a simple shape to represent the app
    d.ellipse((12, 12, 52, 52), fill=(100, 100, 255))
    d.polygon([(32, 12), (52, 52), (12, 52)], fill=(255, 255, 255))
    return image

def set_state(icon, item):
    """Handles system tray menu clicks."""
    if item.text == "Start Hand Tracking":
        engine.start(use_face=False)
    elif item.text == "Start Face Tracking":
        engine.start(use_face=True)
    elif item.text == "Stop Tracking":
        engine.stop()
    elif item.text == "Toggle Camera Preview":
        engine.show_preview = not engine.show_preview
    elif item.text == "Exit":
        engine.stop()
        icon.stop()

# Define the system tray menu
menu = pystray.Menu(
    pystray.MenuItem("Start Hand Tracking", set_state),
    pystray.MenuItem("Start Face Tracking", set_state),
    pystray.MenuItem("Toggle Camera Preview", set_state, checked=lambda item: engine.show_preview),
    pystray.MenuItem("Stop Tracking", set_state),
    pystray.MenuItem("Exit", set_state)
)

# Create the system tray icon
icon = pystray.Icon("Ghost Mouse", create_image(), "Ghost Mouse Control", menu)

if __name__ == "__main__":
    # This blocks the main thread and runs the system tray icon
    print("Ghost Mouse Widget started in system tray.")
    icon.run()
