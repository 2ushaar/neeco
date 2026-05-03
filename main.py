# Standard library imports
import sys
import os
import io
import base64
import threading
from pathlib import Path

# Third-party libraries for screen capture, hotkeys, and AI
from dotenv import load_dotenv
from PIL import Image
from mss import mss
import keyboard
from google import genai
from google.genai import types

# PyQt6 widgets for the desktop UI
from PyQt6.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu,
    QTextEdit, QLineEdit, QVBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QAction, QPainter, QColor

# Load environment variables (GEMINI_API_KEY) from .env
load_dotenv()

class SignalBridge(QObject):
    """Bridge to send signals from background threads to the Qt main thread."""
    # Carries a streamed text chunk from Gemini
    text_chunk = pyqtSignal(str)
    # Fires when the full response has finished streaming
    response_done = pyqtSignal()
    # Carries status messages like "Capturing screen..."
    status_update = pyqtSignal(str)

class GeminiClient:
    """Handles communication with the Gemini API."""

    def __init__(self, api_key: str):
        # Create a Gemini client using the provided API key
        self.client = genai.Client(api_key=api_key)
        # Set the model to Gemini 2.5 Flash (free tier, vision-capable)
        self.model = "gemini-2.5-flash"
        # Store conversation history so Gemini remembers previous messages
        self.history = []

    def analyze_screenshot(self, image_bytes: bytes, prompt: str):
        """Send a screenshot to Gemini and yield streamed text chunks."""
        # Wrap the raw JPEG bytes as an image part Gemini can read
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        # Wrap the text prompt as a separate part
        text_part = types.Part.from_text(text=prompt)

        # Bundle both parts into a single user message
        user_content = types.Content(
            role="user",
            parts=[image_part, text_part]
        )
        self.history.append(user_content)

        # Stream the response so chunks arrive as they are generated
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=self.history,
        )

        # Yield each text chunk as it arrives and accumulate the full response
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        # Save the model's complete reply to history for context in follow-ups
        assistant_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=full_response)]
        )
        self.history.append(assistant_content)

    def ask_followup(self, question: str):
        """Send a text-only follow-up question."""
        # Create a text-only message (no image this time)
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        )
        self.history.append(user_content)

        # Stream the response the same way as analyze_screenshot
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=self.history,
        )

        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        # Save the model's reply to history
        assistant_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=full_response)]
        )
        self.history.append(assistant_content)

    def clear_history(self):
        """Reset conversation history."""
        self.history = []



def capture_screenshot(monitor_index: int = 1) -> bytes:
    """Capture a screenshot and return it as JPEG bytes."""
    with mss() as sct:
        # Select the target monitor (index 1 = primary monitor)
        monitor = sct.monitors[monitor_index]
        screenshot = sct.grab(monitor)

        # Convert raw BGRA pixels to RGB, then compress as JPEG
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

class ChatPanel(QWidget):
    """Floating chat panel for displaying AI responses."""

    def __init__(self, gemini_client: GeminiClient, signal_bridge: SignalBridge):
        super().__init__()
        self.gemini_client = gemini_client
        self.signals = signal_bridge
        self.monitor_index = 1

        # Window title and behavior flags
        self.setWindowTitle("AI Screen Reader")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setFixedSize(450, 500)

        # Vertical layout with some padding
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Read-only text area for AI responses
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText(
            "Press Ctrl+Alt to capture your screen...\n"
            "AI analysis will appear here."
        )
        layout.addWidget(self.chat_display)

        # Text input for follow-up questions
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a follow-up question...")
        self.input_field.returnPressed.connect(self.send_followup)
        layout.addWidget(self.input_field)

        self.setLayout(layout)

        # Connect signals from background threads to display methods
        self.signals.text_chunk.connect(self.append_text)
        self.signals.response_done.connect(self.on_response_done)
        self.signals.status_update.connect(self.show_status)

    def append_text(self, text: str):
        """Append a text chunk to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_response_done(self):
        """Called when streaming is complete."""
        self.append_text("\n\n")
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def show_status(self, message: str):
        """Show a status message in the chat."""
        self.chat_display.append(f"\n--- {message} ---\n")

    def trigger_capture(self):
        """Capture screen and send to Gemini."""
        # Bring the chat panel to the front
        self.show()
        self.raise_()
        self.activateWindow()
        self.input_field.setEnabled(False)
        self.signals.status_update.emit("Capturing screen...")

        # Run capture and analysis in a background thread
        thread = threading.Thread(target=self._capture_and_analyze, daemon=True)
        thread.start()

    def _capture_and_analyze(self):
        """Background thread: capture and stream Gemini response."""
        try:
            image_bytes = capture_screenshot(self.monitor_index)
            self.signals.status_update.emit("Analyzing with Gemini...")

            prompt = "What do you see on this screen? Be helpful and concise."
            for chunk in self.gemini_client.analyze_screenshot(image_bytes, prompt):
                self.signals.text_chunk.emit(chunk)

            self.signals.response_done.emit()
        except Exception as e:
            self.signals.status_update.emit(f"Error: {e}")
            self.signals.response_done.emit()

    def send_followup(self):
        """Send a follow-up text question."""
        question = self.input_field.text().strip()
        if not question:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.chat_display.append(f"\nYou: {question}\n")

        # Stream the follow-up response in a background thread
        thread = threading.Thread(
            target=self._stream_followup, args=(question,), daemon=True
        )
        thread.start()

    def _stream_followup(self, question: str):
        """Background thread: stream follow-up response."""
        try:
            for chunk in self.gemini_client.ask_followup(question):
                self.signals.text_chunk.emit(chunk)
            self.signals.response_done.emit()
        except Exception as e:
            self.signals.status_update.emit(f"Error: {e}")
            self.signals.response_done.emit()

def create_tray_icon() -> QPixmap:
    """Create a simple colored icon for the system tray."""
    # Start with a transparent 32x32 canvas
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    # Draw a blue circle as the outer ring
    painter.setBrush(QColor(66, 133, 244))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 28, 28)
    # Draw a white circle as the inner eye
    painter.setBrush(QColor(255, 255, 255))
    painter.drawEllipse(10, 10, 12, 12)
    painter.end()
    return pixmap

# # --- Temporary test code (remove in Step 4) ---
# if __name__ == "__main__":
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         print("Error: GEMINI_API_KEY not found. Check your .env file.")
#         sys.exit(1)

#     gemini = GeminiClient(api_key)

#     def on_hotkey():
#         print("Capturing screen...")
#         image_bytes = capture_screenshot()
#         print("Sending to Gemini...")
#         for chunk in gemini.analyze_screenshot(
#             image_bytes, "What do you see on this screen? Be concise."
#         ):
#             print(chunk, end="", flush=True)
#         print("\n\nDone!")

#     keyboard.add_hotkey("ctrl+alt", on_hotkey)
#     print("Press Ctrl+Alt to capture and analyze your screen.")
#     print("Press Ctrl+C to quit.")
#     keyboard.wait()

def main():
    # Load the API key from .env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        print("Create a .env file with: GEMINI_API_KEY=your-key-here")
        print("Get your key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Create the Qt application (keeps running even if all windows close)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Set up the core components
    signal_bridge = SignalBridge()
    gemini_client = GeminiClient(api_key)
    chat_panel = ChatPanel(gemini_client, signal_bridge)

    # Create the system tray icon
    tray_icon = QSystemTrayIcon()
    tray_icon.setIcon(QIcon(create_tray_icon()))
    tray_icon.setToolTip("AI Screen Reader - Ctrl+Alt to capture")

    # Build the right-click context menu
    tray_menu = QMenu()
    show_action = QAction("Show Panel")
    show_action.triggered.connect(chat_panel.show)
    tray_menu.addAction(show_action)

    clear_action = QAction("Clear History")
    clear_action.triggered.connect(gemini_client.clear_history)
    clear_action.triggered.connect(chat_panel.chat_display.clear)
    tray_menu.addAction(clear_action)

    tray_menu.addSeparator()

    quit_action = QAction("Quit")
    quit_action.triggered.connect(app.quit)
    tray_menu.addAction(quit_action)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()

    # Register the global hotkey to trigger screen capture
    keyboard.add_hotkey("ctrl+alt", chat_panel.trigger_capture)

    # Show the panel on startup and start the event loop
    chat_panel.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
