import sys
import os

# 1. Tell Python where the MinGW C++ runtime DLLs are located
os.add_dll_directory("C:/msys64/mingw64/bin")

# 2. Now import your module! (Make sure the .pyd file is in the same folder as main.py)
import my_backend 

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt + C++ Demo")
        self.resize(300, 150)

        # 1. Create Layout
        layout = QVBoxLayout()

        # 2. Create Widgets
        self.input_field = QLineEdit("Type something here...")
        self.button = QPushButton("Send to C++")
        self.result_label = QLabel("Result will appear here")

        # 3. Add widgets to layout
        layout.addWidget(self.input_field)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # 4. Connect the PyQt signal to the Python slot
        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        """This function runs when the button is clicked."""
        
        # Grab the text from the UI
        text_to_process = self.input_field.text()
        
        # CALL THE C++ FUNCTION
        # This jumps out of Python, runs the C++ code at native speed, and returns the result.
        result = my_backend.process_data(text_to_process)
        
        # Update the UI with the result from C++
        self.result_label.setText(result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())