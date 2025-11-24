# YOLO_Supported_Annotation_Tool
# 🏷️ Hybrid YOLO Annotation Tool (PyQt6 & Ultralytics)

This application is a feature-rich, Python-based Graphical User Interface (GUI) tool for efficiently annotating images in the standard **YOLO format**. It combines the speed of model-based prediction with the precision of manual editing, making it ideal for large-scale dataset creation and refinement.

## ✨ Core Features & Technical Details

* **Hybrid Annotation Workflow:** Automatically generates initial bounding boxes by running predictions via a loaded **Ultralytics YOLO model (`.pt`)**.
* **Interactive GUI (PyQt6):** Built using **PyQt6** for a responsive and modern interface.
* **Resizable Bounding Boxes:** Implements a custom **`ResizableRectItem`** class inherited from `QGraphicsRectItem`, enabling drag-and-drop resizing from 8 different handles (corners and edges).
* **File Management:** Supports loading image directories and specifying a separate directory for saving/loading label files. Includes an option (`QCheckBox`) to load existing `.txt` annotations first, skipping re-prediction.
* **YOLO Format I/O:**
    * **Input:** Loads existing normalized YOLO format (`class x_center y_center w h`) annotations.
    * **Output:** Saves final annotations to `.txt` files in the correct normalized YOLO format, using **`xyxy_to_yolo`** conversion.
* **Navigation & Shortcuts:**
    * **Next/Previous:** Navigate images quickly using **`D`** and **`A`** keys, triggering automatic saving of annotations.
    * **Skip:** Use the **`E`** key to skip the current image, **deleting the existing label file** if one is present, and moving to the next image.
* **Viewing Controls:**
    * **Zoom:** Cursor-focused zoom via the mouse wheel.
    * **Mode Switch:** Toggle between **'DRAW'** mode (drawing/editing bounding boxes) and **'PAN'** mode (hand drag scrolling) using the **`V`** key.
    * **Fit to View:** **`F`** key shortcut to fit the image to the viewer.
* **Editing Features:** Delete selected boxes (`Delete`/`Backspace`/`X` keys) or change the class of selected boxes using number keys (`0`-`9`).
* **Model Configuration:** Adjustable **Confidence Threshold** (`QSlider`) to filter low-confidence predictions from the YOLO model.

## 🛠️ Installation and Setup

### 📦 Dependencies

The script relies on the following Python libraries:

| Library Name | Installation Command |
| :--- | :--- |
| **`ultralytics`** | `pip install ultralytics` |
| **`PyQt6`** | `pip install PyQt6` |
| **`opencv-python`** | `pip install opencv-python` |
| **`numpy`** | `pip install numpy` |

A full environment setup can be done with a single command:

```bash
pip install ultralytics PyQt6 opencv-python numpy

🚀 Running the ToolEnsure all dependencies are installed.Execute the script:Bashpython your_script_name.py
In the GUI:Click "1. Load Model (.pt)" to load your YOLO weights.Click "2. Select Image Folder" to load your dataset images.Click "3. Select Label Folder" to specify the output directory for your .txt files.⌨️ ShortcutsKeyActionASave annotations and move to the Previous image.DSave annotations and move to the Next image.ESkip the image (delete label file if present) and move to the next image.VToggle between DRAW (Annotation) and PAN (Scrolling) modes.RRe-run Prediction on the current image.FFit the image to the viewer bounds.X / DeleteDelete selected bounding boxes (or the last drawn box if none are selected).0-9Change the class ID of selected bounding boxes.
