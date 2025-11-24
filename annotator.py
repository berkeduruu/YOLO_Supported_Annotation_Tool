import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QSlider, QLabel, QListWidget, QListWidgetItem, QAbstractItemView,
    QGraphicsItem, QStyle, QStatusBar, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QColor, QCursor
from PyQt6.QtCore import Qt, QPointF, QRectF, QThread, pyqtSignal

def xyxy_to_yolo(box, img_w, img_h):
    cls_id, x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def yolo_to_xyxy(yolo_line, img_w, img_h):
    try:
        cls_id, x_c, y_c, w_norm, h_norm = map(float, yolo_line.split())
        x_c_abs = x_c * img_w
        y_c_abs = y_c * img_h
        w_abs = w_norm * img_w
        h_abs = h_norm * img_h
        x1 = int(x_c_abs - (w_abs / 2))
        y1 = int(y_c_abs - (h_abs / 2))
        x2 = int(x_c_abs + (w_abs / 2))
        y2 = int(y_c_abs + (h_abs / 2))
        return [int(cls_id), x1, y1, x2, y2]
    except Exception:
        return None


class FileLoaderThread(QThread):
    files_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            self.files_loaded.emit(image_files)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, *args, main_window=None, **kwargs):
        super(ResizableRectItem, self).__init__(*args, **kwargs)
        self.main_window = main_window 
        self.handles = {}
        self.handle_size = 8.0
        self.handle_space = 4.0
        
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None
        
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        
        self.update_handles()
        self.show_handles(False)

    def handle_positions(self):
        rect = self.rect()
        size = self.handle_size
        return {
            'top_left': QRectF(rect.topLeft().x() - size/2, rect.topLeft().y() - size/2, size, size),
            'top_middle': QRectF(rect.center().x() - size/2, rect.top() - size/2, size, size),
            'top_right': QRectF(rect.topRight().x() - size/2, rect.topRight().y() - size/2, size, size),
            'middle_left': QRectF(rect.left() - size/2, rect.center().y() - size/2, size, size),
            'middle_right': QRectF(rect.right() - size/2, rect.center().y() - size/2, size, size),
            'bottom_left': QRectF(rect.bottomLeft().x() - size/2, rect.bottomLeft().y() - size/2, size, size),
            'bottom_middle': QRectF(rect.center().x() - size/2, rect.bottom() - size/2, size, size),
            'bottom_right': QRectF(rect.bottomRight().x() - size/2, rect.bottomRight().y() - size/2, size, size),
        }

    def update_handles(self):
        positions = self.handle_positions()
        for key, rect in positions.items():
            if key in self.handles:
                self.handles[key].setRect(rect)
            else:
                self.handles[key] = QGraphicsRectItem(rect, self) 
                self.handles[key].setBrush(QBrush(Qt.GlobalColor.white))
                self.handles[key].setPen(QPen(Qt.GlobalColor.black, 1.0))

    def show_handles(self, show):
        for handle in self.handles.values():
            handle.setVisible(show)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self.show_handles(bool(value))
        return super(ResizableRectItem, self).itemChange(change, value)

    def hoverMoveEvent(self, event):
        if self.isSelected() and self.main_window and self.main_window.mode == 'DRAW':
            pos = event.pos()
            cursor = Qt.CursorShape.ArrowCursor
            self.handle_selected = None
            
            for key, rect in self.handle_positions().items():
                if rect.contains(pos):
                    self.handle_selected = key
                    if key in ['top_left', 'bottom_right']:
                        cursor = Qt.CursorShape.SizeFDiagCursor
                    elif key in ['top_right', 'bottom_left']:
                        cursor = Qt.CursorShape.SizeBDiagCursor
                    elif key in ['top_middle', 'bottom_middle']:
                        cursor = Qt.CursorShape.SizeVerCursor
                    else:
                        cursor = Qt.CursorShape.SizeHorCursor
                    break
            self.setCursor(QCursor(cursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)) 
            self.handle_selected = None
            
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if self.handle_selected and event.button() == Qt.MouseButton.LeftButton and \
           self.main_window and self.main_window.mode == 'DRAW':
            self.mouse_press_pos = event.pos()
            self.mouse_press_rect = self.rect()
        else:
            self.handle_selected = None
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.handle_selected and self.main_window and self.main_window.mode == 'DRAW':
            pos = event.pos()
            delta = pos - self.mouse_press_pos
            rect = QRectF(self.mouse_press_rect) 

            # (Logic for resizing box based on handle selected)
            if self.handle_selected == 'top_left': rect.setTopLeft(rect.topLeft() + delta)
            elif self.handle_selected == 'top_middle': rect.setTop(rect.top() + delta.y())
            elif self.handle_selected == 'top_right': rect.setTopRight(rect.topRight() + delta)
            elif self.handle_selected == 'middle_left': rect.setLeft(rect.left() + delta.x())
            elif self.handle_selected == 'middle_right': rect.setRight(rect.right() + delta.x())
            elif self.handle_selected == 'bottom_left': rect.setBottomLeft(rect.bottomLeft() + delta)
            elif self.handle_selected == 'bottom_middle': rect.setBottom(rect.bottom() + delta.y())
            elif self.handle_selected == 'bottom_right': rect.setBottomRight(rect.bottomRight() + delta)
            # ...
            
            self.prepareGeometryChange()
            self.setRect(rect.normalized())
            self.update_handles()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget):
        # (Drawing the box and label)
        painter.setPen(self.pen()) 
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())
        
        if option.state & QStyle.StateFlag.State_Selected:
            select_pen = QPen(QColor(255, 255, 0), 3, Qt.PenStyle.DashLine)
            painter.setPen(select_pen)
            painter.setBrush(Qt.GlobalColor.transparent) 
            painter.drawRect(self.rect())

        class_name = self.data(1)
        confidence = self.data(2)
        label = ""
        if class_name:
            if confidence is not None:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = f"{class_name}"
        
        if not (option.state & QStyle.StateFlag.State_Selected) and label:
            painter.setPen(self.pen()) 
            font = painter.font()
            font.setPixelSize(14) 
            painter.setFont(font)
            text_pos = self.rect().topLeft() + QPointF(0, -5)
            painter.drawText(text_pos, label)


# --- Cursor-focused Zoom View Class ---
class AnnotationView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super(AnnotationView, self).__init__(scene, parent)
        self.start_point = QPointF()
        self.temp_rect = None 
        self.drawing = False
        self.main_window = parent
        
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event):
        if not self.main_window.image_loaded:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        
        if (item is None or isinstance(item, QGraphicsPixmapItem)) and \
           event.button() == Qt.MouseButton.LeftButton and \
           self.main_window.image_loaded and \
           self.main_window.mode == 'DRAW':
            
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            
            if self.temp_rect:
                self.scene().removeItem(self.temp_rect)
            
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            self.temp_rect = self.scene().addRect(
                self.start_point.x(), self.start_point.y(), 0, 0, pen
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.main_window.mode == 'DRAW':
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            self.temp_rect.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_rect:
                self.scene().removeItem(self.temp_rect)
                self.temp_rect = None
                
                end_point = self.mapToScene(event.pos())
                rect = QRectF(self.start_point, end_point).normalized()
                
                if rect.width() > 5 and rect.height() > 5:
                    self.main_window.add_box_to_scene(
                        rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height(),
                        is_manual=True
                    )
        super().mouseReleaseEvent(event)

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid YOLO Annotator (PyQt6 - v11)") # Title is English
        self.setGeometry(100, 100, 1200, 800)
        
        self.model = None
        self.class_names = {}
        self.colors = []
        self.image_folder = ""
        self.label_folder = ""
        self.image_files = []
        self.current_index = -1
        self.image_loaded = False
        self.current_image_cv = None 
        self.mode = 'DRAW' 
        self.file_loader_thread = None 

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        # BUTTONS
        self.btn_load_model = QPushButton("1. Load Model (.pt)")
        self.btn_load_images = QPushButton("2. Select Image Folder")
        self.btn_load_labels = QPushButton("3. Select Label Folder")
        left_layout.addWidget(self.btn_load_model)
        left_layout.addWidget(self.btn_load_images)
        left_layout.addWidget(self.btn_load_labels)

        # CHECKBOX
        self.chk_load_txt = QCheckBox("Load existing .txt labels")
        self.chk_load_txt.setChecked(False) 
        left_layout.addWidget(self.chk_load_txt)

        # CONFIDENCE SLIDER
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Confidence Threshold (%):"))
        self.slider_conf = QSlider(Qt.Orientation.Horizontal) 
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(25)
        self.slider_conf.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_conf.setTickInterval(10)
        self.lbl_conf = QLabel("25")
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.lbl_conf)
        left_layout.addLayout(conf_layout)
        
        self.btn_repredict = QPushButton("Re-predict (R)")
        left_layout.addWidget(self.btn_repredict)
        
        # CLASS LIST
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Label Classes (Select for manual add):"))
        self.list_classes = QListWidget()
        self.list_classes.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left_layout.addWidget(self.list_classes)

        # FILE LIST
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Image Files:"))
        self.list_files = QListWidget()
        self.list_files.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left_layout.addWidget(self.list_files)

        # NAVIGATION BUTTONS
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<< Previous (A)")
        self.btn_next = QPushButton("Next (D)")
        self.btn_skip = QPushButton("Skip (E) >>") 
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_skip)
        right_layout.addLayout(nav_layout)
        
        # ZOOM BUTTONS
        zoom_layout = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom In (+)")
        self.btn_zoom_out = QPushButton("Zoom Out (-)")
        self.btn_zoom_fit = QPushButton("Fit to View (F)")
        zoom_layout.addWidget(self.btn_zoom_in)
        zoom_layout.addWidget(self.btn_zoom_out)
        zoom_layout.addWidget(self.btn_zoom_fit)
        right_layout.addLayout(zoom_layout)
        
        # VIEWER
        self.scene = QGraphicsScene()
        self.view = AnnotationView(self.scene, self)
        right_layout.addWidget(self.view)
        
        self.setStatusBar(QStatusBar(self))
        
        # --- Signal-Slot Connections ---
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_load_labels.clicked.connect(self.load_labels)
        self.slider_conf.valueChanged.connect(self.update_conf_label)
        self.btn_repredict.clicked.connect(self.run_prediction)
        
        self.list_files.currentItemChanged.connect(self.on_file_selected) 
        self.list_classes.currentItemChanged.connect(lambda: self.update_status_bar()) 
        
        self.btn_next.clicked.connect(self.next_image)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_skip.clicked.connect(self.skip_image) 
        
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.zoom_fit)
        
        self.update_status_bar("Load model and folders to start.") # Initial status bar text

    def update_view_mode(self):
        if self.mode == 'DRAW':
            self.mode = 'PAN'
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.mode = 'DRAW'
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.update_status_bar()

    def update_status_bar(self, message=None):
        if message and isinstance(message, str):
            self.statusBar().showMessage(message)
            return

        if not self.image_loaded:
            self.statusBar().showMessage("Load model and folders to start.")
            return

        mode_text = f"MODE: {self.mode} (Press 'V' to switch)"
        
        if not self.image_files or self.current_index < 0 or self.current_index >= len(self.image_files):
             file_text = "File: Not Loaded"
        else:
            file_text = f"File: [{self.current_index + 1}/{len(self.image_files)}] {self.image_files[self.current_index]}"
        
        class_text = "Active Class: NONE"
        if self.list_classes.currentItem():
            class_text = f"Active Class: {self.list_classes.currentItem().text()}"
            
        self.statusBar().showMessage(f"{mode_text}  |  {class_text}  |  {file_text}")

    def zoom_in(self):
        self.view.scale(1.2, 1.2)
        
    def zoom_out(self):
        self.view.scale(0.8, 0.8)
        
    def zoom_fit(self):
        if self.image_loaded:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q: 
            self.close()
            
        if not self.image_loaded: 
            super().keyPressEvent(event)
            return
            
        key = event.key()
        
        if key == Qt.Key.Key_V:
            self.update_view_mode()
        
        elif key == Qt.Key.Key_A:
            self.prev_image()
        elif key == Qt.Key.Key_D:
            self.next_image()
        elif key == Qt.Key.Key_E: 
            self.skip_image()
        
        elif key == Qt.Key.Key_R:
            self.run_prediction()
        elif key == Qt.Key.Key_F:
            self.zoom_fit()
        elif key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace or key == Qt.Key.Key_X: 
            self.delete_selected_boxes()

        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            self.change_selected_class(key - Qt.Key.Key_0) 

        else:
            super().keyPressEvent(event)

    def delete_selected_boxes(self):
        selected_items = self.scene.selectedItems()
        if selected_items:
            for item in selected_items:
                if isinstance(item, ResizableRectItem):
                    self.scene.removeItem(item)
            print(f"Deleted {len(selected_items)} selected box(es).")
        else: 
            all_boxes = self.get_boxes_from_scene()
            if all_boxes:
                self.scene.removeItem(all_boxes[-1])
                print("Deleted the last drawn box.")
            else:
                 print("No boxes to delete.")

    def change_selected_class(self, class_id):
        if class_id >= self.list_classes.count() or class_id not in self.class_names:
            print(f"Error: Class ID {class_id} not found.")
            return
            
        selected_items = self.scene.selectedItems()
        if not selected_items:
            print("Select a box first to change its class.")
            return
        
        class_name = self.class_names[class_id]
        pen_color = self.colors[class_id]
        
        for item in selected_items:
            if isinstance(item, ResizableRectItem):
                new_pen = QPen(QColor(int(pen_color[0]), int(pen_color[1]), int(pen_color[2])), 3)
                
                item.setData(0, class_id)
                item.setData(1, class_name)
                item.setData(2, None)
                item.setPen(new_pen)
        
        print(f"Changed class of {len(selected_items)} selected box(es) to -> {class_name}")


    # --- File Loading Functions ---
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PT Files (*.pt)")
        if path:
            try:
                self.model = YOLO(path)
                self.class_names = self.model.names
                self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
                
                self.list_classes.clear()
                for name in self.class_names.values():
                    self.list_classes.addItem(name)
                    
                self.list_classes.setCurrentRow(0)
                print(f"Model loaded. Classes: {self.class_names}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")

    def load_images(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.image_folder = path
            self.update_status_bar("Scanning image files...")
            
            self.btn_load_images.setEnabled(False)
            self.btn_load_model.setEnabled(False)
            
            self.file_loader_thread = FileLoaderThread(path)
            self.file_loader_thread.files_loaded.connect(self.on_files_loaded)
            self.file_loader_thread.error_occurred.connect(self.on_files_load_error)
            self.file_loader_thread.start()

    def on_files_loaded(self, image_files):
        self.image_files = image_files
        self.list_files.clear()
        
        self.list_files.addItems(self.image_files) 
        
        self.btn_load_images.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        
        if self.image_files:
            self.list_files.setCurrentRow(0)
            self.update_status_bar("Files loaded. Displaying first image.")
        else:
            self.update_status_bar("No images found in this folder.")
            print("Error: No images found.")
            
    def on_files_load_error(self, error_msg):
        self.btn_load_images.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        self.update_status_bar(f"Error: {error_msg}")
        print(f"[ERROR] Failed to load files: {error_msg}")

    def load_labels(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Labels")
        if path:
            self.label_folder = path
            print(f"Labels will be saved here: {path}")

    def update_conf_label(self, value):
        self.lbl_conf.setText(str(value))
    
    def on_file_selected(self, current_item, previous_item):
        """ Triggered when an item is selected from the file list (manually or by code) """
        if not current_item:
            return
            
        if previous_item:
            try:
                prev_index = self.image_files.index(previous_item.text())
                # --- FIX (BUG) ---
                # Check for the skip flag to prevent saving when the 'skip' button is pressed.
                if not hasattr(self, '_skip_flag') or not self._skip_flag:
                    self.save_annotations(prev_index)
                self._skip_flag = False # Reset the flag
            except ValueError:
                pass 

        self.current_index = self.image_files.index(current_item.text())
        self.display_image(self.current_index)
        self.update_status_bar()

    # --- Navigation Functions ---
    def next_image(self):
        """ 'D' key: Save and move to the next image """
        if not self.image_files: return
        if self.current_index < len(self.image_files) - 1:
            # Update the list, 'on_file_selected' handles save, display, and status update
            self.list_files.setCurrentRow(self.current_index + 1)
        else:
            print("You are on the last image. (Press A to save the current one)")
            # Save the last image's annotations
            self.save_annotations(self.current_index)
            
    def prev_image(self):
        """ 'A' key: Save and move to the previous image """
        if not self.image_files: return
        if self.current_index > 0:
            # Update the list
            self.list_files.setCurrentRow(self.current_index - 1)
        else:
            print("You are on the first image.")

    # --- FIX (BUG) ---
    def skip_image(self):
        """ 'E' key: Do not save, delete label if exists, and move to the next image """
        if not self.image_files: return
        if self.current_index < len(self.image_files) - 1:
            
            # 1. Perform deletion/skip
            if self.label_folder:
                label_name = os.path.splitext(self.image_files[self.current_index])[0] + ".txt"
                label_path = os.path.join(self.label_folder, label_name)
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Skipped and deleted: {label_path}")
                else:
                    print(f"Skipped (No label to delete): {self.image_files[self.current_index]}")
            else:
                 print(f"Skipped (Label folder not set): {self.image_files[self.current_index]}")

            # 2. Set the flag to prevent 'on_file_selected' from saving
            self._skip_flag = True 
            
            # 3. Update the list and index (This triggers 'on_file_selected')
            self.list_files.setCurrentRow(self.current_index + 1)
        else:
            print("You are on the last image.")


    # --- Display and Prediction Functions ---
    def display_image(self, index):
        if not (0 <= index < len(self.image_files)):
            return
            
        self.image_loaded = False
        img_path = os.path.join(self.image_folder, self.image_files[index])
        
        self.current_image_cv = cv2.imread(img_path)
        if self.current_image_cv is None:
            print(f"Error: Could not load image: {img_path}")
            # Automatically skip corrupted image
            self.skip_image() 
            return
            
        h, w, ch = self.current_image_cv.shape
        bytes_per_line = ch * w
        q_img = QImage(self.current_image_cv.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.image_loaded = True
        
        self.zoom_fit()
        
        load_existing = self.chk_load_txt.isChecked()
        txt_path = ""
        if load_existing and self.label_folder:
            label_name = os.path.splitext(self.image_files[index])[0] + ".txt"
            txt_path = os.path.join(self.label_folder, label_name)
        
        if load_existing and os.path.exists(txt_path):
            self.load_annotations_from_file(txt_path)
        else:
            self.run_prediction()

    def load_annotations_from_file(self, txt_path):
        """ Loads existing .txt annotations instead of running the model """
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                self.scene.removeItem(item)
                
        h, w, _ = self.current_image_cv.shape
        box_count = 0
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                box_data = yolo_to_xyxy(line, w, h)
                if box_data:
                    cls_id, x1, y1, x2, y2 = box_data
                    class_name = self.class_names.get(cls_id, "Unknown")
                    self.add_box_to_scene(x1, y1, x2, y2, cls_id, class_name, 
                                          confidence=None, is_manual=False)
                    box_count += 1
            print(f"Loaded {box_count} boxes from existing .txt file.")
        except Exception as e:
            print(f"[ERROR] Error reading .txt file: {e}")

    def run_prediction(self):
        if not self.image_loaded or not self.model:
            print("Model or image not loaded.")
            return
            
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                self.scene.removeItem(item)
                
        conf = self.slider_conf.value() / 100.0
        
        try:
            results = self.model.predict(self.current_image_cv, conf=conf, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                data = results[0].boxes.data.cpu().numpy()
                
                for row in data:
                    x1, y1, x2, y2 = row[:4].astype(int)
                    confidence = row[4]
                    class_id = int(row[5])
                    class_name = self.class_names.get(class_id, "Unknown")
                    
                    self.add_box_to_scene(x1, y1, x2, y2, class_id, class_name, 
                                          confidence=confidence, is_manual=False)
                print(f"Prediction ran. Found {len(data)} objects.")
            else:
                print("Prediction ran. Found 0 objects.")
            
        except Exception as e:
            print(f"[ERROR] Error during model prediction: {e}")

    def add_box_to_scene(self, x1, y1, x2, y2, class_id=None, class_name=None, confidence=None, is_manual=False):
        
        if is_manual:
            if self.list_classes.currentItem():
                class_id = self.list_classes.currentRow()
                class_name = self.class_names.get(class_id, "Unknown")
            else:
                class_id = 0 
                class_name = self.class_names.get(0, "Unknown")
            confidence = None
        
        class_id = int(class_id)
        
        if is_manual:
            pen = QPen(QColor(255, 0, 0), 3) # Manual drawing is always RED
        else:
            pen_color = self.colors[class_id]
            pen = QPen(QColor(int(pen_color[0]), int(pen_color[1]), int(pen_color[2])), 3) 

        rect_item = ResizableRectItem(x1, y1, x2 - x1, y2 - y1, main_window=self)
        rect_item.setPen(pen)
        rect_item.setBrush(QBrush(Qt.GlobalColor.transparent)) 
        
        rect_item.setData(0, class_id)
        rect_item.setData(1, class_name)
        rect_item.setData(2, confidence)
        
        self.scene.addItem(rect_item)

    def get_boxes_from_scene(self):
        boxes = []
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                boxes.append(item)
        return boxes

    def save_annotations(self, index):
        if not self.label_folder or not (0 <= index < len(self.image_files)) or not self.current_image_cv is not None:
            return
            
        label_name = os.path.splitext(self.image_files[index])[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)
        
        h, w, _ = self.current_image_cv.shape
        
        all_boxes = self.get_boxes_from_scene()
        
        try:
            with open(label_path, "w") as f:
                if not all_boxes:
                    pass # Empty file (background)
                else:
                    for item in all_boxes:
                        rect = item.rect()
                        x1 = rect.x()
                        y1 = rect.y()
                        x2 = x1 + rect.width()
                        y2 = y1 + rect.height()
                        cls_id = item.data(0) 
                        
                        box_data = [cls_id, x1, y1, x2, y2]
                        yolo_line = xyxy_to_yolo(box_data, w, h)
                        f.write(yolo_line + "\n")
            print(f"Saved: {label_path} ({len(all_boxes)} boxes)")
        except Exception as e:
            print(f"Error: Failed to save labels! {e}")

# --- Start Application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())