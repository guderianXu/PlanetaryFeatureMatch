#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotator_data
import geometry_predictor


DISPLAY_MAX_SIDE = 4096
POINT_RADIUS = 7.0
CLICK_FEEDBACK_RADIUS = 18.0
CLICK_FEEDBACK_MS = 650
UI_FONT_POINT_SIZE = 11
CONTROL_HEIGHT = 26
LIST_ROW_HEIGHT = 42
POINT_COLOR_A = "#ff2d55"
POINT_COLOR_B = "#008cff"
SELECTED_POINT_COLOR = "#ffd400"
PENDING_POINT_COLOR = "#ff9f0a"


def image_file_filter() -> str:
    patterns = " ".join(f"*{extension}" for extension in sorted(annotator_data.IMAGE_EXTENSIONS))
    return f"影像文件 ({patterns});;所有文件 (*)"


def pil_to_pixmap(image: Image.Image) -> QtGui.QPixmap:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    qimage = QtGui.QImage(rgba.tobytes("raw", "RGBA"), width, height, QtGui.QImage.Format.Format_RGBA8888).copy()
    return QtGui.QPixmap.fromImage(qimage)


def load_display_pixmap(path: Path) -> tuple[QtGui.QPixmap, tuple[int, int]]:
    with Image.open(path) as source:
        display = annotator_data.to_display_image(source, DISPLAY_MAX_SIDE)
        display_size = (display.width, display.height)
        return pil_to_pixmap(display), display_size


def point_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.hypot(left["x"] - right["x"], left["y"] - right["y"])


def format_point(point: dict[str, float]) -> str:
    return f"{point['x']:.1f}, {point['y']:.1f}"


def side_name(side: str) -> str:
    return "左图" if side == "a" else "右图"


class ImageView(QtWidgets.QGraphicsView):
    pointPressed = QtCore.Signal(str, float, float)
    pointMoved = QtCore.Signal(str, float, float)
    pointReleased = QtCore.Signal(str)

    def __init__(self, side: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.side = side
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self.original_size = (1, 1)
        self.display_size = (1, 1)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def load_image(self, path: Path, original_size: tuple[int, int]) -> None:
        pixmap, display_size = load_display_pixmap(path)
        self.original_size = original_size
        self.display_size = display_size
        self.scene_obj.clear()
        self.pixmap_item = self.scene_obj.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene_obj.setSceneRect(0, 0, display_size[0], display_size[1])
        self.resetTransform()
        self.fit_to_view()

    def fit_to_view(self) -> None:
        rect = self.scene_obj.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def one_to_one(self) -> None:
        self.resetTransform()

    def zoom_by(self, factor: float) -> None:
        self.scale(factor, factor)

    def original_to_scene(self, point: dict[str, float]) -> QtCore.QPointF:
        scale_x = self.display_size[0] / max(1.0, float(self.original_size[0]))
        scale_y = self.display_size[1] / max(1.0, float(self.original_size[1]))
        return QtCore.QPointF(point["x"] * scale_x, point["y"] * scale_y)

    def scene_to_original(self, point: QtCore.QPointF) -> dict[str, float] | None:
        if point.x() < 0 or point.y() < 0 or point.x() > self.display_size[0] or point.y() > self.display_size[1]:
            return None
        scale_x = self.original_size[0] / max(1.0, float(self.display_size[0]))
        scale_y = self.original_size[1] / max(1.0, float(self.display_size[1]))
        return geometry_predictor.clamp_point({"x": point.x() * scale_x, "y": point.y() * scale_y}, self.original_size)

    def redraw_points(
        self,
        matches: list[dict[str, Any]],
        pending: dict[str, float] | None,
        selected_id: int | None,
    ) -> None:
        for item in list(self.scene_obj.items()):
            if item is not self.pixmap_item:
                self.scene_obj.removeItem(item)
        for index, match in enumerate(matches):
            point = match[self.side]
            selected = match.get("id") == selected_id
            color = QtGui.QColor(
                SELECTED_POINT_COLOR if selected else POINT_COLOR_A if self.side == "a" else POINT_COLOR_B
            )
            self.add_point(point, color, str(index + 1))
        if pending is not None:
            self.add_point(pending, QtGui.QColor(PENDING_POINT_COLOR), "+")

    def add_point(self, point: dict[str, float], color: QtGui.QColor, label: str) -> None:
        center = self.original_to_scene(point)
        outer = self.scene_obj.addEllipse(
            center.x() - POINT_RADIUS - 2.5,
            center.y() - POINT_RADIUS - 2.5,
            (POINT_RADIUS + 2.5) * 2,
            (POINT_RADIUS + 2.5) * 2,
            QtGui.QPen(QtGui.QColor("#111111"), 2.0),
            QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
        )
        outer.setZValue(9)
        ellipse = self.scene_obj.addEllipse(
            center.x() - POINT_RADIUS,
            center.y() - POINT_RADIUS,
            POINT_RADIUS * 2,
            POINT_RADIUS * 2,
            QtGui.QPen(QtGui.QColor("white"), 2.4),
            QtGui.QBrush(color),
        )
        ellipse.setZValue(10)
        text = self.scene_obj.addText(label)
        text_font = QtGui.QFont()
        text_font.setPointSize(UI_FONT_POINT_SIZE)
        text_font.setBold(True)
        text.setFont(text_font)
        text.setDefaultTextColor(QtGui.QColor("#172026"))
        text.setPos(center.x() + 8, center.y() - 14)
        text.setZValue(11)

    def event_to_point(self, event: QtGui.QMouseEvent) -> dict[str, float] | None:
        if self.pixmap_item is None:
            return None
        scene_point = self.mapToScene(event.position().toPoint())
        return self.scene_to_original(scene_point)

    def flash_point(self, point: dict[str, float]) -> None:
        if self.pixmap_item is None:
            return
        center = self.original_to_scene(point)
        pen = QtGui.QPen(QtGui.QColor("#f2c94c"), 2.4)
        pen.setCosmetic(True)
        radius = CLICK_FEEDBACK_RADIUS
        ring = self.scene_obj.addEllipse(
            center.x() - radius,
            center.y() - radius,
            radius * 2,
            radius * 2,
            pen,
            QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
        )
        horizontal = self.scene_obj.addLine(center.x() - radius, center.y(), center.x() + radius, center.y(), pen)
        vertical = self.scene_obj.addLine(center.x(), center.y() - radius, center.x(), center.y() + radius, pen)
        items: list[QtWidgets.QGraphicsItem] = [ring, horizontal, vertical]
        for item in items:
            item.setZValue(80)
        QtCore.QTimer.singleShot(CLICK_FEEDBACK_MS, lambda items=items: self.remove_feedback_items(items))

    def remove_feedback_items(self, items: list[QtWidgets.QGraphicsItem]) -> None:
        for item in items:
            if item.scene() is self.scene_obj:
                self.scene_obj.removeItem(item)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            point = self.event_to_point(event)
            if point is not None:
                event.accept()
                self.pointPressed.emit(self.side, point["x"], point["y"])
                self.flash_point(point)
                return
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        point = self.event_to_point(event)
        if point is not None:
            self.pointMoved.emit(self.side, point["x"], point["y"])
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.pointReleased.emit(self.side)
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.zoom_by(factor)


class ViewPanel(QtWidgets.QWidget):
    def __init__(self, side: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = ImageView(side)
        self.side_label = QtWidgets.QLabel(side_name(side))
        self.side_label.setMinimumWidth(36)
        side_font = self.side_label.font()
        side_font.setBold(True)
        self.side_label.setFont(side_font)
        self.name_label = QtWidgets.QLabel("")
        self.name_label.setMinimumWidth(60)
        self.name_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.coord_label = QtWidgets.QLabel("")
        self.coord_label.setMinimumWidth(100)
        self.coord_label.setMaximumWidth(112)
        self.coord_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.coord_label.setStyleSheet("color: #65717d; font-family: monospace;")
        self.fit_button = QtWidgets.QPushButton("适配")
        self.one_button = QtWidgets.QPushButton("1:1")
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.select_button = QtWidgets.QPushButton("选图")
        for button, width in (
            (self.select_button, 50),
            (self.fit_button, 46),
            (self.one_button, 40),
            (self.zoom_out_button, 28),
            (self.zoom_in_button, 28),
        ):
            button.setFixedWidth(width)
            button.setFixedHeight(CONTROL_HEIGHT)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(3)
        toolbar.addWidget(self.side_label)
        toolbar.addWidget(self.name_label, 1)
        toolbar.addWidget(self.coord_label)
        toolbar.addWidget(self.select_button)
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.one_button)
        toolbar.addWidget(self.zoom_out_button)
        toolbar.addWidget(self.zoom_in_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.view, 1)

        self.fit_button.clicked.connect(self.view.fit_to_view)
        self.one_button.clicked.connect(self.view.one_to_one)
        self.zoom_out_button.clicked.connect(lambda: self.view.zoom_by(1 / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self.view.zoom_by(1.25))


class AnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "annotations").mkdir(exist_ok=True)
        self.pairs = annotator_data.discover_pairs(self.root)
        self.filtered_pair_indices = list(range(len(self.pairs)))
        self.current_index = -1
        self.current_pair: annotator_data.ImagePair | None = None
        self.annotation: dict[str, Any] = {"matches": []}
        self.pending_a: dict[str, float] | None = None
        self.pending_b: dict[str, float] | None = None
        self.prediction_method = ""
        self.selected_id: int | None = None
        self.dragging: dict[str, Any] | None = None
        self.dirty = False
        self.selected_pair_images: dict[str, str | None] = {"a": None, "b": None}

        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self.save_annotation)

        self.build_ui()
        self.refresh_pair_widgets()
        if self.pairs:
            self.load_pair(0)
        else:
            self.set_feedback("没有图像对；请先选择左图和右图。左键拖动影像，中键标点")

    def build_ui(self) -> None:
        self.setWindowTitle("匹配标注器")
        self.resize(1480, 860)
        self.setMinimumSize(1080, 680)
        self.apply_compact_style()

        toolbar = QtWidgets.QToolBar("tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        self.addToolBar(toolbar)
        self.pair_combo = QtWidgets.QComboBox()
        self.pair_combo.setMinimumWidth(300)
        self.pair_combo.setMaximumWidth(460)
        self.pair_combo.setFixedHeight(CONTROL_HEIGHT)
        self.prev_button = QtGui.QAction("上一对", self)
        self.next_button = QtGui.QAction("下一对", self)
        self.confirm_button = QtGui.QAction("确认点", self)
        self.cancel_button = QtGui.QAction("取消", self)
        self.save_button = QtGui.QAction("保存", self)
        self.export_button = QtGui.QAction("导出", self)
        toolbar.addWidget(QtWidgets.QLabel("图像对 "))
        toolbar.addWidget(self.pair_combo)
        toolbar.addAction(self.prev_button)
        toolbar.addAction(self.next_button)
        toolbar.addSeparator()
        toolbar.addAction(self.confirm_button)
        toolbar.addAction(self.cancel_button)
        toolbar.addAction(self.save_button)
        toolbar.addAction(self.export_button)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("过滤图像对")
        self.filter_edit.setFixedHeight(CONTROL_HEIGHT)
        self.pair_list = QtWidgets.QListWidget()
        self.match_list = QtWidgets.QListWidget()
        self.pair_list.setUniformItemSizes(True)
        self.match_list.setUniformItemSizes(True)
        self.match_count_label = QtWidgets.QLabel("0 点")
        self.delete_match_button = QtWidgets.QPushButton("删除点对")
        self.delete_match_button.setFixedHeight(CONTROL_HEIGHT)
        self.delete_match_button.setFixedWidth(86)
        self.delete_match_button.setEnabled(False)
        self.delete_match_button.setToolTip("删除当前选中的匹配点对")

        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(420)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(5, 5, 5, 5)
        side_layout.setSpacing(4)
        pair_header = QtWidgets.QHBoxLayout()
        pair_header.setContentsMargins(0, 0, 0, 0)
        pair_header.setSpacing(4)
        pair_header.addWidget(QtWidgets.QLabel("图像对"))
        pair_header.addStretch(1)
        match_header = QtWidgets.QHBoxLayout()
        match_header.setContentsMargins(0, 0, 0, 0)
        match_header.setSpacing(4)
        match_header.addWidget(QtWidgets.QLabel("匹配点"))
        match_header.addWidget(self.match_count_label)
        match_header.addStretch(1)
        match_header.addWidget(self.delete_match_button)
        side_layout.addWidget(self.filter_edit)
        side_layout.addLayout(pair_header)
        side_layout.addWidget(self.pair_list, 2)
        side_layout.addLayout(match_header)
        side_layout.addWidget(self.match_list, 3)

        self.panel_a = ViewPanel("a")
        self.panel_b = ViewPanel("b")
        self.panel_a.select_button.clicked.connect(lambda: self.choose_pair_image("a"))
        self.panel_b.select_button.clicked.connect(lambda: self.choose_pair_image("b"))
        image_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        image_splitter.addWidget(self.panel_a)
        image_splitter.addWidget(self.panel_b)
        image_splitter.setCollapsible(0, False)
        image_splitter.setCollapsible(1, False)
        image_splitter.setStretchFactor(0, 1)
        image_splitter.setStretchFactor(1, 1)
        image_splitter.setSizes([590, 590])

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(image_splitter)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1160])
        self.setCentralWidget(splitter)

        self.pair_combo.currentIndexChanged.connect(self.on_combo_changed)
        self.prev_button.triggered.connect(lambda: self.load_pair(max(0, self.current_index - 1)))
        self.next_button.triggered.connect(lambda: self.load_pair(min(len(self.pairs) - 1, self.current_index + 1)))
        self.confirm_button.triggered.connect(self.confirm_prediction)
        self.cancel_button.triggered.connect(self.cancel_prediction)
        self.save_button.triggered.connect(self.save_annotation)
        self.export_button.triggered.connect(self.export_annotations)
        self.delete_match_button.clicked.connect(self.delete_selected_match)
        self.filter_edit.textChanged.connect(self.refresh_pair_widgets)
        self.pair_list.currentRowChanged.connect(self.on_pair_list_changed)
        self.match_list.currentRowChanged.connect(self.on_match_list_changed)

        for view in (self.panel_a.view, self.panel_b.view):
            view.pointPressed.connect(self.on_point_pressed)
            view.pointMoved.connect(self.on_point_moved)
            view.pointReleased.connect(self.on_point_released)

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, self.delete_selected_match)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace), self, self.delete_selected_match)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return), self, self.confirm_prediction)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self, self.cancel_prediction)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self, self.save_annotation)

    def apply_compact_style(self) -> None:
        font = self.font()
        font.setPointSize(UI_FONT_POINT_SIZE)
        self.setFont(font)
        self.setStyleSheet(
            f"""
            QWidget {{
                font-size: {UI_FONT_POINT_SIZE}pt;
            }}
            QToolBar {{
                spacing: 3px;
                padding: 1px 3px;
            }}
            QToolButton {{
                min-height: {CONTROL_HEIGHT}px;
                padding: 1px 7px;
            }}
            QLineEdit,
            QComboBox,
            QPushButton {{
                min-height: {CONTROL_HEIGHT}px;
                padding: 1px 5px;
            }}
            QListWidget {{
                outline: 0;
            }}
            QListWidget::item {{
                padding: 2px 4px;
            }}
            QStatusBar {{
                min-height: {CONTROL_HEIGHT}px;
            }}
            """
        )

    def set_feedback(self, message: str, timeout_ms: int = 5000) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    def update_coord_label(self, side: str, point: dict[str, float]) -> None:
        panel = self.panel_a if side == "a" else self.panel_b
        panel.coord_label.setText(format_point(point))

    def match_number(self, match_id: int) -> int:
        for index, match in enumerate(self.annotation.get("matches", [])):
            if int(match.get("id", 0)) == match_id:
                return index + 1
        return match_id

    def choose_pair_image(self, side: str) -> None:
        if side not in {"a", "b"}:
            return
        side_name = "左图" if side == "a" else "右图"
        title = f"选择{side_name}"
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            title,
            str(self.image_dialog_directory(side)),
            image_file_filter(),
        )
        if not file_name:
            return

        selected = self.relative_image_from_dialog(Path(file_name))
        if selected is None:
            return
        self.selected_pair_images[side] = selected
        self.show_selected_image_preview(side, selected)

        if self.selected_pair_images["a"] and self.selected_pair_images["b"]:
            self.load_selected_pair_images()
            return

        missing = "右图" if side == "a" else "左图"
        self.set_feedback(f"已选择{side_name}: {selected}；请继续选择{missing}")

    def image_dialog_directory(self, side: str) -> Path:
        selected = self.selected_pair_images.get(side)
        if selected:
            return annotator_data.require_inside_root(self.root, selected).parent
        if self.current_pair is not None:
            relative = self.current_pair.image_a if side == "a" else self.current_pair.image_b
            return annotator_data.require_inside_root(self.root, relative).parent
        return self.root

    def relative_image_from_dialog(self, path: Path) -> str | None:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "影像不在标注目录内",
                f"请选择标注根目录内的影像。\n\n标注根目录: {self.root}\n已选择: {resolved}",
            )
            return None
        if not annotator_data.is_image_file(resolved):
            QtWidgets.QMessageBox.warning(self, "不支持的文件", f"不是支持的影像文件: {resolved}")
            return None
        return annotator_data.relative_path(self.root, resolved)

    def show_selected_image_preview(self, side: str, relative: str) -> None:
        panel = self.panel_a if side == "a" else self.panel_b
        path = annotator_data.require_inside_root(self.root, relative)
        try:
            panel.name_label.setText(relative)
            panel.view.load_image(path, annotator_data.image_size(path))
            panel.view.redraw_points([], None, None)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "加载影像失败", str(exc))

    def load_selected_pair_images(self) -> None:
        image_a = self.selected_pair_images["a"]
        image_b = self.selected_pair_images["b"]
        if image_a is None or image_b is None:
            return
        try:
            pair = annotator_data.make_image_pair(self.root, image_a, image_b)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "无法载入图像对", str(exc))
            return
        index = self.upsert_pair(pair)
        self.load_pair(index)

    def upsert_pair(self, pair: annotator_data.ImagePair) -> int:
        for index, existing in enumerate(self.pairs):
            same_id = existing.pair_id == pair.pair_id
            same_images = existing.image_a == pair.image_a and existing.image_b == pair.image_b
            if same_id or same_images:
                self.pairs[index] = pair
                return index
        self.pairs.append(pair)
        return len(self.pairs) - 1

    def refresh_pair_widgets(self) -> None:
        query = self.filter_edit.text().strip().lower() if hasattr(self, "filter_edit") else ""
        self.filtered_pair_indices = [
            index
            for index, pair in enumerate(self.pairs)
            if not query or query in f"{pair.name} {pair.image_a} {pair.image_b}".lower()
        ]

        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        for index, pair in enumerate(self.pairs):
            self.pair_combo.addItem(f"{pair.name} ({pair.annotation_count})", index)
        self.pair_combo.blockSignals(False)
        self.pair_combo.setEnabled(bool(self.pairs))

        self.pair_list.blockSignals(True)
        self.pair_list.clear()
        for index in self.filtered_pair_indices:
            pair = self.pairs[index]
            item = QtWidgets.QListWidgetItem(f"{pair.name}  ·  {pair.annotation_count} 点\n{Path(pair.image_a).name} ↔ {Path(pair.image_b).name}")
            item.setSizeHint(QtCore.QSize(0, LIST_ROW_HEIGHT))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            self.pair_list.addItem(item)
            if index == self.current_index:
                self.pair_list.setCurrentItem(item)
        self.pair_list.blockSignals(False)
        if self.current_index >= 0:
            combo_index = self.pair_combo.findData(self.current_index)
            if combo_index >= 0:
                self.pair_combo.setCurrentIndex(combo_index)

    def on_combo_changed(self, combo_index: int) -> None:
        pair_index = self.pair_combo.itemData(combo_index)
        if isinstance(pair_index, int) and pair_index != self.current_index:
            self.load_pair(pair_index)

    def on_pair_list_changed(self, row: int) -> None:
        item = self.pair_list.item(row)
        if item is None:
            return
        pair_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(pair_index, int) and pair_index != self.current_index:
            self.load_pair(pair_index)

    def on_match_list_changed(self, row: int) -> None:
        matches = self.annotation.get("matches", [])
        if 0 <= row < len(matches):
            self.selected_id = int(matches[row].get("id", 0))
            self.set_feedback(f"已选中匹配点 #{row + 1}")
            self.redraw()
        elif row < 0:
            self.selected_id = None
            self.redraw()

    def load_pair(self, index: int) -> None:
        if index < 0 or index >= len(self.pairs):
            return
        if self.dirty:
            self.save_annotation()
        self.current_index = index
        self.current_pair = self.pairs[index]
        self.selected_pair_images["a"] = self.current_pair.image_a
        self.selected_pair_images["b"] = self.current_pair.image_b
        self.pending_a = None
        self.pending_b = None
        self.prediction_method = ""
        self.selected_id = None
        self.dragging = None

        path_a = annotator_data.require_inside_root(self.root, self.current_pair.image_a)
        path_b = annotator_data.require_inside_root(self.root, self.current_pair.image_b)
        self.panel_a.name_label.setText(self.current_pair.image_a)
        self.panel_b.name_label.setText(self.current_pair.image_b)
        self.panel_a.view.load_image(path_a, self.current_pair.size_a)
        self.panel_b.view.load_image(path_b, self.current_pair.size_b)
        self.annotation = annotator_data.load_annotation(self.root, self.current_pair.pair_id)
        self.annotation.setdefault("matches", [])
        self.dirty = False
        self.panel_a.coord_label.clear()
        self.panel_b.coord_label.clear()
        self.set_feedback(
            f"已加载图像对: {self.current_pair.image_a} ↔ {self.current_pair.image_b}；左键拖动影像，中键标点",
            5000,
        )
        self.refresh_pair_widgets()
        self.refresh_match_list()
        self.redraw()

    def refresh_match_list(self) -> None:
        self.match_list.blockSignals(True)
        self.match_list.clear()
        matches = self.annotation.get("matches", [])
        for index, match in enumerate(matches):
            item = QtWidgets.QListWidgetItem(
                f"#{index + 1}  A {match['a']['x']:.1f}, {match['a']['y']:.1f}\n"
                f"     B {match['b']['x']:.1f}, {match['b']['y']:.1f}"
            )
            item.setSizeHint(QtCore.QSize(0, LIST_ROW_HEIGHT))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, int(match.get("id", 0)))
            self.match_list.addItem(item)
            if match.get("id") == self.selected_id:
                self.match_list.setCurrentItem(item)
        self.match_list.blockSignals(False)
        self.update_match_actions()

    def update_match_actions(self) -> None:
        matches = self.annotation.get("matches", [])
        self.match_count_label.setText(f"{len(matches)} 点")
        self.delete_match_button.setEnabled(self.selected_id is not None)

    def redraw(self) -> None:
        matches = self.annotation.get("matches", [])
        self.panel_a.view.redraw_points(matches, self.pending_a, self.selected_id)
        self.panel_b.view.redraw_points(matches, self.pending_b, self.selected_id)
        self.confirm_button.setEnabled(bool(self.pending_a and self.pending_b))
        self.cancel_button.setEnabled(bool(self.pending_a or self.pending_b))
        self.update_match_actions()

    def nearest_match(self, point: dict[str, float], side: str) -> dict[str, Any] | None:
        threshold = 8.0
        best: dict[str, Any] | None = None
        best_distance = float("inf")
        for match in self.annotation.get("matches", []):
            distance = point_distance(point, match[side])
            if distance < threshold and distance < best_distance:
                best = match
                best_distance = distance
        return best

    def begin_prediction(self, point: dict[str, float]) -> None:
        if self.current_pair is None:
            return
        prediction = geometry_predictor.predict_match_point(
            point,
            self.annotation.get("matches", []),
            self.current_pair.size_a,
            self.current_pair.size_b,
        )
        self.pending_a = point
        self.pending_b = prediction.point
        self.prediction_method = prediction.method
        self.selected_id = None
        self.set_feedback(
            f"左图中键标点成功 A {format_point(point)}；已预测右图 B {format_point(prediction.point)} ({prediction.method})"
        )
        self.redraw()

    def confirm_prediction(self, point_b: dict[str, float] | None = None) -> None:
        if self.pending_a is None:
            return
        final_b = point_b or self.pending_b
        if final_b is None:
            return
        match = {
            "id": self.next_match_id(),
            "a": {"x": float(self.pending_a["x"]), "y": float(self.pending_a["y"])},
            "b": {"x": float(final_b["x"]), "y": float(final_b["y"])},
            "label": "match",
            "prediction_method": self.prediction_method,
        }
        self.annotation.setdefault("matches", []).append(match)
        self.pending_a = None
        self.pending_b = None
        self.prediction_method = ""
        self.selected_id = match["id"]
        self.mark_dirty()
        self.refresh_match_list()
        self.redraw()
        self.set_feedback(
            f"已添加匹配点 #{self.match_number(match['id'])}: A {format_point(match['a'])} ↔ B {format_point(match['b'])}"
        )

    def cancel_prediction(self) -> None:
        self.pending_a = None
        self.pending_b = None
        self.prediction_method = ""
        if self.current_pair is not None:
            self.set_feedback("已取消当前待确认点")
        self.redraw()

    def next_match_id(self) -> int:
        return max([0] + [int(match.get("id", 0)) for match in self.annotation.get("matches", [])]) + 1

    def on_point_pressed(self, side: str, x: float, y: float) -> None:
        point = {"x": x, "y": y}
        self.update_coord_label(side, point)
        if side == "b" and self.pending_a is not None:
            if self.pending_b is not None and point_distance(point, self.pending_b) <= 10.0:
                self.dragging = {"pending": True, "side": "b"}
                self.set_feedback(f"右图预测点已选中 B {format_point(point)}；中键拖动后松开即可确认")
            else:
                self.confirm_prediction(point)
            return

        hit = self.nearest_match(point, side)
        if hit is not None:
            self.selected_id = int(hit.get("id", 0))
            self.dragging = {"id": self.selected_id, "side": side}
            self.set_feedback(f"已选中匹配点 #{self.match_number(self.selected_id)}；中键拖动可修改{side_name(side)}坐标")
            self.refresh_match_list()
            self.redraw()
            return

        if side == "a":
            self.begin_prediction(point)
            return

        self.set_feedback("右图中键点击已收到；新增匹配需要先中键点击左图，再在右图确认")

    def on_point_moved(self, side: str, x: float, y: float) -> None:
        point = {"x": x, "y": y}
        self.update_coord_label(side, point)
        if not self.dragging or self.dragging.get("side") != side:
            return
        if self.dragging.get("pending"):
            self.pending_b = point
            self.redraw()
            return
        match_id = self.dragging.get("id")
        for match in self.annotation.get("matches", []):
            if match.get("id") == match_id:
                match[side] = point
                self.refresh_match_list()
                self.redraw()
                return

    def on_point_released(self, side: str) -> None:
        if not self.dragging or self.dragging.get("side") != side:
            return
        if self.dragging.get("pending"):
            self.dragging = None
            self.confirm_prediction(self.pending_b)
            return
        match_id = int(self.dragging.get("id", 0))
        self.dragging = None
        self.mark_dirty()
        self.set_feedback(f"已移动匹配点 #{self.match_number(match_id)} 的{side_name(side)}坐标")

    def delete_selected_match(self) -> None:
        if self.selected_id is None:
            self.set_feedback("请先在图像或匹配点列表中选中一个点对")
            return
        deleted_id = self.selected_id
        deleted_number = self.match_number(deleted_id)
        self.annotation["matches"] = [
            match for match in self.annotation.get("matches", []) if int(match.get("id", 0)) != deleted_id
        ]
        self.selected_id = None
        self.mark_dirty()
        self.refresh_match_list()
        self.redraw()
        self.set_feedback(f"已删除匹配点 #{deleted_number}")

    def mark_dirty(self) -> None:
        self.dirty = True
        self.autosave_timer.start(700)

    def save_annotation(self) -> None:
        if self.current_pair is None:
            return
        payload = {
            "pair_id": self.current_pair.pair_id,
            "image_a": self.current_pair.image_a,
            "image_b": self.current_pair.image_b,
            "matches": self.annotation.get("matches", []),
        }
        path = annotator_data.save_annotation(self.root, payload)
        self.dirty = False
        self.current_pair = annotator_data.make_image_pair(
            self.root,
            self.current_pair.image_a,
            self.current_pair.image_b,
            pair_id=self.current_pair.pair_id,
            name=self.current_pair.name,
        )
        self.current_index = self.upsert_pair(self.current_pair)
        self.statusBar().showMessage(f"已保存: {path.relative_to(self.root)}")
        self.refresh_pair_widgets()

    def export_annotations(self) -> None:
        output = self.root / "annotations_export.json"
        output.write_text(
            json.dumps(annotator_data.export_annotations(self.root), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.statusBar().showMessage(f"已导出: {output}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.dirty:
            self.save_annotation()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Native GUI for image-pair correspondence annotation.")
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path, help="Annotation data directory")
    parser.add_argument("--list-pairs", action="store_true", help="List discovered pairs and exit")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.list_pairs:
        for pair in annotator_data.discover_pairs(root):
            print(f"{pair.pair_id}\t{pair.image_a}\t{pair.image_b}\t{pair.annotation_count}")
        return 0

    app = QtWidgets.QApplication(sys.argv)
    window = AnnotatorWindow(root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
