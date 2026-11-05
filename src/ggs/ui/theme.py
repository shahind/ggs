"""Look of the app: one dark theme, rounded cards, one accent colour."""

ACCENT = "#7c8cff"
BG = "#1b1c21"
PANEL = "#23252c"
CARD = "#2b2e36"
LINE = "#363944"
TEXT = "#e8e9ee"
MUTED = "#9a9db0"
GOOD = "#3fb27f"
BAD = "#e0646a"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog, QWidget#root, QStackedWidget, QScrollArea#root {{ background: {BG}; }}
QWidget#panel, QFrame#panel {{ background: {PANEL}; border-radius: 10px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#card:hover {{ border-color: {ACCENT}; }}
QLabel#h1 {{ font-size: 20pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 13pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#chip {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 9px; padding: 1px 8px; color: {MUTED}; }}
QPushButton {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {CARD}; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: #10121a; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {LINE}; color: {MUTED}; }}
QPushButton#flat {{ background: transparent; border: none; color: {MUTED}; }}
QPushButton#flat:hover {{ color: {TEXT}; }}
QPushButton#tile {{ background: {CARD}; border: 2px solid {LINE}; border-radius: 12px; padding: 10px; }}
QPushButton#tile:checked {{ border-color: {ACCENT}; background: #313552; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 5px 8px;
    selection-background-color: {ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QTreeWidget, QListWidget {{ background: {PANEL}; border: none; border-radius: 10px; padding: 4px; outline: 0; }}
QTreeWidget::item, QListWidget::item {{ padding: 4px 2px; border-radius: 6px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: #313552; color: {TEXT}; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: #2a2d38; }}
QProgressBar {{ background: {CARD}; border: none; border-radius: 3px; height: 6px; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {LINE}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
"""
