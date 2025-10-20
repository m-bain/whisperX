"""
Custom widget for displaying transcription history items with rich styling.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class HistoryItemWidget(QWidget):
    """Custom widget for displaying a transcription history item."""

    def __init__(self, filename, timestamp, file_size_mb, transcription_time, status='success', parent=None):
        super().__init__(parent)
        self.setObjectName("historyItemBox")

        self.status = status

        # Set up the layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(4)

        # First row: Filename with status icon
        filename_layout = QHBoxLayout()
        filename_layout.setSpacing(8)

        status_icon = "" if status == 'success' else ""
        self.filename_label = QLabel(f"{status_icon} {filename}")
        filename_font = QFont()
        filename_font.setBold(True)
        filename_font.setPointSize(10)
        self.filename_label.setFont(filename_font)
        filename_layout.addWidget(self.filename_label)
        filename_layout.addStretch()

        main_layout.addLayout(filename_layout)

        # Second row: Timestamp
        self.timestamp_label = QLabel(f"{timestamp}")
        timestamp_font = QFont()
        timestamp_font.setPointSize(9)
        self.timestamp_label.setFont(timestamp_font)
        main_layout.addWidget(self.timestamp_label)

        # Third row: File size and transcription time
        info_layout = QHBoxLayout()
        info_layout.setSpacing(15)

        self.size_label = QLabel(f"{file_size_mb:.1f} MB")
        info_font = QFont()
        info_font.setPointSize(9)
        self.size_label.setFont(info_font)

        self.time_label = QLabel(f"{transcription_time:.1f}s")
        self.time_label.setFont(info_font)

        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.time_label)
        info_layout.addStretch()

        main_layout.addLayout(info_layout)

        # Apply styling
        self._apply_styling()

    def _apply_styling(self):
        """Apply modern card-style CSS styling."""
        print("PRINTING ", self.status, "   ", str(self))
        if self.status == 'success':
            # Success state - clean card design
            self.setStyleSheet("""
                #historyItemBox {
                    background-color:red !important;
                }
                HistoryItemWidget {
                    background-color: red !important;
                    border: 1px solid #3e3e42;
                    border-left: 4px solid #007acc;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 2px;
                }
                HistoryItemWidget:hover {
                    background-color: red !important;
                    border-left: 4px solid #1e8ad6;
                    border: 1px solid #4e4e52;
                }
                QLabel:hover {
                    color: #cccccc;
                    background-color: transparent;
                    border: none;
                }
            """)
        else:
            # Error state - red accent
            self.setStyleSheet("""
                #historyItemBox {
                    background-color:red !important;
                }
                
                HistoryItemWidget {
                    background-color: red !important;
                    border: 1px solid #3e3e42;
                    border-left: 4px solid #d13438;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 2px;
                }
                HistoryItemWidget:hover {
                    background-color: red !important;
                    border-left: 4px solid #e6484e;
                    border: 1px solid #4e4e52;
                }
                QLabel {
                    color: #cccccc;
                    background-color: transparent;
                    border: none;
                }
            """)

        # Set consistent sizing
        self.setMinimumHeight(90)
        self.setMaximumHeight(100)