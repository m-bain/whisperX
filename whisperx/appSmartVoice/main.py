########################################################################
## QT GUI BY SPINN TV(YOUTUBE)
########################################################################

########################################################################
## IMPORTS
########################################################################
import os
import sys
from datetime import datetime
########################################################################
# IMPORT GUI FILE
from src.ui_interface import *
########################################################################

########################################################################
# IMPORT Custom widgets
from Custom_Widgets import *
from Custom_Widgets.QAppSettings import QAppSettings
from Custom_Widgets.QCustomQToolTip import QCustomQToolTipFilter
########################################################################
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QFileDialog, QMessageBox

########################################################################
## MAIN WINDOW CLASS
########################################################################
from whisperx.app.transcription_manager import TranscriptionManager
from whisperx.appSmartVoice.src.gui_functions import GUIFunctions
from whisperx.SubtitlesProcessor import SubtitlesProcessor

from whisperx.app.history_manager import HistoryManager, TranscriptionRecord
from whisperx.appSmartVoice.src.history_item_widget import HistoryItemWidget

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ########################
        # ADDING WORKERS FOR TRANSCRIPTION PROCESSING AND MODEL LOADING
        ##########
        self.transcriptionResult = None
        self.lastAudioFile = None
        self.transcriptionActive = False

        self.threadPool = QThreadPool()
        self.transcriptionManager = TranscriptionManager(self.threadPool)

        # transcription manager signals
        self.transcriptionManager.progress_updated.connect(self.onProgressUpdate)
        self.transcriptionManager.status_updated.connect(self.onStatusUpdate)
        self.transcriptionManager.transcription_completed.connect(self.onTranscriptionCompleted)
        self.transcriptionManager.error_occurred.connect(self.onTranscriptionError)
        self.transcriptionManager.models_loaded.connect(self.onModelsLoaded)

        ########################
        # HISTORY MANAGER
        ########################
        try:
            self.historyManager = HistoryManager()
        except Exception as e:
            print(f"Warning: Could not initialize history manager: {e}")
            self.historyManager = None

        # Initialize transcription display variables
        self.fullTranscription = ""
        self.timestampedTranscription = ""
        self.speakerTranscription = ""
        self.rawTranscription = ""

        ########################################################################
        # APPLY JSON STYLESHEET
        ########################################################################
        # self = QMainWindow class
        # self.ui = Ui_MainWindow / user interface class
        #Use this if you only have one json file named "style.json" inside the root directory, "json" directory or "jsonstyles" folder.
        # loadJsonStyle(self, self.ui) 

        # Use this to specify your json file(s) path/name
        loadJsonStyle(self, self.ui, jsonFiles = {
            "json-styles/style.json"
            }) 

        ########################################################################

        #######################################################################
        # SHOW WINDOW
        #######################################################################
        self.show()

        ########################################################################
        # UPDATE APP SETTINGS LOADED FROM JSON STYLESHEET 
        # ITS IMPORTANT TO RUN THIS AFTER SHOWING THE WINDOW
        # THIS PROCESS WILL RUN ON A SEPARATE THREAD WHEN GENERATING NEW ICONS
        # TO PREVENT THE WINDOW FROM BEING UNRESPONSIVE
        ########################################################################
        # self = QMainWindow class
        QAppSettings.updateAppSettings(self)

        self.app_functions = GUIFunctions(self)

        ########################
        # HISTORY LIST CONNECTIONS
        ########################
        self.ui.transcrHistoryList.itemClicked.connect(self._onHistoryItemClicked)
        self.ui.transcrHistoryList.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.transcrHistoryList.customContextMenuRequested.connect(self._showHistoryContextMenu)

        # Load initial history
        if self.historyManager:
            self._refreshHistoryList()

        # self.ui.transcrHistoryList.setStyleSheet("""
        #             QListWidget {
        #                 background-color: #1e1e1e;
        #                 border: 1px solid #3e3e42;
        #                 border-radius: 5px;
        #                 outline: none;
        #                 padding: 5px;
        #             }
        #             QListWidget::item {
        #                 background-color: transparent;
        #                 border: none;
        #                 padding: 2px;
        #                 margin: 3px;
        #             }
        #             QListWidget::item:selected {
        #                 background-color: transparent;
        #                 border: none;
        #             }
        #             QListWidget::item:hover {
        #                 background-color: transparent;
        #             }
        #         """)
        # self.ui.transcrHistoryList.setSpacing(5)
        # self.ui.transcrHistoryList.setUniformItemSizes(False)

    def onProgressUpdate(self, progress):
        """Handle progress updates from transcription manager."""
        self.ui.progressBar_2.setValue(progress)

    def onStatusUpdate(self, status):
        """Handle status updates from transcription manager."""
        self.ui.statusLabel.setText(status)

    def onTranscriptionCompleted(self, result):
        """Handle transcription completion."""
        self.transcriptionActive = False

        # Store the complete transcription result for SRT download
        self.transcriptionResult = result

        # Store the audio file path for naming the SRT file
        if hasattr(self, 'app_functions') and hasattr(self.app_functions, 'ui'):
            self.lastAudioFile = self.ui.lineEdit.text().strip()

        # Update transcription display with results
        if 'formatted' in result:
            formatted = result['formatted']
            self.fullTranscription = formatted.get('full', 'Transcription completed')
            self.timestampedTranscription = formatted.get('timestamped', 'Transcription completed')
            self.speakerTranscription = formatted.get('speakers', 'Transcription completed')
            self.rawTranscription = formatted.get('raw', 'Transcription completed')

        # Update display
        self.updateTranscriptionDisplay()
        self.ui.statusLabel.setText("Transcription completed successfully")

        self._saveTranscriptionToHistory(result)

    def onTranscriptionError(self, error_message):
        """Handle transcription errors."""
        self.transcriptionActive = False
        self.ui.progressBar_2.setValue(0)
        self.ui.statusLabel.setText(f"Error: {error_message}")

        # Show error in transcription area
        self.ui.transcriptionTextArea.setPlainText(f"Transcription failed: {error_message}")

    def onModelsLoaded(self):
        """Handle model loading completion."""
        self.ui.statusLabel.setText("Models loaded, starting transcription...")

    def updateTranscriptionDisplay(self):
        showTimestamps = self.ui.showTimestamps.isChecked()
        showSpeakers = self.ui.showSpeakers.isChecked()

        if showTimestamps and showSpeakers:
            # Show full transcription with both timestamps and speakers
            self.ui.transcriptionTextArea.setPlainText(self.fullTranscription)
        elif showTimestamps:
            # Show only timestamps
            self.ui.transcriptionTextArea.setPlainText(self.timestampedTranscription)
        elif showSpeakers:
            # Show only speakers
            self.ui.transcriptionTextArea.setPlainText(self.speakerTranscription)
        else:
            # Show raw transcription
            self.ui.transcriptionTextArea.setPlainText(self.rawTranscription)

        ########################################################################
        # HISTORY MANAGEMENT METHODS
        ########################################################################

    def _saveTranscriptionToHistory(self, result):
        """Save completed transcription to history database."""
        if not self.historyManager:
            return

        try:
            # Extract data from result
            filepath = result.get('filepath', '')
            file_size = result.get('file_size', 0)
            transcription_time = result.get('elapsed_time', 0.0)

            # Get config from result or app_config
            config_dict = result.get('config', {})
            if not config_dict:
                config = self.transcriptionManager.get_current_config()
                config_dict = {
                    'model_name': config.model_name,
                    'language': config.language,
                    'enable_alignment': config.enable_alignment,
                    'enable_diarization': config.enable_diarization
                }

            # Extract formatted results
            formatted = result.get('formatted', {})

            # Create record
            record = TranscriptionRecord(
                timestamp=datetime.now().isoformat(),
                filepath=filepath,
                file_size=file_size,
                transcription_time=transcription_time,
                model_name=config_dict.get('model_name', 'unknown'),
                language=config_dict.get('language'),
                alignment_enabled=config_dict.get('enable_alignment', False),
                diarization_enabled=config_dict.get('enable_diarization', False),
                result_raw=formatted.get('raw'),
                result_timestamped=formatted.get('timestamped'),
                result_speakers=formatted.get('speakers'),
                result_full=formatted.get('full'),
                status='success'
            )

            # Save to database
            record_id = self.historyManager.add_transcription(record)
            print(f"Saved transcription to history with ID: {record_id}")

            # Refresh history list
            self._refreshHistoryList()

        except Exception as e:
            print(f"Error saving transcription to history: {e}")

    def _refreshHistoryList(self):
        """Refresh the history list widget with data from database."""
        if not self.historyManager:
            return

        # IMPORT THE NEW WIDGET CLASS AT THE TOP OF THIS METHOD:
        from whisperx.appSmartVoice.src.history_item_widget import HistoryItemWidget

        try:
            # Clear existing items
            self.ui.transcrHistoryList.clear()

            # Get recent transcriptions
            records = self.historyManager.get_all_transcriptions(limit=50)

            if not records:
                # Show empty state
                item = QListWidgetItem("No transcriptions yet")
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
                self.ui.transcrHistoryList.addItem(item)
                return

            # Add items for each record using custom widget
            for record in records:
                # Create list item
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, record.id)  # Store record ID

                # Create custom widget
                filename = record.get_filename()
                timestamp = record.get_formatted_timestamp()
                file_size_mb = record.get_file_size_mb()

                widget = HistoryItemWidget(
                    filename=filename,
                    timestamp=timestamp,
                    file_size_mb=file_size_mb,
                    transcription_time=record.transcription_time,
                    status=record.status
                )

                # Add item to list
                self.ui.transcrHistoryList.addItem(item)

                # Set the custom widget for this item
                item.setSizeHint(widget.sizeHint())
                self.ui.transcrHistoryList.setItemWidget(item, widget)

        except Exception as e:
            print(f"Error refreshing history list: {e}")

    def _onHistoryItemClicked(self, item):
        """Handle click on history list item to restore transcription."""
        # Get record ID from item data
        record_id = item.data(Qt.ItemDataRole.UserRole)

        if record_id is None:
            return  # Empty state item

        try:
            # Retrieve record from database
            record = self.historyManager.get_transcription_by_id(record_id)

            if not record:
                self.ui.statusLabel.setText("Error: Could not load transcription")
                return

            # Restore transcription to display
            self.fullTranscription = record.result_full or ''
            self.timestampedTranscription = record.result_timestamped or ''
            self.speakerTranscription = record.result_speakers or ''
            self.rawTranscription = record.result_raw or ''

            # Update display
            self.updateTranscriptionDisplay()

            # Update status with metadata
            status_text = f"Loaded: {record.get_filename()} | "
            status_text += f"Model: {record.model_name} | "
            status_text += f"Time: {record.transcription_time:.1f}s"
            self.ui.statusLabel.setText(status_text)

            # Switch to results page
            # self.ui.stackedMainPages.setCurrentIndex(1)
            # Switch to results page - find it by object name
            for i in range(self.ui.stackedMainPages.count()):
                widget = self.ui.stackedMainPages.widget(i)
                if widget.objectName() == "resultsPage":
                    self.ui.stackedMainPages.setCurrentIndex(i)
                    break

            # Enable results button
            self.ui.resultsBtn.setEnabled(True)

        except Exception as e:
            print(f"Error loading history item: {e}")
            self.ui.statusLabel.setText(f"Error loading transcription: {e}")

    def _showHistoryContextMenu(self, pos):
        """Show context menu for history list item."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction

        # Get the item at the position
        item = self.ui.transcrHistoryList.itemAt(pos)

        if not item:
            return

        record_id = item.data(Qt.ItemDataRole.UserRole)

        if record_id is None:
            return  # Empty state item

        # Create context menu
        menu = QMenu(self)

        # Add actions
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._deleteHistoryItem(record_id))
        menu.addAction(delete_action)

        view_details_action = QAction("View Details", self)
        view_details_action.triggered.connect(lambda: self._viewHistoryDetails(record_id))
        menu.addAction(view_details_action)

        export_action = QAction("Export...", self)
        export_action.triggered.connect(lambda: self._exportHistoryItem(record_id))
        menu.addAction(export_action)

        # Show menu at cursor position
        menu.exec(self.ui.transcrHistoryList.mapToGlobal(pos))

    def _deleteHistoryItem(self, record_id):
        """Delete a history item."""
        try:
            if self.historyManager.delete_transcription(record_id):
                self._refreshHistoryList()
                self.ui.statusLabel.setText("Transcription deleted from history")
            else:
                self.ui.statusLabel.setText("Error: Could not delete transcription")
        except Exception as e:
            print(f"Error deleting history item: {e}")
            self.ui.statusLabel.setText(f"Error: {e}")

    def _viewHistoryDetails(self, record_id):
        """Show detailed information about a history item."""
        from PySide6.QtWidgets import QMessageBox

        try:
            record = self.historyManager.get_transcription_by_id(record_id)

            if not record:
                return

            # Format details
            details = f"File: {record.filepath}\n"
            details += f"Timestamp: {record.get_formatted_timestamp()}\n"
            details += f"File Size: {record.get_file_size_mb():.2f} MB\n"
            details += f"Transcription Time: {record.transcription_time:.2f}s\n"
            details += f"Model: {record.model_name}\n"
            details += f"Language: {record.language or 'Auto-detected'}\n"
            details += f"Alignment: {'Yes' if record.alignment_enabled else 'No'}\n"
            details += f"Diarization: {'Yes' if record.diarization_enabled else 'No'}\n"
            details += f"Status: {record.status}\n"

            if record.error_message:
                details += f"\nError: {record.error_message}"

            # Show dialog
            QMessageBox.information(self, "Transcription Details", details)

        except Exception as e:
            print(f"Error showing history details: {e}")

    def _exportHistoryItem(self, record_id):
        """Export a history item to a text file."""
        from PySide6.QtWidgets import QFileDialog

        try:
            record = self.historyManager.get_transcription_by_id(record_id)

            if not record:
                return

            # Show file save dialog
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Transcription",
                f"{record.get_filename()}_transcription.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Transcription of: {record.filepath}\n")
                    f.write(f"Date: {record.get_formatted_timestamp()}\n")
                    f.write(f"Model: {record.model_name}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(record.result_full or record.result_raw or "No transcription available")

                self.ui.statusLabel.setText(f"Exported to: {filename}")

        except Exception as e:
            print(f"Error exporting history item: {e}")
            self.ui.statusLabel.setText(f"Error exporting: {e}")

    def downloadSRTFile(self):
        """Generate and download an SRT file containing the final transcription."""
        if not hasattr(self, 'transcriptionResult') or not self.transcriptionResult:
            QMessageBox.warning(self, "No Transcription", "No transcription data available to download.")
            return

        try:
            # Get save location from user
            suggested_filename = "transcription.srt"
            if hasattr(self, 'lastAudioFile') and self.lastAudioFile:
                base_name = os.path.splitext(os.path.basename(self.lastAudioFile))[0]
                suggested_filename = f"{base_name}_transcription.srt"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save SRT File",
                suggested_filename,
                "SRT Files (*.srt);;All Files (*)"
            )

            if not file_path:
                return  # User cancelled

            # Ensure .srt extension
            if not file_path.lower().endswith('.srt'):
                file_path += '.srt'

            # Get the segments from transcription result
            segments = self._getTranscriptionSegments()
            if not segments:
                QMessageBox.warning(self, "No Data", "No transcription segments available.")
                return

            # Write SRT file directly
            from whisperx.utils import format_timestamp

            subtitle_count = 0
            with open(file_path, 'w', encoding='utf-8') as f:
                for idx, segment in enumerate(segments, start=1):
                    # Get segment data
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '').strip()

                    # Skip empty segments
                    if not text:
                        continue

                    # Format timestamps in SRT format (HH:MM:SS,mmm)
                    start_srt = format_timestamp(start_time, always_include_hours=True, decimal_marker=',')
                    end_srt = format_timestamp(end_time, always_include_hours=True, decimal_marker=',')

                    # Add speaker if available
                    speaker = segment.get('speaker')
                    if speaker:
                        text = f"[{speaker}]: {text}"

                    # Write SRT entry
                    f.write(f"{subtitle_count + 1}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{text}\n\n")

                    subtitle_count += 1

            QMessageBox.information(
                self,
                "SRT Downloaded",
                f"SRT file saved successfully!\nLocation: {file_path}\nSubtitles: {subtitle_count}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save SRT file:\n{str(e)}")

    # def downloadSRTFile(self):
    #     """Generate and download an SRT file containing the final transcription."""
    #     if not hasattr(self, 'transcriptionResult') or not self.transcriptionResult:
    #         QMessageBox.warning(self, "No Transcription", "No transcription data available to download.")
    #         return
    #
    #     try:
    #         # Get save location from user
    #         suggested_filename = "transcription.srt"
    #         if hasattr(self, 'lastAudioFile') and self.lastAudioFile:
    #             base_name = os.path.splitext(os.path.basename(self.lastAudioFile))[0]
    #             suggested_filename = f"{base_name}_transcription.srt"
    #
    #         file_path, _ = QFileDialog.getSaveFileName(
    #             self,
    #             "Save SRT File",
    #             suggested_filename,
    #             "SRT Files (*.srt);;All Files (*)"
    #         )
    #
    #         if not file_path:
    #             return  # User cancelled
    #
    #         # Ensure .srt extension
    #         if not file_path.lower().endswith('.srt'):
    #             file_path += '.srt'
    #
    #         # Get the segments from transcription result
    #         segments = self._getTranscriptionSegments()
    #         if not segments:
    #             QMessageBox.warning(self, "No Data", "No transcription segments available.")
    #             return
    #
    #         # Detect language from transcription result
    #         language = self.transcriptionResult.get('transcription', {}).get('language', 'en')
    #         if not language:
    #             # Try other possible locations for language
    #             language = (self.transcriptionResult.get('aligned_transcription', {}).get('language') or
    #                         self.transcriptionResult.get('segments_with_speakers', {}).get('language') or 'en')
    #
    #         # Create SubtitlesProcessor and generate SRT
    #         processor = SubtitlesProcessor(segments, language, is_vtt=False)
    #         subtitle_count = processor.save(file_path, advanced_splitting=False)
    #
    #         QMessageBox.information(
    #             self,
    #             "SRT Downloaded",
    #             f"SRT file saved successfully!\nLocation: {file_path}\nSubtitles: {subtitle_count}"
    #         )
    #
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to save SRT file:\n{str(e)}")

    def _getTranscriptionSegments(self):
        """Extract segments from transcription result for SRT generation."""
        if not self.transcriptionResult:
            return None

        # Try to get segments with speakers first (most complete), then aligned, then basic
        if 'segments_with_speakers' in self.transcriptionResult:
            segments_data = self.transcriptionResult['segments_with_speakers']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                return segments_data['segments']
            return segments_data

        elif 'aligned_transcription' in self.transcriptionResult:
            segments_data = self.transcriptionResult['aligned_transcription']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                return segments_data['segments']
            return segments_data

        elif 'transcription' in self.transcriptionResult:
            segments_data = self.transcriptionResult['transcription']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                return segments_data['segments']
            return segments_data

        return None
########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ########################################################################
    ##
    # app_tooltip_filter = QCustomQToolTipFilter(tailPosition="bottom-right")
    # app.installEventFilter(app_tooltip_filter)
    # LOOOOL
    ########################################################################
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
########################################################################
## END===>
########################################################################  
