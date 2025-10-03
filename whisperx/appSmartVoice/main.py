########################################################################
## QT GUI BY SPINN TV(YOUTUBE)
########################################################################

########################################################################
## IMPORTS
########################################################################
import os
import sys
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
