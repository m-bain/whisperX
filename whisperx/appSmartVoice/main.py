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

########################################################################
## MAIN WINDOW CLASS
########################################################################
from whisperx.app.transcription_manager import TranscriptionManager
from whisperx.appSmartVoice.src.gui_functions import GUIFunctions


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ########################
        # ADDING WORKERS FOR TRANSCRIPTION PROCESSING AND MODEL LOADING
        ##########
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
