import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel, QPushButton, \
    QHBoxLayout, QLineEdit, QFileDialog, QCheckBox, QGroupBox, QComboBox, QSpacerItem, QSizePolicy, QProgressBar, \
    QTextEdit, QMenu
from PySide6.QtCore import Qt

# Enable high-DPI scaling
QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

from PySide6.QtCore import QThreadPool
from transcription_manager import TranscriptionManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.fullTranscription = "[0:0 - 1:78] Speaker 1: Full Transcription"
        self.timestampedTranscription = "[0:0 - 1:78]: Full Transcription"
        self.speakerTranscription = "Speaker 1: Full Transcription"
        self.rawTranscription = "Full Transcription"


        self.transcriptionActive = False

        self.threadPool = QThreadPool()
        self.transcriptionManager = TranscriptionManager(self.threadPool)

        # transcription manager signals
        self.transcriptionManager.progress_updated.connect(self.onProgressUpdate)
        self.transcriptionManager.status_updated.connect(self.onStatusUpdate)
        self.transcriptionManager.transcription_completed.connect(self.onTranscriptionCompleted)
        self.transcriptionManager.error_occurred.connect(self.onTranscriptionError)
        self.transcriptionManager.models_loaded.connect(self.onModelsLoaded)


        self.mainWindow = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWindow)
        self.setCentralWidget(self.mainWindow)

        # nav buttons
        navLayout = QHBoxLayout()
        navLayout.addSpacerItem(QSpacerItem(40,20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.setupPageButton = QPushButton("Setup")
        self.resultsPageButton = QPushButton("Results")
        self.resultsPageButton.setEnabled(False)

        self.setupPageButton.clicked.connect(lambda: self.switchToPage(0))
        self.resultsPageButton.clicked.connect(lambda: self.switchToPage(1))

        navLayout.addWidget(self.setupPageButton)
        navLayout.addWidget(self.resultsPageButton)

        # pages

        self.stackedWidget = QStackedWidget()

        self.setupPage = self.createSetupPage() # index 0
        self.resultPage = self.createResultPage() # index 1

        self.stackedWidget.addWidget(self.setupPage)
        self.stackedWidget.addWidget(self.resultPage)

        self.mainLayout.addLayout(navLayout)
        self.mainLayout.addWidget(self.stackedWidget)
        self.switchToPage(0)


    def createSetupPage(self):
        page = QWidget()
        layout = QVBoxLayout(page) # set it directly

        # File selection section
        self.createFileSelectionSection(layout)
        # Settings section
        self.createSettingsSection(layout)

        self.runTranscriptionButton = QPushButton("Run transcription")
        self.runTranscriptionButton.setEnabled(False)

        self.runTranscriptionButton.clicked.connect(self.runTranscription)

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.runTranscriptionButton)
        buttonLayout.addStretch()

        layout.addLayout(buttonLayout)
        layout.addStretch()

        return page

    def createResultPage(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.createProgressSection(layout)

        self.createTextResultSection(layout)

        layout.addStretch() # Push up

        return page

    def createProgressSection(self, parentLayout):
        progressGroup = QGroupBox("Progress")
        progressLayout = QVBoxLayout(progressGroup)

        self.progressResultsBar = QProgressBar()
        self.progressResultsBar.setRange(0, 100)
        self.progressResultsBar.setValue(0)
        progressLayout.addWidget(self.progressResultsBar)

        self.statusLabel = QLabel("Starting transcription")
        self.statusLabel.setWordWrap(True)
        progressLayout.addWidget(self.statusLabel)

        parentLayout.addWidget(progressGroup)


    def createTextResultSection(self, parentLayout):
        textDisplayOptionsGroup = QGroupBox("Display Options")
        textDisplayOptionsLayout = QHBoxLayout(textDisplayOptionsGroup)

        self.showTimestampsCheckbox = QCheckBox("Timestamps")
        self.showTimestampsCheckbox.stateChanged.connect(self.updateTranscriptionDisplay)

        self.showSpeakersCheckbox = QCheckBox("Speakers")
        self.showSpeakersCheckbox.stateChanged.connect(self.updateTranscriptionDisplay)

        textDisplayOptionsLayout.addWidget(self.showTimestampsCheckbox)
        textDisplayOptionsLayout.addWidget(self.showSpeakersCheckbox)
        textDisplayOptionsLayout.addStretch() # Push left

        parentLayout.addWidget(textDisplayOptionsGroup)

        # Textarea
        transcriptionGroup = QGroupBox("Transcription Results")
        transcriptionLayout = QVBoxLayout(transcriptionGroup)

        self.transcriptionTextArea = QTextEdit()
        self.transcriptionTextArea.setReadOnly(True)
        self.transcriptionTextArea.setPlaceholderText("Transcription results will appear here...")
        self.transcriptionTextArea.setMinimumHeight(400)

        transcriptionLayout.addWidget(self.transcriptionTextArea)
        parentLayout.addWidget(transcriptionGroup)


    def createSettingsSection(self, parentLayout):
        settingsGroup = QGroupBox("Settings")
        settingsLayout = QVBoxLayout(settingsGroup)

        self.alignTimestampsCheckbox = QCheckBox("Enable timestamp alignment")
        settingsLayout.addWidget(self.alignTimestampsCheckbox)

        self.diarizationChecbox = QCheckBox("Enable speaker diarization")
        settingsLayout.addWidget(self.diarizationChecbox)

        languageLayout = QHBoxLayout()
        languageLayout.addWidget(QLabel("Language (optional):"))

        # TODO:
        self.languageDropdown = QPushButton("Select language...")
        self.languageDropdown.setStyleSheet("text-align: left; padding: 5px;")
        self.languageMenu = QMenu(self.languageDropdown)
        self.setupLanguageDropdown()

        languageLayout.addWidget(self.languageDropdown)
        languageLayout.addStretch()
        settingsLayout.addLayout(languageLayout)

        parentLayout.addWidget(settingsGroup)

    def updateTranscriptionDisplay(self):
        showTimestamps = self.showTimestampsCheckbox.isChecked()
        showSpeakers = self.showSpeakersCheckbox.isChecked()

        if showTimestamps and showSpeakers:
            # Show full transcription with both timestamps and speakers
            self.transcriptionTextArea.setPlainText(self.fullTranscription)
        elif showTimestamps:
            # Show only timestamps
            self.transcriptionTextArea.setPlainText(self.timestampedTranscription)
        elif showSpeakers:
            # Show only speakers
            self.transcriptionTextArea.setPlainText(self.speakerTranscription)
        else:
            # Show raw transcription
            self.transcriptionTextArea.setPlainText(self.rawTranscription)

    def switchToPage(self, index):
        self.stackedWidget.setCurrentIndex(index)
        # Update button states to show current page
        self.setupPageButton.setEnabled(index != 0)
        self.resultsPageButton.setEnabled(self.transcriptionActive and index != 1)

    def setupLanguageDropdown(self):
        # Language mapping: Display Name -> Language Code
        # Complete Whisper language mapping: Display Name -> Language Code
        self.languageMapping = {
            "": "",  # Empty option for unselected
            "Afrikaans": "af",
            "Albanian": "sq",
            "Amharic": "am",
            "Arabic": "ar",
            "Armenian": "hy",
            "Assamese": "as",
            "Azerbaijani": "az",
            "Bashkir": "ba",
            "Basque": "eu",
            "Belarusian": "be",
            "Bengali": "bn",
            "Bosnian": "bs",
            "Breton": "br",
            "Bulgarian": "bg",
            "Burmese": "my",
            "Castilian": "es",
            "Catalan": "ca",
            "Chinese": "zh",
            "Croatian": "hr",
            "Czech": "cs",
            "Danish": "da",
            "Dutch": "nl",
            "English": "en",
            "Estonian": "et",
            "Faroese": "fo",
            "Finnish": "fi",
            "Flemish": "nl",
            "French": "fr",
            "Galician": "gl",
            "Georgian": "ka",
            "German": "de",
            "Greek": "el",
            "Gujarati": "gu",
            "Haitian": "ht",
            "Haitian Creole": "ht",
            "Hausa": "ha",
            "Hawaiian": "haw",
            "Hebrew": "he",
            "Hindi": "hi",
            "Hungarian": "hu",
            "Icelandic": "is",
            "Indonesian": "id",
            "Irish": "ga",
            "Italian": "it",
            "Japanese": "ja",
            "Javanese": "jv",
            "Kannada": "kn",
            "Kazakh": "kk",
            "Khmer": "km",
            "Korean": "ko",
            "Lao": "lo",
            "Latin": "la",
            "Latvian": "lv",
            "Lingala": "ln",
            "Lithuanian": "lt",
            "Luxembourgish": "lb",
            "Macedonian": "mk",
            "Malagasy": "mg",
            "Malay": "ms",
            "Malayalam": "ml",
            "Maltese": "mt",
            "Mandarin": "zh",
            "Maori": "mi",
            "Marathi": "mr",
            "Mongolian": "mn",
            "Nepali": "ne",
            "Norwegian": "no",
            "Nynorsk": "nn",
            "Occitan": "oc",
            "Pashto": "ps",
            "Persian": "fa",
            "Polish": "pl",
            "Portuguese": "pt",
            "Punjabi": "pa",
            "Romanian": "ro",
            "Russian": "ru",
            "Sanskrit": "sa",
            "Serbian": "sr",
            "Shona": "sn",
            "Sindhi": "sd",
            "Sinhala": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Spanish": "es",
            "Sundanese": "su",
            "Swahili": "sw",
            "Swedish": "sv",
            "Tagalog": "tl",
            "Tajik": "tg",
            "Tamil": "ta",
            "Tatar": "tt",
            "Telugu": "te",
            "Thai": "th",
            "Tibetan": "bo",
            "Turkish": "tr",
            "Turkmen": "tk",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Welsh": "cy",
            "Yiddish": "yi",
            "Yoruba": "yo"
        }

        # menu actions for each language
        for display_name, code in self.languageMapping.items():
            if display_name:  # Skip empty entry
                action = self.languageMenu.addAction(display_name)
                action.triggered.connect(lambda checked, name=display_name, code=code: self.selectLanguage(name, code))
            else:
                # separator and "None" - empty selection
                self.languageMenu.addSeparator()
                clear_action = self.languageMenu.addAction("Clear selection")
                clear_action.triggered.connect(lambda: self.selectLanguage("Select language...", ""))

        self.languageDropdown.clicked.connect(self.showLanguageMenu)

        self.selectLanguageCode = ""
        self.selectLanguageDisplay = ""


    def showLanguageMenu(self):
        # position menu below button
        button_pos = self.languageDropdown.mapToGlobal(self.languageDropdown.rect().bottomLeft())
        self.languageMenu.exec(button_pos)

    def selectLanguage(self, display_name, code):
        self.languageDropdown.setText(display_name)
        self.selectLanguageCode = code
        self.selectLanguageDisplay = display_name if display_name != "Select language..." else ""

        print(f"Selected: {display_name} ({code})")

    def createFileSelectionSection(self, parentLayout):
        parentLayout.addWidget(QLabel("Select a file"))
        fileSectionLayout = QHBoxLayout()
        self.filePathEdit = QLineEdit()
        self.filePathEdit.setPlaceholderText("No file selected...")
        self.filePathEdit.setReadOnly(True)
        self.filePathEdit.textChanged.connect(self.onFilePathChanged)
        fileSectionLayout.addWidget(self.filePathEdit)

        self.fileBrowseButton = QPushButton("Browse")
        self.fileBrowseButton.setFixedWidth(100)
        self.fileBrowseButton.clicked.connect(self.selectFile)
        fileSectionLayout.addWidget(self.fileBrowseButton)

        parentLayout.addLayout(fileSectionLayout)


    def selectFile(self):
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select a file",                    # Dialog title
            "",                                 # Starting directory (empty = current dir)
            "All Files (*);;Text Files (*.txt);;Python Files (*.py);;Image Files (*.png *.jpg *.jpeg *.gif *.bmp)"  # File filters
        )
        if file_path:
            self.filePathEdit.setText(file_path)
            print(f"Selected file: {file_path}")
            print(f"Filter used: {selected_filter}")
        else:
            print("No file selected")

    def onFilePathChanged(self, text):
        self.runTranscriptionButton.setEnabled(bool(text.strip()))

    def runTranscription(self):
        """Start transcription process using threading."""
        if self.transcriptionManager.is_running():
            return

        # Collect UI configuration
        ui_config = self.getUIConfig()
        audio_file = self.filePathEdit.text().strip()

        if not audio_file:
            self.onTranscriptionError("No audio file selected")
            return

        # Reset UI state
        self.transcriptionActive = True
        self.resultsPageButton.setEnabled(True)
        self.progressResultsBar.setValue(0)
        self.statusLabel.setText("Starting transcription...")

        # Switch to results page
        self.switchToPage(1)

        # Start transcription
        self.transcriptionManager.start_transcription(audio_file, ui_config)


    # GETTERS

    def getTimestampAlignmentEnabled(self):
        return self.alignTimestampsCheckbox.isChecked()

    def getDiarizationEnabled(self):
        return self.diarizationChecbox.isChecked()

    def getUIConfig(self):
        """Collect configuration from UI elements."""
        return {
            'enable_alignment': self.getTimestampAlignmentEnabled(),
            'enable_diarization': self.getDiarizationEnabled(),
            'language': self.selectLanguageCode if hasattr(self, 'selectLanguageCode') else None,
            'show_timestamps': getattr(self, 'showTimestampsCheckbox',
                                       None) and self.showTimestampsCheckbox.isChecked(),
            'show_speakers': getattr(self, 'showSpeakersCheckbox', None) and self.showSpeakersCheckbox.isChecked(),
        }

    def onProgressUpdate(self, progress):
        """Handle progress updates from transcription manager."""
        self.progressResultsBar.setValue(progress)

    def onStatusUpdate(self, status):
        """Handle status updates from transcription manager."""
        self.statusLabel.setText(status)

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
        self.statusLabel.setText("Transcription completed successfully")

    def onTranscriptionError(self, error_message):
        """Handle transcription errors."""
        self.transcriptionActive = False
        self.progressResultsBar.setValue(0)
        self.statusLabel.setText(f"Error: {error_message}")

        # Show error in transcription area
        self.transcriptionTextArea.setPlainText(f"Transcription failed: {error_message}")

    def onModelsLoaded(self):
        """Handle model loading completion."""
        self.statusLabel.setText("Models loaded, starting transcription...")

app = QApplication(sys.argv)

# Set a larger default font for the entire application
font = QFont()
font.setPointSize(12)  # Adjust size as needed (default is usually 8-9)
app.setFont(font)

window = MainWindow()
window.show()
app.exec()