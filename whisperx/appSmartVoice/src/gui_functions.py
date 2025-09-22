from Custom_Widgets import *
from Custom_Widgets.QAppSettings import QAppSettings
from Custom_Widgets.QCustomTipOverlay import QCustomTipOverlay
from Custom_Widgets.QCustomLoadingIndicators import QCustom3CirclesLoader

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QMenu, QFileDialog


class GUIFunctions():
    def __init__(self, MainWindow):
        self.main = MainWindow
        self.ui = MainWindow.ui

        self.transcriptionActive = False



        # INIT
        self.ui.searchBtn.clicked.connect(self.showSearchResults)
        self.ui.fileBrowseBtn.clicked.connect(self.selectFile)
        self.ui.runTranscriptionBtn.clicked.connect(self.runTranscription)
        self.ui.downloadSRTBtn.clicked.connect(self.main.downloadSRTFile)

        # Results display mode
        self.ui.showTimestamps.stateChanged.connect(self.main.updateTranscriptionDisplay)
        self.ui.showSpeakers.stateChanged.connect(self.main.updateTranscriptionDisplay)

        self.connectMenuButtons()
        self.addLanguageSelector()


    def runTranscription(self):
        """Start transcription process using threading."""
        if self.main.transcriptionManager.is_running():
            return

        # Collect UI configuration
        ui_config = self.getUIConfig()
        audio_file = self.ui.lineEdit.text().strip()

        if not audio_file:
            self.main.onTranscriptionError("No audio file selected")
            return

        # Reset UI state
        self.transcriptionActive = True
        self.ui.resultsBtn.setEnabled(True)
        self.ui.progressBar_2.setValue(0)
        self.ui.statusLabel.setText("Starting transcription...")
        self.ui.transcriptionTextArea.setPlainText("")

        # Switch to results page
        self.ui.stackedMainPages.setCurrentIndex(1)

        # Start transcription
        self.main.transcriptionManager.start_transcription(audio_file, ui_config)
        # self.transcriptionManager.start_transcription_direct(audio_file, ui_config)

    def getUIConfig(self):
        """Collect configuration from UI elements."""
        return {
            'enable_alignment': self.ui.timestampsEnabled.isChecked(),
            'enable_diarization': self.ui.diarizationEnabled.isChecked(),
            'language': self.selectLanguageCode if hasattr(self, 'selectLanguageCode') else None,
            'show_timestamps': getattr(self, 'showTimestampsCheckbox',
                                       None) and self.ui.showTimestamps.isChecked(),
            'show_speakers': getattr(self, 'showSpeakersCheckbox', None) and self.ui.showSpeakers.isChecked(),
        }


    def selectFile(self):
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self.main,
            "Select a file",                    # Dialog title
            "",                                 # Starting directory (empty = current dir)
            "All Files (*);;Text Files (*.txt);;Python Files (*.py);;Image Files (*.png *.jpg *.jpeg *.gif *.bmp)"  # File filters
        )
        if file_path:
            self.ui.lineEdit.setText(file_path)
            print(f"Selected file: {file_path}")
            print(f"Filter used: {selected_filter}")
        else:
            print("No file selected")

    def addLanguageSelector(self):
        self.ui.languageMenu = QMenu(self.ui.languageDropdownBtn)
        self.ui.languageMenu.setFixedHeight(700)
        self.ui.languageMenu.setStyleSheet("QMenu { menu-scrollable: 1; }")

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
                action = self.ui.languageMenu.addAction(display_name)
                action.triggered.connect(lambda checked, name=display_name, code=code: self.selectLanguage(name, code))
            else:
                # separator and "None" - empty selection
                self.ui.languageMenu.addSeparator()
                clear_action = self.ui.languageMenu.addAction("Clear selection")
                clear_action.triggered.connect(lambda: self.selectLanguage(" Select language", ""))

        self.ui.languageDropdownBtn.clicked.connect(self.showLanguageMenu)

        self.selectLanguageCode = ""
        self.selectLanguageDisplay = ""

    def showLanguageMenu(self):
        # position menu below button
        button_pos = self.ui.languageDropdownBtn.mapToGlobal(self.ui.languageDropdownBtn.rect().bottomLeft())
        # print("BTTN POSITION IS: ", button_pos)
        # button_pos.setY(button_pos.x() + 100)
        print("BTTN POSITION AFTRER IS: ", button_pos)
        self.ui.languageMenu.exec(button_pos)

    def selectLanguage(self, display_name, code):
        self.ui.languageDropdownBtn.setText(display_name)
        self.selectLanguageCode = code
        self.selectLanguageDisplay = display_name if display_name != "Select language..." else ""

        print(f"Selected: {display_name} ({code})")


    def connectMenuButtons(self):
        # TODO: change page of the stacked widget
        self.ui.historyBtn.clicked.connect(lambda: self.ui.rightMenu.expandMenu())
        self.ui.profileBtn.clicked.connect(lambda: self.ui.rightMenu.expandMenu())

        self.ui.rightMenuBtn.clicked.connect(lambda: self.ui.rightMenu.collapseMenu())


    def createSearchTipOverlay(self):
        self.searchTooltip = QCustomTipOverlay(
            title="Search results",
            description="Searching...",
            icon=self.main.theme.PATH_RESOURCES + "/feather/icons/feather/search.png",
            image="image.png",
            isClosable=True,
            target=self.ui.searchInput,
            parent=self.main,
            aniType="pull-up",
            # isDeleteOnClose=True,
            duration=-1,
            tailPosition="top-center",
            closeIcon=self.main.theme.PATH_RESOURCES + "/material_design/close.png"  # Add a close icon
        )
        self.loader = loader = QCustom3CirclesLoader(
            parent=self.searchTooltip,
            color=QColor(self.main.theme.COLOR_ACCENT_2),
            penWidth=20,
            animationDuration=400
        )

        # Loader animation
        self.searchTooltip.addWidget(self.loader)

    def showSearchResults(self):
        searchPhrase = self.ui.searchInput.text()
        if not searchPhrase:
            return
        try:
            self.searchTooltip.show()
        except:
            self.createSearchTipOverlay()
            self.searchTooltip.show()

        self.searchTooltip.setDescription(f"Search results for \"{searchPhrase}\"")