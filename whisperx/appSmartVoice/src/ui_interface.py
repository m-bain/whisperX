# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'new_interface.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QProgressBar,
    QPushButton, QSizePolicy, QSpacerItem, QTextEdit,
    QVBoxLayout, QWidget)

from Custom_Widgets.QCustomQStackedWidget import QCustomQStackedWidget
from Custom_Widgets.QCustomSlideMenu import QCustomSlideMenu
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1043, 662)
        font = QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(1043, 635))
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.leftMenu = QCustomSlideMenu(self.centralwidget)
        self.leftMenu.setObjectName(u"leftMenu")
        self.verticalLayout = QVBoxLayout(self.leftMenu)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.widget = QWidget(self.leftMenu)
        self.widget.setObjectName(u"widget")
        self.widget.setMinimumSize(QSize(46, 42))
        self.verticalLayout_2 = QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 5, 0, -1)
        self.menuBtn = QPushButton(self.widget)
        self.menuBtn.setObjectName(u"menuBtn")
        icon = QIcon()
        icon.addFile(u":/feather/icons/feather/menu.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.menuBtn.setIcon(icon)
        self.menuBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_2.addWidget(self.menuBtn, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout.addWidget(self.widget, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.widget_2 = QWidget(self.leftMenu)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setMinimumSize(QSize(93, 76))
        self.verticalLayout_3 = QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 5, 0, 5)
        self.setupBtn = QPushButton(self.widget_2)
        self.setupBtn.setObjectName(u"setupBtn")
        icon1 = QIcon()
        icon1.addFile(u":/material_design/icons/material_design/display_settings.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.setupBtn.setIcon(icon1)
        self.setupBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_3.addWidget(self.setupBtn)

        self.resultsBtn = QPushButton(self.widget_2)
        self.resultsBtn.setObjectName(u"resultsBtn")
        self.resultsBtn.setEnabled(False)
        icon2 = QIcon()
        icon2.addFile(u":/feather/icons/feather/file-text.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.resultsBtn.setIcon(icon2)
        self.resultsBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_3.addWidget(self.resultsBtn)


        self.verticalLayout.addWidget(self.widget_2, 0, Qt.AlignmentFlag.AlignTop)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.widget_3 = QWidget(self.leftMenu)
        self.widget_3.setObjectName(u"widget_3")
        self.widget_3.setMinimumSize(QSize(114, 108))
        self.verticalLayout_4 = QVBoxLayout(self.widget_3)
        self.verticalLayout_4.setSpacing(5)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 5, 0, 5)
        self.settingsBtn = QPushButton(self.widget_3)
        self.settingsBtn.setObjectName(u"settingsBtn")
        icon3 = QIcon()
        icon3.addFile(u":/feather/icons/feather/settings.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.settingsBtn.setIcon(icon3)
        self.settingsBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_4.addWidget(self.settingsBtn)

        self.infoBtn = QPushButton(self.widget_3)
        self.infoBtn.setObjectName(u"infoBtn")
        icon4 = QIcon()
        icon4.addFile(u":/feather/icons/feather/info.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.infoBtn.setIcon(icon4)
        self.infoBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_4.addWidget(self.infoBtn)

        self.helpBtn = QPushButton(self.widget_3)
        self.helpBtn.setObjectName(u"helpBtn")
        icon5 = QIcon()
        icon5.addFile(u":/feather/icons/feather/help-circle.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.helpBtn.setIcon(icon5)
        self.helpBtn.setIconSize(QSize(20, 20))

        self.verticalLayout_4.addWidget(self.helpBtn)


        self.verticalLayout.addWidget(self.widget_3, 0, Qt.AlignmentFlag.AlignBottom)


        self.horizontalLayout.addWidget(self.leftMenu)

        self.mainBody = QWidget(self.centralwidget)
        self.mainBody.setObjectName(u"mainBody")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainBody.sizePolicy().hasHeightForWidth())
        self.mainBody.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.mainBody)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.header = QWidget(self.mainBody)
        self.header.setObjectName(u"header")
        self.horizontalLayout_5 = QHBoxLayout(self.header)
        self.horizontalLayout_5.setSpacing(5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(5, 0, 5, 0)
        self.titleText = QLabel(self.header)
        self.titleText.setObjectName(u"titleText")
        font1 = QFont()
        font1.setPointSize(12)
        font1.setBold(True)
        self.titleText.setFont(font1)

        self.horizontalLayout_5.addWidget(self.titleText)

        self.headerMenuFrame = QFrame(self.header)
        self.headerMenuFrame.setObjectName(u"headerMenuFrame")
        self.headerMenuFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.headerMenuFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.headerMenuFrame)
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.historyBtn = QPushButton(self.headerMenuFrame)
        self.historyBtn.setObjectName(u"historyBtn")
        icon6 = QIcon()
        icon6.addFile(u":/material_design/icons/material_design/view_list.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.historyBtn.setIcon(icon6)
        self.historyBtn.setIconSize(QSize(24, 24))

        self.horizontalLayout_4.addWidget(self.historyBtn)

        self.profileBtn = QPushButton(self.headerMenuFrame)
        self.profileBtn.setObjectName(u"profileBtn")
        icon7 = QIcon()
        icon7.addFile(u":/feather/icons/feather/user.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.profileBtn.setIcon(icon7)
        self.profileBtn.setIconSize(QSize(24, 24))

        self.horizontalLayout_4.addWidget(self.profileBtn)


        self.horizontalLayout_5.addWidget(self.headerMenuFrame)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.headerSearchFrame = QFrame(self.header)
        self.headerSearchFrame.setObjectName(u"headerSearchFrame")
        self.headerSearchFrame.setMaximumSize(QSize(400, 16777215))
        self.headerSearchFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.headerSearchFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.headerSearchFrame)
        self.horizontalLayout_6.setSpacing(10)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(5, 5, 5, 5)
        self.label_5 = QLabel(self.headerSearchFrame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(20, 20))
        self.label_5.setMaximumSize(QSize(16, 16))
        self.label_5.setPixmap(QPixmap(u":/feather/icons/feather/search.png"))
        self.label_5.setScaledContents(True)

        self.horizontalLayout_6.addWidget(self.label_5)

        self.searchInput = QLineEdit(self.headerSearchFrame)
        self.searchInput.setObjectName(u"searchInput")
        self.searchInput.setMinimumSize(QSize(200, 0))
        self.searchInput.setToolTipDuration(-1)

        self.horizontalLayout_6.addWidget(self.searchInput)

        self.searchBtn = QPushButton(self.headerSearchFrame)
        self.searchBtn.setObjectName(u"searchBtn")

        self.horizontalLayout_6.addWidget(self.searchBtn)


        self.horizontalLayout_5.addWidget(self.headerSearchFrame)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_2)

        self.windowControlsFrame = QFrame(self.header)
        self.windowControlsFrame.setObjectName(u"windowControlsFrame")
        self.windowControlsFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.windowControlsFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.windowControlsFrame)
        self.horizontalLayout_7.setSpacing(2)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.minimizeBtn = QPushButton(self.windowControlsFrame)
        self.minimizeBtn.setObjectName(u"minimizeBtn")
        icon8 = QIcon()
        icon8.addFile(u":/feather/icons/feather/minus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.minimizeBtn.setIcon(icon8)

        self.horizontalLayout_7.addWidget(self.minimizeBtn)

        self.restoreBtn = QPushButton(self.windowControlsFrame)
        self.restoreBtn.setObjectName(u"restoreBtn")
        icon9 = QIcon()
        icon9.addFile(u":/feather/icons/feather/square.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.restoreBtn.setIcon(icon9)

        self.horizontalLayout_7.addWidget(self.restoreBtn)

        self.closeBtn = QPushButton(self.windowControlsFrame)
        self.closeBtn.setObjectName(u"closeBtn")
        icon10 = QIcon()
        icon10.addFile(u":/feather/icons/feather/window_close.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.closeBtn.setIcon(icon10)

        self.horizontalLayout_7.addWidget(self.closeBtn)


        self.horizontalLayout_5.addWidget(self.windowControlsFrame, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_5.addWidget(self.header, 0, Qt.AlignmentFlag.AlignTop)

        self.mainContent = QWidget(self.mainBody)
        self.mainContent.setObjectName(u"mainContent")
        self.horizontalLayout_8 = QHBoxLayout(self.mainContent)
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(5, 5, 5, 5)
        self.mainPage = QWidget(self.mainContent)
        self.mainPage.setObjectName(u"mainPage")
        self.verticalLayout_6 = QVBoxLayout(self.mainPage)
        self.verticalLayout_6.setSpacing(5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(5, 5, 5, 5)
        self.stackedMainPages = QCustomQStackedWidget(self.mainPage)
        self.stackedMainPages.setObjectName(u"stackedMainPages")
        self.setupPage = QWidget()
        self.setupPage.setObjectName(u"setupPage")
        self.verticalLayout_10 = QVBoxLayout(self.setupPage)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.groupBox_6 = QGroupBox(self.setupPage)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.verticalLayout_18 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.label_8 = QLabel(self.groupBox_6)
        self.label_8.setObjectName(u"label_8")

        self.verticalLayout_18.addWidget(self.label_8)

        self.frame_2 = QFrame(self.groupBox_6)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.lineEdit = QLineEdit(self.frame_2)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setMaximumSize(QSize(700, 16777215))
        self.lineEdit.setReadOnly(True)

        self.horizontalLayout_13.addWidget(self.lineEdit)

        self.fileBrowseBtn = QPushButton(self.frame_2)
        self.fileBrowseBtn.setObjectName(u"fileBrowseBtn")

        self.horizontalLayout_13.addWidget(self.fileBrowseBtn, 0, Qt.AlignmentFlag.AlignRight)


        self.verticalLayout_18.addWidget(self.frame_2)


        self.verticalLayout_10.addWidget(self.groupBox_6)

        self.groupBox = QGroupBox(self.setupPage)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_11 = QVBoxLayout(self.groupBox)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.timestampsEnabled = QCheckBox(self.groupBox)
        self.timestampsEnabled.setObjectName(u"timestampsEnabled")

        self.verticalLayout_11.addWidget(self.timestampsEnabled)

        self.diarizationEnabled = QCheckBox(self.groupBox)
        self.diarizationEnabled.setObjectName(u"diarizationEnabled")

        self.verticalLayout_11.addWidget(self.diarizationEnabled)

        self.frame_3 = QFrame(self.groupBox)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_9 = QLabel(self.frame_3)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_14.addWidget(self.label_9)

        self.languageDropdownBtn = QPushButton(self.frame_3)
        self.languageDropdownBtn.setObjectName(u"languageDropdownBtn")
        icon11 = QIcon()
        icon11.addFile(u":/feather/icons/feather/arrow_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.languageDropdownBtn.setIcon(icon11)

        self.horizontalLayout_14.addWidget(self.languageDropdownBtn, 0, Qt.AlignmentFlag.AlignRight)


        self.verticalLayout_11.addWidget(self.frame_3)


        self.verticalLayout_10.addWidget(self.groupBox)

        self.runTranscriptionBtn = QPushButton(self.setupPage)
        self.runTranscriptionBtn.setObjectName(u"runTranscriptionBtn")

        self.verticalLayout_10.addWidget(self.runTranscriptionBtn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_10.addItem(self.verticalSpacer_2)

        self.stackedMainPages.addWidget(self.setupPage)
        self.settingsPage = QWidget()
        self.settingsPage.setObjectName(u"settingsPage")
        self.verticalLayout_16 = QVBoxLayout(self.settingsPage)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.groupBox_5 = QGroupBox(self.settingsPage)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_15 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.widget_5 = QWidget(self.groupBox_5)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_11 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_11 = QLabel(self.widget_5)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_11.addWidget(self.label_11, 0, Qt.AlignmentFlag.AlignTop)

        self.comboBox_2 = QComboBox(self.widget_5)
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.horizontalLayout_11.addWidget(self.comboBox_2, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_15.addWidget(self.widget_5)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_15.addItem(self.verticalSpacer_3)


        self.verticalLayout_16.addWidget(self.groupBox_5)

        self.stackedMainPages.addWidget(self.settingsPage)
        self.informationPage = QWidget()
        self.informationPage.setObjectName(u"informationPage")
        self.verticalLayout_17 = QVBoxLayout(self.informationPage)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.label_10 = QLabel(self.informationPage)
        self.label_10.setObjectName(u"label_10")

        self.verticalLayout_17.addWidget(self.label_10)

        self.stackedMainPages.addWidget(self.informationPage)
        self.helpPage = QWidget()
        self.helpPage.setObjectName(u"helpPage")
        self.horizontalLayout_12 = QHBoxLayout(self.helpPage)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_3 = QLabel(self.helpPage)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_12.addWidget(self.label_3)

        self.stackedMainPages.addWidget(self.helpPage)
        self.resultsPage = QWidget()
        self.resultsPage.setObjectName(u"resultsPage")
        self.verticalLayout_12 = QVBoxLayout(self.resultsPage)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.groupBox_2 = QGroupBox(self.resultsPage)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_13 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.progressBar_2 = QProgressBar(self.groupBox_2)
        self.progressBar_2.setObjectName(u"progressBar_2")
        self.progressBar_2.setValue(0)

        self.verticalLayout_13.addWidget(self.progressBar_2)

        self.statusLabel = QLabel(self.groupBox_2)
        self.statusLabel.setObjectName(u"statusLabel")
        self.statusLabel.setWordWrap(True)

        self.verticalLayout_13.addWidget(self.statusLabel)


        self.verticalLayout_12.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.resultsPage)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.showTimestamps = QCheckBox(self.groupBox_3)
        self.showTimestamps.setObjectName(u"showTimestamps")

        self.horizontalLayout_10.addWidget(self.showTimestamps)

        self.showSpeakers = QCheckBox(self.groupBox_3)
        self.showSpeakers.setObjectName(u"showSpeakers")

        self.horizontalLayout_10.addWidget(self.showSpeakers)


        self.verticalLayout_12.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.resultsPage)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_14 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.transcriptionTextArea = QTextEdit(self.groupBox_4)
        self.transcriptionTextArea.setObjectName(u"transcriptionTextArea")
        self.transcriptionTextArea.setMinimumSize(QSize(0, 250))
        self.transcriptionTextArea.setReadOnly(True)

        self.verticalLayout_14.addWidget(self.transcriptionTextArea)


        self.verticalLayout_12.addWidget(self.groupBox_4)

        self.downloadSRTBtn = QPushButton(self.resultsPage)
        self.downloadSRTBtn.setObjectName(u"downloadSRTBtn")

        self.verticalLayout_12.addWidget(self.downloadSRTBtn, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignBottom)

        self.stackedMainPages.addWidget(self.resultsPage)

        self.verticalLayout_6.addWidget(self.stackedMainPages)


        self.horizontalLayout_8.addWidget(self.mainPage)

        self.rightMenu = QCustomSlideMenu(self.mainContent)
        self.rightMenu.setObjectName(u"rightMenu")
        self.rightMenu.setMinimumSize(QSize(450, 0))
        self.verticalLayout_7 = QVBoxLayout(self.rightMenu)
        self.verticalLayout_7.setSpacing(5)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(5, 5, 5, 5)
        self.widget_4 = QWidget(self.rightMenu)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_9 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(5, 5, 0, 5)
        self.label_4 = QLabel(self.widget_4)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_9.addWidget(self.label_4, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.rightMenuBtn = QPushButton(self.widget_4)
        self.rightMenuBtn.setObjectName(u"rightMenuBtn")
        icon12 = QIcon()
        icon12.addFile(u":/feather/icons/feather/chevrons-right.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.rightMenuBtn.setIcon(icon12)
        self.rightMenuBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_9.addWidget(self.rightMenuBtn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_7.addWidget(self.widget_4, 0, Qt.AlignmentFlag.AlignTop)

        self.rightMenuStacked = QCustomQStackedWidget(self.rightMenu)
        self.rightMenuStacked.setObjectName(u"rightMenuStacked")
        self.historyPage = QWidget()
        self.historyPage.setObjectName(u"historyPage")
        self.verticalLayout_8 = QVBoxLayout(self.historyPage)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_6 = QLabel(self.historyPage)
        self.label_6.setObjectName(u"label_6")

        self.verticalLayout_8.addWidget(self.label_6)

        self.transcrHistoryList = QListWidget(self.historyPage)
        self.transcrHistoryList.setObjectName(u"transcrHistoryList")
        self.transcrHistoryList.setStyleSheet(u"#historyItemBox {\n"
"                    background-color:red !important;\n"
"                }\n"
"HistoryItemWidget {\n"
"                    background-color: red !important;\n"
"                    border: 1px solid #3e3e42;\n"
"                    border-left: 4px solid #007acc;\n"
"                    border-radius: 8px;\n"
"                    padding: 12px;\n"
"                    margin: 2px;\n"
" }")

        self.verticalLayout_8.addWidget(self.transcrHistoryList)

        self.rightMenuStacked.addWidget(self.historyPage)
        self.profilePage = QWidget()
        self.profilePage.setObjectName(u"profilePage")
        self.verticalLayout_9 = QVBoxLayout(self.profilePage)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_7 = QLabel(self.profilePage)
        self.label_7.setObjectName(u"label_7")

        self.verticalLayout_9.addWidget(self.label_7)

        self.rightMenuStacked.addWidget(self.profilePage)

        self.verticalLayout_7.addWidget(self.rightMenuStacked)


        self.horizontalLayout_8.addWidget(self.rightMenu)


        self.verticalLayout_5.addWidget(self.mainContent)

        self.footer = QWidget(self.mainBody)
        self.footer.setObjectName(u"footer")
        self.horizontalLayout_2 = QHBoxLayout(self.footer)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.label = QLabel(self.footer)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.frame = QFrame(self.footer)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2, 0, Qt.AlignmentFlag.AlignLeft)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(24)

        self.horizontalLayout_3.addWidget(self.progressBar)


        self.horizontalLayout_2.addWidget(self.frame, 0, Qt.AlignmentFlag.AlignHCenter)

        self.sizeGrip = QFrame(self.footer)
        self.sizeGrip.setObjectName(u"sizeGrip")
        self.sizeGrip.setMinimumSize(QSize(15, 15))
        self.sizeGrip.setMaximumSize(QSize(15, 15))
        self.sizeGrip.setFrameShape(QFrame.Shape.StyledPanel)
        self.sizeGrip.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_2.addWidget(self.sizeGrip, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignBottom)


        self.verticalLayout_5.addWidget(self.footer, 0, Qt.AlignmentFlag.AlignBottom)


        self.horizontalLayout.addWidget(self.mainBody)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.menuBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Maximize main menu", None))
#endif // QT_CONFIG(tooltip)
        self.menuBtn.setText("")
#if QT_CONFIG(tooltip)
        self.setupBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Input file specification", None))
#endif // QT_CONFIG(tooltip)
        self.setupBtn.setText(QCoreApplication.translate("MainWindow", u"   Setup", None))
#if QT_CONFIG(tooltip)
        self.resultsBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Results of the transcription", None))
#endif // QT_CONFIG(tooltip)
        self.resultsBtn.setText(QCoreApplication.translate("MainWindow", u"   Results", None))
#if QT_CONFIG(tooltip)
        self.settingsBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Application settings", None))
#endif // QT_CONFIG(tooltip)
        self.settingsBtn.setText(QCoreApplication.translate("MainWindow", u"   Settings", None))
#if QT_CONFIG(tooltip)
        self.infoBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Information about application", None))
#endif // QT_CONFIG(tooltip)
        self.infoBtn.setText(QCoreApplication.translate("MainWindow", u"   Information", None))
#if QT_CONFIG(tooltip)
        self.helpBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Get help", None))
#endif // QT_CONFIG(tooltip)
        self.helpBtn.setText(QCoreApplication.translate("MainWindow", u"   Help", None))
        self.titleText.setText(QCoreApplication.translate("MainWindow", u"SmartVoice", None))
#if QT_CONFIG(tooltip)
        self.historyBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Open history of transcriptions", None))
#endif // QT_CONFIG(tooltip)
        self.historyBtn.setText("")
#if QT_CONFIG(tooltip)
        self.profileBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Log in", None))
#endif // QT_CONFIG(tooltip)
        self.profileBtn.setText("")
        self.label_5.setText("")
        self.searchInput.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search...", None))
#if QT_CONFIG(tooltip)
        self.searchBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Search", None))
#endif // QT_CONFIG(tooltip)
        self.searchBtn.setText(QCoreApplication.translate("MainWindow", u"Ctrl+F", None))
        self.minimizeBtn.setText("")
        self.restoreBtn.setText("")
        self.closeBtn.setText("")
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Select a file", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Select a file", None))
        self.lineEdit.setText(QCoreApplication.translate("MainWindow", u"No file selected...", None))
#if QT_CONFIG(tooltip)
        self.fileBrowseBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Select input file", None))
#endif // QT_CONFIG(tooltip)
        self.fileBrowseBtn.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
#if QT_CONFIG(tooltip)
        self.timestampsEnabled.setToolTip(QCoreApplication.translate("MainWindow", u"Enable additional timestamp alignment for better precision", None))
#endif // QT_CONFIG(tooltip)
        self.timestampsEnabled.setText(QCoreApplication.translate("MainWindow", u"Enable timestamp alignment", None))
#if QT_CONFIG(tooltip)
        self.diarizationEnabled.setToolTip(QCoreApplication.translate("MainWindow", u"Enable identification of speakers and assignment of IDs", None))
#endif // QT_CONFIG(tooltip)
        self.diarizationEnabled.setText(QCoreApplication.translate("MainWindow", u"Enable speaker diarization", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Language (Optional):", None))
        self.languageDropdownBtn.setText(QCoreApplication.translate("MainWindow", u" Select a language", None))
        self.runTranscriptionBtn.setText(QCoreApplication.translate("MainWindow", u"Run Transcription", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Style", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Theme", None))
        self.comboBox_2.setCurrentText("")
        self.comboBox_2.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Default", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Information Page", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Help page", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Progress", None))
        self.statusLabel.setText(QCoreApplication.translate("MainWindow", u"Starting transcription...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Display options", None))
#if QT_CONFIG(tooltip)
        self.showTimestamps.setToolTip(QCoreApplication.translate("MainWindow", u"Display timestamps", None))
#endif // QT_CONFIG(tooltip)
        self.showTimestamps.setText(QCoreApplication.translate("MainWindow", u"Timestamps", None))
#if QT_CONFIG(tooltip)
        self.showSpeakers.setToolTip(QCoreApplication.translate("MainWindow", u"Display speakers' IDs", None))
#endif // QT_CONFIG(tooltip)
        self.showSpeakers.setText(QCoreApplication.translate("MainWindow", u"Speakers", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Transcription result", None))
        self.transcriptionTextArea.setPlaceholderText(QCoreApplication.translate("MainWindow", u"No results yet...", None))
        self.downloadSRTBtn.setText(QCoreApplication.translate("MainWindow", u"Download SRT", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"History", None))
#if QT_CONFIG(tooltip)
        self.rightMenuBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Close menu", None))
#endif // QT_CONFIG(tooltip)
        self.rightMenuBtn.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Past transcriptions", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Login Page", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"SmartVoice", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Progress", None))
#if QT_CONFIG(tooltip)
        self.sizeGrip.setToolTip(QCoreApplication.translate("MainWindow", u"Drag to rescale window", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

