# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfacepqAvEB.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)
import _icons_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 471)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.leftMenu = QWidget(self.centralwidget)
        self.leftMenu.setObjectName(u"leftMenu")
        self.verticalLayout = QVBoxLayout(self.leftMenu)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(self.leftMenu)
        self.widget.setObjectName(u"widget")
        self.menuBtn = QPushButton(self.widget)
        self.menuBtn.setObjectName(u"menuBtn")
        self.menuBtn.setGeometry(QRect(10, 10, 75, 24))
        icon = QIcon()
        icon.addFile(u":/feather/icons/feather/menu.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.menuBtn.setIcon(icon)

        self.verticalLayout.addWidget(self.widget)

        self.widget_2 = QWidget(self.leftMenu)
        self.widget_2.setObjectName(u"widget_2")
        self.setup = QPushButton(self.widget_2)
        self.setup.setObjectName(u"setup")
        self.setup.setGeometry(QRect(10, 10, 75, 24))
        icon1 = QIcon()
        icon1.addFile(u":/material_design/icons/material_design/display_settings.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.setup.setIcon(icon1)
        self.results = QPushButton(self.widget_2)
        self.results.setObjectName(u"results")
        self.results.setGeometry(QRect(10, 40, 75, 24))
        icon2 = QIcon()
        icon2.addFile(u":/feather/icons/feather/file-text.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.results.setIcon(icon2)

        self.verticalLayout.addWidget(self.widget_2)

        self.widget_3 = QWidget(self.leftMenu)
        self.widget_3.setObjectName(u"widget_3")

        self.verticalLayout.addWidget(self.widget_3)


        self.horizontalLayout.addWidget(self.leftMenu)

        self.centerMenu = QWidget(self.centralwidget)
        self.centerMenu.setObjectName(u"centerMenu")

        self.horizontalLayout.addWidget(self.centerMenu)

        self.mainBody = QWidget(self.centralwidget)
        self.mainBody.setObjectName(u"mainBody")

        self.horizontalLayout.addWidget(self.mainBody)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.menuBtn.setText("")
        self.setup.setText(QCoreApplication.translate("MainWindow", u"Setup", None))
        self.results.setText(QCoreApplication.translate("MainWindow", u"Results", None))
    # retranslateUi

