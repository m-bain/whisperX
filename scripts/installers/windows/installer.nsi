; WhisperX SmartVoice Installer Script for NSIS
; Creates a Windows installer that:
; - Installs launcher and main application
; - Creates shortcuts
; - Preserves user data on reinstall
; - Supports uninstallation

;--------------------------------
; Include Modern UI

!include "MUI2.nsh"
!include "FileFunc.nsh"

;--------------------------------
; General

; Name and file
!define PRODUCT_NAME "WhisperX SmartVoice"
!define PRODUCT_VERSION "3.4.2"
!define PRODUCT_PUBLISHER "WhisperX"
!define PRODUCT_WEB_SITE "https://github.com/xlazarik/whisperX"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "SmartVoice-Setup-${PRODUCT_VERSION}.exe"

; Default installation folder
InstallDir "$PROGRAMFILES64\WhisperX\SmartVoice"

; Get installation folder from registry if available
InstallDirRegKey HKLM "Software\WhisperX\SmartVoice" "InstallDir"

; Request application privileges
RequestExecutionLevel admin

;--------------------------------
; Interface Settings

!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

;--------------------------------
; Pages

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\SmartVoiceLauncher.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch SmartVoice Launcher"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
; Languages

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Sections

Section "SmartVoice Application" SecMain

  SetOutPath "$INSTDIR"

  ; Check if application is running
  Call CheckAndCloseRunningApp

  ; Install launcher
  SetOutPath "$INSTDIR\Launcher"
  File /r "..\..\..\dist\SmartVoiceLauncher\*.*"

  ; Install main application
  SetOutPath "$INSTDIR\SmartVoice"
  File /r "..\..\..\dist\SmartVoice\*.*"

  ; Create launcher shortcut in install dir
  CreateShortCut "$INSTDIR\SmartVoiceLauncher.exe.lnk" "$INSTDIR\Launcher\SmartVoiceLauncher.exe"

  ; Store installation folder
  WriteRegStr HKLM "Software\WhisperX\SmartVoice" "InstallDir" $INSTDIR

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Write uninstall information
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "DisplayIcon" "$INSTDIR\Launcher\SmartVoiceLauncher.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "NoRepair" 1

  ; Estimate size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice" "EstimatedSize" "$0"

SectionEnd

Section "Start Menu Shortcuts" SecShortcuts

  CreateDirectory "$SMPROGRAMS\WhisperX SmartVoice"
  CreateShortCut "$SMPROGRAMS\WhisperX SmartVoice\SmartVoice Launcher.lnk" "$INSTDIR\Launcher\SmartVoiceLauncher.exe"
  CreateShortCut "$SMPROGRAMS\WhisperX SmartVoice\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

SectionEnd

Section "Desktop Shortcut" SecDesktop

  CreateShortCut "$DESKTOP\SmartVoice.lnk" "$INSTDIR\Launcher\SmartVoiceLauncher.exe"

SectionEnd

;--------------------------------
; Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "The main SmartVoice application files."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecShortcuts} "Add shortcuts to the Start Menu."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Add a shortcut to the Desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Functions

Function CheckAndCloseRunningApp
  ; Check if SmartVoice or Launcher is running
  nsExec::ExecToStack 'tasklist /FI "IMAGENAME eq SmartVoiceLauncher.exe" /NH'
  Pop $0
  Pop $1
  ${If} $0 == 0
    MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "SmartVoice Launcher is currently running. It will be closed to continue installation." IDOK close
    Abort
    close:
    nsExec::ExecToStack 'taskkill /F /IM SmartVoiceLauncher.exe'
    nsExec::ExecToStack 'taskkill /F /IM SmartVoice.exe'
    Sleep 2000
  ${EndIf}
FunctionEnd

Function .onInit
  ; Check if already installed
  ReadRegStr $R0 HKLM "Software\WhisperX\SmartVoice" "InstallDir"
  ${If} $R0 != ""
    ; Previous installation found
    MessageBox MB_YESNO|MB_ICONQUESTION "A previous installation was detected at:$\n$\n$R0$\n$\nDo you want to install to the same location? (User data will be preserved)" IDYES useexisting
    ; User chose to change location
    Goto done
    useexisting:
    StrCpy $INSTDIR $R0
    done:
  ${EndIf}
FunctionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"

  ; Check if apps are running
  nsExec::ExecToStack 'taskkill /F /IM SmartVoiceLauncher.exe'
  nsExec::ExecToStack 'taskkill /F /IM SmartVoice.exe'
  Sleep 2000

  ; Remove application files
  RMDir /r "$INSTDIR\Launcher"
  RMDir /r "$INSTDIR\SmartVoice"
  Delete "$INSTDIR\Uninstall.exe"
  Delete "$INSTDIR\SmartVoiceLauncher.exe.lnk"

  ; Remove shortcuts
  Delete "$SMPROGRAMS\WhisperX SmartVoice\SmartVoice Launcher.lnk"
  Delete "$SMPROGRAMS\WhisperX SmartVoice\Uninstall.lnk"
  RMDir "$SMPROGRAMS\WhisperX SmartVoice"
  Delete "$DESKTOP\SmartVoice.lnk"

  ; Remove installation directory if empty
  RMDir "$INSTDIR"

  ; NOTE: User data in %USERPROFILE%\.whisperx_app is NOT deleted
  ; This preserves configuration and history across reinstalls

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperX SmartVoice"
  DeleteRegKey HKLM "Software\WhisperX\SmartVoice"

  ; Inform user about data preservation
  MessageBox MB_ICONINFORMATION "SmartVoice has been uninstalled.$\n$\nYour configuration and transcription history have been preserved in:$\n%USERPROFILE%\.whisperx_app$\n$\nYou can manually delete this folder if desired."

SectionEnd
