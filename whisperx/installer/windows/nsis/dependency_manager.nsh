# Dependency Manager Module for WhisperX Installer
# Handles installation of CUDA, cuDNN, FFmpeg, and VCRedist

!include "LogicLib.nsh"

# Function to install CUDA Toolkit
Function InstallCUDA
    Push $R0  # Exit code
    Push $R1  # Installer path
    Push $R2  # Temp

    DetailPrint "Installing CUDA Toolkit..."

    # Check if CUDA is already installed
    Call CheckExistingCUDA
    Pop $R2

    ${If} $R2 != ""
        DetailPrint "CUDA $R2 is already installed, skipping installation"
        Goto cuda_done
    ${EndIf}

    # Find CUDA installer
    FindFirst $R1 $R2 "$INSTDIR\dependencies\cuda\*.exe"
    ${If} $R1 == ""
        DetailPrint "CUDA installer not found"
        Goto cuda_error
    ${EndIf}
    FindClose $R1

    StrCpy $R1 "$INSTDIR\dependencies\cuda\$R2"

    DetailPrint "Installing CUDA from $R1"

    # Install CUDA silently with minimal components
    nsExec::ExecToLog '"$R1" -s nvcc_12.1 cuobjdump_12.1 nvprune_12.1 cupti_12.1 cublas_dev_12.1 cudart_12.1 cufft_dev_12.1 curand_dev_12.1 cusolver_dev_12.1 cusparse_dev_12.1 thrust_12.1 npp_dev_12.1 nvrtc_dev_12.1 nvml_dev_12.1'
    Pop $R0

    ${If} $R0 == 0
        DetailPrint "CUDA installation completed successfully"
    ${Else}
        DetailPrint "CUDA installation failed with exit code $R0"
        MessageBox MB_YESNO "CUDA installation failed. Continue without GPU acceleration?" IDYES cuda_done
        Abort
    ${EndIf}

    Goto cuda_done

    cuda_error:
    MessageBox MB_YESNO "CUDA installer not found. Continue without GPU acceleration?" IDYES cuda_done
    Abort

    cuda_done:
    Pop $R2
    Pop $R1
    Pop $R0
FunctionEnd

# Function to install cuDNN
Function InstallCUDNN
    Push $R0  # Return code
    Push $R1  # Source path
    Push $R2  # CUDA path
    Push $R3  # Temp

    DetailPrint "Installing cuDNN..."

    # Check if cuDNN is already installed
    Call CheckExistingCUDNN
    Pop $R3

    ${If} $R3 == "true"
        DetailPrint "cuDNN is already installed, skipping installation"
        Goto cudnn_done
    ${EndIf}

    # Find CUDA installation directory
    ReadRegStr $R2 HKLM "SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA" "InstallDir"
    ${If} $R2 == ""
        StrCpy $R2 "$PROGRAMFILES\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    ${EndIf}

    ${If} ${FileExists} "$R2\bin\nvcc.exe"
        DetailPrint "Found CUDA installation at $R2"
    ${Else}
        DetailPrint "CUDA installation not found, cannot install cuDNN"
        Goto cudnn_error
    ${EndIf}

    # Find extracted cuDNN directory
    StrCpy $R1 "$INSTDIR\dependencies\cudnn\extracted"
    ${If} ${FileExists} "$R1\bin\cudnn64_8.dll"
        DetailPrint "Installing cuDNN from $R1"

        # Copy cuDNN files to CUDA installation
        CopyFiles /SILENT "$R1\bin\*" "$R2\bin\"
        CopyFiles /SILENT "$R1\include\*" "$R2\include\"
        CopyFiles /SILENT "$R1\lib\x64\*" "$R2\lib\x64\"

        DetailPrint "cuDNN installation completed"
    ${Else}
        DetailPrint "cuDNN files not found in $R1"
        Goto cudnn_error
    ${EndIf}

    Goto cudnn_done

    cudnn_error:
    MessageBox MB_YESNO "cuDNN installation failed. Continue without full GPU optimization?" IDYES cudnn_done
    Abort

    cudnn_done:
    Pop $R3
    Pop $R2
    Pop $R1
    Pop $R0
FunctionEnd

# Function to install FFmpeg
Function InstallFFmpeg
    Push $R0  # Return code
    Push $R1  # Source path
    Push $R2  # Destination path
    Push $R3  # Temp

    DetailPrint "Installing FFmpeg..."

    # Check if FFmpeg is already available
    nsExec::ExecToStack 'ffmpeg -version'
    Pop $R0
    ${If} $R0 == 0
        DetailPrint "FFmpeg is already available in PATH"
        Goto ffmpeg_done
    ${EndIf}

    # Create FFmpeg directory in installation
    StrCpy $R2 "$INSTDIR\ffmpeg"
    CreateDirectory "$R2"

    # Find extracted FFmpeg directory
    StrCpy $R1 "$INSTDIR\dependencies\ffmpeg\extracted"
    ${If} ${FileExists} "$R1\bin\ffmpeg.exe"
        DetailPrint "Installing FFmpeg from $R1"

        # Copy FFmpeg binaries
        CopyFiles /SILENT "$R1\bin\*" "$R2\"

        # Add to system PATH
        ${EnvVarUpdate} $R0 "PATH" "A" "HKLM" "$R2"

        DetailPrint "FFmpeg installation completed"
    ${Else}
        DetailPrint "FFmpeg binaries not found"
        Goto ffmpeg_error
    ${EndIf}

    Goto ffmpeg_done

    ffmpeg_error:
    MessageBox MB_YESNO "FFmpeg installation failed. Some audio formats may not be supported. Continue?" IDYES ffmpeg_done
    Abort

    ffmpeg_done:
    Pop $R3
    Pop $R2
    Pop $R1
    Pop $R0
FunctionEnd

# Function to install Visual C++ Redistributable
Function InstallVCRedist
    Push $R0  # Exit code
    Push $R1  # Installer path
    Push $R2  # Temp

    DetailPrint "Installing Visual C++ Redistributable..."

    # Check if already installed
    ReadRegStr $R2 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Installed"
    ${If} $R2 == "1"
        DetailPrint "Visual C++ Redistributable is already installed"
        Goto vcredist_done
    ${EndIf}

    # Find VCRedist installer
    FindFirst $R1 $R2 "$INSTDIR\dependencies\vcredist\*.exe"
    ${If} $R1 == ""
        DetailPrint "VCRedist installer not found"
        Goto vcredist_error
    ${EndIf}
    FindClose $R1

    StrCpy $R1 "$INSTDIR\dependencies\vcredist\$R2"

    DetailPrint "Installing VCRedist from $R1"

    # Install VCRedist silently
    nsExec::ExecToLog '"$R1" /install /quiet'
    Pop $R0

    ${If} $R0 == 0
        DetailPrint "VCRedist installation completed"
    ${ElseIf} $R0 == 1638
        DetailPrint "VCRedist already installed (newer version)"
    ${Else}
        DetailPrint "VCRedist installation failed with exit code $R0"
        MessageBox MB_YESNO "Visual C++ Redistributable installation failed. Continue anyway?" IDYES vcredist_done
        Abort
    ${EndIf}

    Goto vcredist_done

    vcredist_error:
    MessageBox MB_YESNO "VCRedist installer not found. Continue anyway?" IDYES vcredist_done
    Abort

    vcredist_done:
    Pop $R2
    Pop $R1
    Pop $R0
FunctionEnd

# Function to update environment variables
!define EnvVarUpdate_NOUNLOAD
!include "EnvVarUpdate.nsh"