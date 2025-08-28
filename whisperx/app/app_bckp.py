import os
import sys
import traceback

from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, \
    QPushButton

import whisperx
import gc
import time

from layout_colorwidget import Color

class WorkerSignals(QObject):
    """Signals from a running worker thread.

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        float indicating % progress
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(float)

class Worker(QRunnable):
    """Worker thread.

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()

        label = QLabel("Select a file")
        layout.addWidget(label)

        self.fileDialog = QFileDialog(self)
        self.fileDialog.setFileMode(QFileDialog.AnyFile)
        self.fileDialog.fileSelected.connect(self.changeSelectedFileDisplay)
        layout.addWidget(self.fileDialog)

        self.label = QLabel(f"Selected: {self.fileDialog.selectedFiles()}")
        layout.addWidget(self.label)


        button = QPushButton("Process")
        button.clicked.connect(self.runProcessing)
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.threadpool = QThreadPool()
        thread_count = self.threadpool.maxThreadCount()
        print(f"Multithreading with maximum {thread_count} threads")

    def runProcessing(self):
        print("Processing Audio File!")
        print("Running with: ", self.fileDialog.selectedFiles())
        self.runTranscription()


    def changeSelectedFileDisplay(self, path:str):
        self.label.setText(path)

    def runPipeline(self, progress_callback):
        path = self.fileDialog.selectedFiles()[0]
        base_dir = os.getcwd()
        relative_path = os.path.relpath(path, base_dir)

        device = "cuda"
        audio_file = str(relative_path)
        batch_size = 8  # 16 reduce if low on GPU mem
        compute_type = "float16"  # change to "int8" float16 if low on GPU mem (may reduce accuracy)

        # 1. Load model
        start = time.time()
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)  # large-v2
        end = time.time()
        progress_callback.emit(36)
        print(f"Model loading took {end - start:.2f} seconds")

        # 2. Load audio
        start = time.time()
        audio = whisperx.load_audio(audio_file)
        end = time.time()
        print(f"Audio loading took {end - start:.2f} seconds")
        progress_callback.emit(40)

        # 3. Transcribe
        start = time.time()
        result = model.transcribe(audio, batch_size=batch_size, language="es")
        end = time.time()
        print(f"Transcription took {end - start:.2f} seconds")
        print(result["segments"])  # before alignment
        progress_callback.emit(80)

        # Free up memory if needed
        # gc.collect(); del model

        # 4. Load alignment model
        start = time.time()
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        end = time.time()
        print(f"Alignment model loading took {end - start:.2f} seconds")
        progress_callback.emit(85)

        # 5. Align
        start = time.time()
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        end = time.time()
        print(f"Alignment took {end - start:.2f} seconds")
        print(result["segments"])  # after alignment
        progress_callback.emit(94)

        # Free up memory if needed
        # gc.collect(); del model_a

        # 6. (Optional) Diarization - add similar timing if used

        print("Final segments with speaker IDs (if diarization applied):")
        print(result["segments"])
        progress_callback.emit(100)

    def runTranscription(self):
        # Pass the function to execute
        worker = Worker(
            self.runPipeline
        )  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        # Execute
        self.threadpool.start(worker)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")


    def progress_fn(self, n):
        print(f"{n:.1f}% done")



app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()