# Repository Guidelines

## Project Structure & Module Organization
WhisperX_for_Windows organizes the transcription pipeline under `whisperx/`, including CLI entry point in `whisperx/__main__.py` and model assets in `whisperx/assets/`. The WPF client lives in `WhisperXGUI/` with `MainWindow.xaml` for layout and `MainWindow.xaml.cs` for code-behind. Shared docs and figures sit at the repository root; keep large checkpoints out of Git or track them with Git LFS.

## Build, Test, and Development Commands
Run `uv pip install -e .` from the root to register the Python package against the locked dependencies in `uv.lock`. Validate the CLI path with `python -m whisperx --help` and smoke test against media via `python -m whisperx examples/sample.wav --output-dir out/`. Build the desktop app using `dotnet build WhisperXGUI/WhisperXGUI.csproj -c Release`, adding `-r win-x64` when you need a distributable bundle. For local GUI iterations, run `dotnet run --project WhisperXGUI/WhisperXGUI.csproj`.

## Coding Style & Naming Conventions
Python code follows PEP 8 with 4-space indentation; prefer descriptive verb_noun function names such as `load_alignment_data`. Format and lint with `black whisperx` and `ruff` before committing. In C#, use .NET casing conventions, keep XAML resources alphabetically ordered, and prefer binding-based updates over code-behind mutations.

## Testing Guidelines
Python unit tests live under `tests/` mirroring module paths and run via `pytest`. Mark GPU-dependent cases with `@pytest.mark.cuda` so they skip safely on CPU hosts. When touching the CLI, run the sample command above and capture notable logs. For GUI work, manually verify load, inference, and cancel flows and document the results in your PR.

## Commit & Pull Request Guidelines
Write imperative, concise commit subjects such as "Add diarization retry." PRs should summarize scope, link related issues, and attach CLI logs or GUI screenshots as evidence. Call out CPU versus GPU coverage, note required model downloads, and highlight any environment variables contributors must set before running.
