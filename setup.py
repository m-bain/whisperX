import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="whisperx",
    py_modules=["whisperx"],
    version="2.0",
    description="Time-Accurate Automatic Speech Recognition using Whisper.",
    readme="README.md",
    python_requires=">=3.8",
    author="Max Bain",
    url="https://github.com/m-bain/whisperx",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points = {
        'console_scripts': ['whisperx=whisperx.transcribe:cli'],
    },
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
