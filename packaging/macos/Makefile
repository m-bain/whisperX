# macOS packaging for the Manuscript (WhisperX web app) — design (B), see
# ../../MACOS_INSTALLER.md. Orchestrates the phases; build.py owns the hard parts
# (relocatable runtime + leaf-first deep codesign).
#
# Quick start (local PoC, ad-hoc signed, no notarization — builds & launches now):
#   make poc && open build/Manuscript.app
#
# Release (needs a 'Developer ID Application' cert + a notarytool profile):
#   make release IDENTITY="Developer ID Application: Anvil7 UG (Q8HKVK78G9)" \
#                NOTARY_PROFILE=manuscript-notary
#
# Per-phase targets let you re-run one step without rebuilding the ~3 GB runtime.

PY := python3
DRIVER := $(PY) build.py

# Signing identity: "-" (ad-hoc) by default; override for a notarizable build.
IDENTITY ?= -
export IDENTITY
export NOTARY_PROFILE

BUILD := build
APP := $(BUILD)/Manuscript.app
STAMP := $(BUILD)/.stamp

.PHONY: all poc release bundle skeleton runtime app ffmpeg launcher sign notarize dmg clean

## poc: assemble + ad-hoc sign a locally launchable .app (default goal)
poc: bundle sign
	@echo "PoC built: $(APP)  (ad-hoc signed — local smoke-test only)"
	@echo "Launch:  open $(APP)   (or: $(APP)/Contents/MacOS/Manuscript for logs)"

## release: signed + notarized + stapled .app, packaged as a DMG
release: bundle sign notarize dmg
	@echo "Release DMG built under $(BUILD)/"

## bundle: full .app tree, no signing
bundle: $(STAMP).skeleton $(STAMP).runtime $(STAMP).app $(STAMP).ffmpeg $(STAMP).launcher

$(STAMP).skeleton:
	$(DRIVER) skeleton && mkdir -p $(BUILD) && touch $@
$(STAMP).runtime: $(STAMP).skeleton
	$(DRIVER) runtime && touch $@
$(STAMP).app: $(STAMP).skeleton
	$(DRIVER) app && touch $@
$(STAMP).ffmpeg: $(STAMP).skeleton
	$(DRIVER) ffmpeg && touch $@
$(STAMP).launcher: $(STAMP).skeleton launcher.c
	$(DRIVER) launcher && touch $@

# Individual phases (force re-run, ignore stamps).
skeleton: ; $(DRIVER) skeleton
runtime:  ; $(DRIVER) runtime
app:      ; $(DRIVER) app
ffmpeg:   ; $(DRIVER) ffmpeg
launcher: ; $(DRIVER) launcher
sign:     ; $(DRIVER) sign
notarize: ; $(DRIVER) notarize
dmg:      ; $(DRIVER) dmg

## clean: remove build/
clean: ; $(DRIVER) clean
