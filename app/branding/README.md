# Manuscript — macOS app icon

Editorial-italic Literata "M" monogram on a parchment squircle (the selected
mark; same art as the web favicon at `app/static/favicon.svg`).

## Contents

- `AppIcon.appiconset/` — drop-in Xcode **Asset Catalog** icon set (mac idiom,
  16/32/128/256/512 @1x+@2x, full coverage, transparent squircle corners).
- `Manuscript.icns` — prebuilt icon for bundling without an asset catalog.
- `Manuscript.svg` — scalable master (live text; convert text→outline before
  handing to a designer so the `M` is locked regardless of installed fonts).
- `Manuscript-1024.png` — 1024² master raster.

## Use in Xcode

**Asset catalog (recommended):** drag `AppIcon.appiconset` into your target's
`Assets.xcassets`, then set *Target → General → App Icons and Launch Screen →
App Icon* to `AppIcon`.

**Direct `.icns`:** add `Manuscript.icns` to the target and set
`CFBundleIconFile = Manuscript` in `Info.plist`.

## Regenerating from the SVG master

The PNGs came from the design export. To rebuild the set from a single master:

```bash
# rasterize Manuscript.svg → Manuscript-{16,32,64,128,256,512,1024}.png first
# (e.g. rsvg-convert / cairosvg / a browser), then:
iconutil -c icns Manuscript.iconset -o Manuscript.icns
```

where `Manuscript.iconset/` holds the standard names
(`icon_16x16.png`, `icon_16x16@2x.png`, … `icon_512x512@2x.png`) — identical to
the files in `AppIcon.appiconset/`.

## Notes

- Icons are **full-bleed squircle** (no extra transparent margin). If you target
  the macOS Big Sur icon grid (art inset ~824px inside 1024² with a system
  shadow), re-export the master with that padding before packaging.
- A dark-appearance variant exists in the design export
  (`Manuscript-dark.svg` / `*-dark-*.png`); add it as a separate appearance in
  the asset catalog if you want a distinct dark dock icon. `icon-dark.svg` is
  also staged in `app/static/`.
