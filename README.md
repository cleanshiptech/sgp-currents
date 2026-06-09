# MPA Current Time-Series (online)

Pick a point on a map → get the tidal current **through the day** (speed time-series,
set-direction, max / mean / %-over-limit), digitised from the MPA Digital Tidal Atlas
current-vector frames. Speed is shown as a **6-band filled grid**; direction as arrows.

## Architecture (no disk cache — built for Streamlit Community Cloud)
- **Runtime cache:** `st.cache_data` (in-memory), shared across users while warm. No disk writes.
- **Persistence = GitHub repo as read-only database, written by CI (not the app):**
  the Action `.github/workflows/refresh-data.yml` runs nightly, digitises every day of the
  **current month** (and **next month** automatically, once MPA — which is a tidal
  *forecast* — publishes it), and commits compact JSON to a **separate `data` branch**.
  It is incremental (skips days already built); pass `force: true` on a manual run to
  rebuild the whole window after an extractor change. The app reads the branch via a raw
  URL (`DATA_BASE` secret), so the code repo is never touched and there's no redeploy loop.
- The app is **read-only**: it never writes data (Streamlit Cloud has no disk and no repo
  credentials). Dates outside the archived window simply show a "not archived yet" note.

Set in Streamlit secrets:
```
DATA_BASE = "https://raw.githubusercontent.com/cleanshiptech/sgp-currents/data/data"
# MPA_XLSX = "..."   # optional: path to the MPA spreadsheet for the exact-value overlay
```

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Pick a date in the current month (it loads instantly from the archive). Choose a basemap,
a map layer (snapshot at a time, or per-cell max/mean over the day), then **click the map**
to drop a worksite.

## CI data builder (run locally or by the Action)
```
python build_data.py SSP 20260615 SSP 20260616
```
Writes `data/SSP_20260615.json`. The workflow commits these to the `data` branch daily.

## Representation
- **Speed → 6-band filled grid** (dark-green→dark-red), bands = the native 0.5-kn source bins (no false precision).
- **Direction → anti-aliased icon darts** (white halo, supersampled), fixed size — direction only.
- Time-series uses the same band scale as background, with the ROV limit (1.3 kn) as a dashed reference.

## Files
`extractor.py` GIF/array→arrows · `render.py` grid+arrow overlays, aggregate, series ·
`fetch.py` MPA frame fetch · `data.py` precomputed/live loading · `build_data.py` CI writer ·
`app.py` Streamlit UI.

## Caveats
- Speed binned to 0.5 kn; the series shows the bin midpoint ±0.25 kn. Exact per-station
  numbers stay in the spreadsheet (overlaid as a dashed line near known stations).
- Arrows under MPA's drawn station markers read poorly; merged fast-channel arrows are
  under-recovered (anchorages are clean).
- **"No reading" ≠ calm.** Unrecovered cells render as a faint grey scaffold, NOT as
  "calm": a blank cell may be genuinely slack OR a 0.5–2 kn arrow we failed to digitise
  (white 0–0.5 merges with the sea-fill; short slow arrows fall below detection). Recovery
  is best at peak flood/ebb and thins toward slack — never read grey as "safe to dive".
- **Small-arrow recovery.** `amin=8` + the elongation gate recover the smallest darts
  (white 0–0.5, short yellow 0.5–1), roughly doubling coverage toward slack. Their
  orientation is reliable (~15° vs the field) but the tip/tail sign flips ~⅓ of the time,
  so `_resolve_heads` re-orients each tiny arrow to agree with its confident neighbours.
  White arrows are still lost where the chart sea-fill is *itself* white (no contrast).
- **Outline recovery of background-colour calm arrows.** Some areas (EBA, in
  `CALM_OUTLINE_AREAS`) draw the 0–0.5 kn band in their *sea-background* colour, so only the
  black outline shows and colour-matching gets none. `_outline_calm` recovers them from the
  dark outline, rejects chart boundary/cable lines via an elongation cap + bbox-fill floor,
  and orients each against the colour-detected field. (SSP draws 0–0.5 in white, so it
  doesn't need this.)
- **Colour-blocked band.** The **2.0–2.5 kn** band (dark blue `0,64,128`) is *excluded*: it's
  indistinguishable from printed depth soundings (0 genuine arrows even at peak ebb), so
  reading it only produced false high-current cells. Reliable range is **0.5–2.0 kn** —
  which brackets the 1.3 kn ROV limit, the operationally relevant threshold.
- Extraction uses an elongation shape-gate (`extractor._elong`) so short slow-current
  arrows are recovered (~430/frame) without admitting round chart-text specks.
- Both **SSP** and **EBA** are calibrated and validated (direction error 1–8° at known
  stations; recovered small arrows ~12° median vs the local field).

## Adding EBA
```python
import extractor as ex, numpy as np; from PIL import Image
rows,cols = ex.detect_graticule(np.asarray(Image.open("TA_EBA....gif").convert("RGB")))
# map rows->printed lat labels, cols->printed lon labels; add GEOREF["EBA"]; uncomment EBA in the workflow
```
