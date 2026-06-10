"""Validation harness for the MPA arrow extractor.

Measures *coverage recall*: of the current-vectors MPA actually draws (by exact band
colour), what fraction lies under a recovered cell. This is merge-proof (a tiled clump
covers its own area) and band-resolved, so it shows exactly which speeds we miss and where.

Run:  python validate.py            # table over the sample set
      python validate.py --map      # also save gap-map PNGs for a few frames
"""
import sys, io, math
import numpy as np
from PIL import Image
from scipy import ndimage
import fetch as F, extractor as ex

CELL = ex.__dict__.get("CELL", 17)          # half-cell footprint (render.CELL)
try:
    import render as R; CELL = R.CELL
except Exception: CELL = 17

# sample spans: 2 areas x tide strength (neap/mid/spring) x time-of-day (slack/flood/ebb)
SAMPLE = [(a, d, t)
          for a in ("SSP", "EBA")
          for d in ("20260609", "20260612", "20260615")          # neap-ish, mid, spring
          for t in ("0700", "1200", "1730")]                      # ~slack, flood, ebb

# measurable bands (exact colour -> label). White & blue handled/[]noted separately.
BANDS = [((255,255,0), "0.5-1.0"), ((128,255,0), "1.0-1.5"),
         ((255,128,0), "1.5-2.0"), ((0,64,128), "2.0-2.5*"),
         ("RED", ">=2.5")]
REDS = [(255,0,0),(128,64,64),(128,0,0),(64,0,0)]

def coverage_mask(shape, arrows):
    H,W = shape[:2]; cov = np.zeros((H,W), bool)
    for a in arrows:
        x,y = int(a["px"]), int(a["py"])
        cov[max(0,y-CELL):y+CELL, max(0,x-CELL):x+CELL] = True
    return cov

def _no_legend(m):
    m[:285, :190] = False        # exclude the colour-key box (top-left) from source counts
    return m

def band_mask(arr, key):
    if key == "RED":
        m = np.zeros(arr.shape[:2], bool)
        for c in REDS: m |= np.all(arr==np.array(c), axis=2)
        return _no_legend(m)
    return _no_legend(np.all(arr==np.array(key), axis=2))

def white_arrow_mask(arr, maxsize=300):
    m = np.all(arr==255, axis=2); lbl,n = ndimage.label(m)
    counts = np.bincount(lbl.ravel())
    small = np.where((counts>0) & (counts<maxsize))[0]
    small = small[small!=0]
    return _no_legend(np.isin(lbl, small))   # white arrows, minus big sea blobs + legend

def frame_recall(area, day, t):
    arr = np.asarray(Image.open(io.BytesIO(F.fetch_bytes(area,day,t))).convert("RGB"))
    arrows = ex.extract_image(arr, area)
    cov = coverage_mask(arr.shape, arrows)
    out = {}
    for key,label in BANDS:
        m = band_mask(arr, key); tot = int(m.sum())
        out[label] = (tot, int((m & cov).sum()))
    w = white_arrow_mask(arr); out["0-0.5 white"] = (int(w.sum()), int((w & cov).sum()))
    return arrows, out

def main(make_map=False):
    agg = {}
    print(f"{'frame':22} {'band':12} {'source_px':>9} {'covered':>8} {'recall':>7}")
    print("-"*64)
    for area,day,t in SAMPLE:
        try: arrows,rec = frame_recall(area,day,t)
        except Exception as e: print(f"{area} {day} {t}: ERROR {e}"); continue
        for label,(tot,cov) in rec.items():
            if tot < 50: continue                      # band not present in this frame
            r = cov/tot
            agg.setdefault(label,[0,0]); agg[label][0]+=tot; agg[label][1]+=cov
            print(f"{area+' '+day[4:]+' '+t:22} {label:12} {tot:9d} {cov:8d} {r:6.0%}")
    print("="*64); print("AGGREGATE coverage recall by band:")
    for label in ["0-0.5 white","0.5-1.0","1.0-1.5","1.5-2.0","2.0-2.5*",">=2.5"]:
        if label in agg:
            tot,cov = agg[label]; print(f"  {label:12} {cov/tot:6.0%}   ({cov:,}/{tot:,} px)")
    print("\n* 2.0-2.5 (blue) is intentionally excluded (collides with depth soundings).")
    if make_map:
        save_gap_maps()

def save_gap_maps():
    for area,day,t in [("SSP","20260615","0400"),("SSP","20260609","1200"),("EBA","20260601","0400")]:
        arr = np.asarray(Image.open(io.BytesIO(F.fetch_bytes(area,day,t))).convert("RGB"))
        arrows = ex.extract_image(arr,area); cov = coverage_mask(arr.shape,arrows)
        src = np.zeros(arr.shape[:2], bool)
        for key,_ in BANDS:
            if key=="2.0-2.5*": continue
            src |= band_mask(arr,key)
        src |= white_arrow_mask(arr)
        out = (np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))*0.6).astype(np.uint8)
        out[src & cov] = (40,170,70)        # covered = green
        out[src & ~cov] = (220,40,40)       # gap = red
        Image.fromarray(out).save(f"_val_{area}_{day}_{t}.png")
        print(f"  saved _val_{area}_{day}_{t}.png  (green=covered, red=gap)")

if __name__ == "__main__":
    main(make_map="--map" in sys.argv)
