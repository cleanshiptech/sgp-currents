"""Fixed-grid extraction (v2).

Verified that arrow glyphs are anchored at fixed cell points (centroid moves ~0.8px between
frames of similar direction; the wobble is the asymmetric dart orbiting its anchor, growing
with direction change). So we recover the fixed grid once per day, then read each cell's
band-colour + angle every frame.

Temporal differencing removes the static chart layer (soundings, depth text, coastlines,
background, legend) -> blue (2.0-2.5, vs soundings) and white-on-white (0-0.5) become
readable. The grid is stable -> clean per-cell time-series. The merged fast channel (no
clean single arrows to anchor) is filled with a lattice clipped to the fast region.

Needs all of a day's frames. data.build() drives it; extractor.extract_image stays for
single-frame use.
"""
import numpy as np, math
from scipy import ndimage
from scipy.spatial import cKDTree
import extractor as ex

SAMPLE_COLS = ex.FILL + [(0,64,128)]          # include blue (separable once de-cluttered)
SPEEDS = {**ex.SPEED, (0,64,128): 2.25}
SP = 34                                        # lattice spacing (px), from NN-distance histogram
# Solid bands have no clutter -> read RAW colour (robust, like the global extractor).
# Ambiguous bands share colour with static chart features -> gate by the change-mask only:
#   white (255,255,255) == sea-fill background;  blue (0,64,128) == depth soundings.
SOLID = [c for c in ex.FILL if c != (255,255,255)]
AMBIG = [(255,255,255), (0,64,128)]

def change_mask(frames):
    """frames: (N,H,W,3) uint8 of one day. True where any pixel varies across time."""
    cm = (frames.max(0) != frames.min(0)).any(-1)
    cm[:285, :190] = False                     # legend box + changing timestamp
    return cm

def _anchors(frames, cm, R=16, support=3, smin=12, smax=200):
    """Fixed cell centres = day-averaged centroids of clean single arrows (the orbit
    averages out, leaving the anchor). Greedy-cluster within R, keep well-supported."""
    pts=[]
    for arr in frames:
        for c in SAMPLE_COLS:
            m = np.all(arr==np.array(c),axis=2) & cm; lbl,n = ndimage.label(m)
            if n==0: continue
            s = np.bincount(lbl.ravel())
            for i in range(1,n+1):
                if smin<=s[i]<=smax:
                    ys,xs = np.where(lbl==i)
                    if ex._elong(ys,xs) >= 2.5: pts.append((xs.mean(), ys.mean()))
    cx=[];cy=[];cn=[]
    for x,y in pts:
        if cx:
            d2=(np.array(cx)-x)**2+(np.array(cy)-y)**2; j=int(d2.argmin())
            if d2[j]<R*R: cn[j]+=1; cx[j]+=(x-cx[j])/cn[j]; cy[j]+=(y-cy[j])/cn[j]; continue
        cx.append(x); cy.append(y); cn.append(1)
    cx=np.array(cx); cy=np.array(cy); cn=np.array(cn); k=cn>=support
    return np.c_[cx[k], cy[k]]

def _fast_fill(frames, cm, anchors, S=SP):
    """Lattice points in the persistent fast region (>=1.5 kn ever) not already anchored —
    covers the merged channel where arrows touch and leave no clean single centroids."""
    H,W = frames.shape[1:3]
    fast = np.zeros((H,W), bool)
    for arr in frames:
        for c in ex.FILL:
            if ex.SPEED[c] >= 1.5: fast |= np.all(arr==np.array(c),axis=2)
    fast &= cm; fast = ndimage.binary_dilation(fast, iterations=2)
    best=(-1,0,0)
    for x0 in range(0,S,4):
        for y0 in range(0,S,4):
            gx,gy = np.meshgrid(np.arange(x0,W,S), np.arange(y0,H,S))
            sc = fast[gy,gx].sum()
            if sc>best[0]: best=(sc,x0,y0)
    _,x0,y0 = best
    gx,gy = np.meshgrid(np.arange(x0,W,S), np.arange(y0,H,S))
    cand = np.c_[gx.ravel(), gy.ravel()]; cand = cand[fast[cand[:,1], cand[:,0]]]
    if len(anchors) and len(cand):
        d,_ = cKDTree(anchors).query(cand); cand = cand[d>18]
    return cand.astype(float)

def build_grid(frames, cm, area, dedup=14, union_max=16, per=None):
    """Complete fixed grid = the union of per-frame detections (so every cell active at any
    tide state is present, incl. the merged fast channel via extract_image's tiling), plus a
    lattice fill of the persistent fast region, deduplicated at `dedup` px. Seeded densest-
    frame-first. `per` (precomputed [extract_image(frame) ...] aligned to frames) is reused
    if given; otherwise the detector runs on up to `union_max` frames across the day."""
    if per is not None:
        order = sorted(range(len(per)), key=lambda f: -len(per[f]))
        pts = [(d["px"], d["py"]) for f in order for d in per[f]]
    else:
        idx = np.unique(np.linspace(0, len(frames)-1, min(len(frames), union_max)).astype(int))
        pe = {f: ex.extract_image(frames[f], area) for f in idx}
        order = sorted(idx, key=lambda f: -len(pe[f]))
        pts = [(d["px"], d["py"]) for f in order for d in pe[f]]
    ff = _fast_fill(frames, cm, np.array(pts) if pts else np.zeros((0,2)))
    pts += [(float(x), float(y)) for x, y in ff]
    cells=[]; hsh={}; D2=dedup*dedup; b=dedup                    # greedy spatial-hash dedup, O(n)
    for x,y in pts:
        gx,gy=int(x//b),int(y//b); near=False
        for ix in (gx-1,gx,gx+1):
            for iy in (gy-1,gy,gy+1):
                for cx,cy in hsh.get((ix,iy),()):
                    if (cx-x)**2+(cy-y)**2<D2: near=True; break
                if near: break
            if near: break
        if not near: cells.append((x,y)); hsh.setdefault((gx,gy),[]).append((x,y))
    return np.array(cells)

def sample_cell(arr, cm, cx, cy, rb=16):
    """Cascade read at a fixed cell: try the UNAMBIGUOUS solid colours first (yellow/green/
    orange/reds — those exact RGBs never appear as chart clutter); only if no solid arrow is
    present near the anchor, fall back to the AMBIGUOUS white/blue gated by the change-mask
    (they share colour with the sea background / depth soundings). Take the blob nearest the
    anchor (handles wobble, avoids neighbours); read its dominant band -> speed, axis -> angle.
    Returns px,py at the FIXED cell position (stable across frames)."""
    yb=max(0,int(cy-rb)); xb=max(0,int(cx-rb)); pb=arr[yb:int(cy+rb),xb:int(cx+rb)]; pmb=cm[yb:int(cy+rb),xb:int(cx+rb)]
    acx, acy = cx-xb, cy-yb
    def nearest(mask):
        lbl,n=ndimage.label(mask); best=None
        for i in range(1,n+1):
            ys,xs=np.where(lbl==i)
            if len(xs)<5: continue
            d=(xs.mean()-acx)**2+(ys.mean()-acy)**2
            if best is None or d<best[0]: best=(d,lbl==i,ys,xs)
        return best
    solid=np.zeros(pb.shape[:2],bool)
    for c in SOLID: solid |= np.all(pb==np.array(c),axis=2)
    b=nearest(solid)
    if b is None:                                            # no clear arrow -> try white/blue
        amb=np.zeros(pb.shape[:2],bool)
        for c in AMBIG: amb |= np.all(pb==np.array(c),axis=2) & pmb
        b=nearest(amb)
        if b is None: return None
    _,blob,ys,xs=b
    cols={c:int((np.all(pb==np.array(c),axis=2)&blob).sum()) for c in SAMPLE_COLS}
    c=max(cols,key=cols.get)
    if cols[c]<5: return None
    return dict(px=float(cx),py=float(cy),speed=SPEEDS[c],n=len(xs))   # speed only; dir set per-frame

def extract_frame(arr, cm, grid, area, ref=None):
    """Per-cell SPEED from the v2 grid read (complete, incl. blue/white); DIRECTION from the
    global extractor, whose headings are validated to ~3 deg everywhere (it derives the merged
    clump's true axis). Each cell inherits the nearest validated flow direction — correct even
    for blue/extra cells, since the current field is locally coherent."""
    p2lon,p2lat,_,_=ex.georef_funcs(area)
    if ref is None: ref=ex.extract_image(arr,area)       # validated direction field
    if ref:
        rxy=np.array([[a["px"],a["py"]] for a in ref]); rdir=np.array([a["dir"] for a in ref])
        tree=cKDTree(rxy)
    out=[]
    for cx,cy in grid:
        s=sample_cell(arr,cm,cx,cy)
        if not s: continue
        s["dir"]=float(rdir[tree.query([cx,cy])[1]]) if ref else 0.0
        s["lat"]=float(p2lat(cy)); s["lon"]=float(p2lon(cx))
        out.append(s)
    return out
