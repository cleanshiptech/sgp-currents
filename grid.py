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
from collections import Counter
from scipy import ndimage
from scipy.spatial import cKDTree
import extractor as ex

SAMPLE_COLS = ex.FILL + [(0,64,128)]          # include blue (separable once de-cluttered)
SPEEDS = {**ex.SPEED, (0,64,128): 2.25}
SP = 34                                        # lattice spacing (px), from NN-distance histogram
BAND_MIDS = [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25]   # speed bins; JSON stores index+1 (0=no reading)
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

def _circ(v,p):
    """Circular-mean phase of values v against period p (robust to the centroid wobble, which is
    ~zero-mean around the true node once the current rotates through the day)."""
    return np.angle(np.mean(np.exp(2j*np.pi*v/p)))/(2*np.pi)*p

def _fit_lattice(P, lo=32, hi=37, step=0.05):
    """Find the fixed grid (pitch, x-phase, y-phase) the glyphs are plotted on, by the pitch that
    minimises the median snap residual. Phase from the circular mean (wobble averages out)."""
    best=None
    for pit in np.arange(lo,hi,step):
        x0=_circ(P[:,0],pit); y0=_circ(P[:,1],pit)
        rx=np.abs(((P[:,0]-x0+pit/2)%pit)-pit/2); ry=np.abs(((P[:,1]-y0+pit/2)%pit)-pit/2)
        r=np.median(np.hypot(rx,ry))
        if best is None or r<best[0]: best=(r,pit,x0,y0)
    return best[1],best[2],best[3]

def build_grid(frames, cm, area, union_max=16, per=None, minsup=2):
    """Fixed grid = the lattice the glyphs are plotted on. The arrows sit on a regular grid; a
    glyph's detected centroid only wobbles ~10px around its node as the current rotates. So we
    (1) fit the lattice pitch+phase from the single-glyph centroids (n>0), then (2) SNAP every
    detection to its nearest lattice node. Snapping is rotation-proof -- all of a glyph's wobbling
    centroids fall within half a cell of the same node and collapse automatically (no clustering,
    no over-sampling). Nodes hit in >= `minsup` frames are kept (drops transient noise)."""
    if per is not None:
        dets = [d for f in per for d in f]
    else:
        idx = np.unique(np.linspace(0, len(frames)-1, min(len(frames), union_max)).astype(int))
        dets = [d for f in idx for d in ex.extract_image(frames[f], area)]
    if not dets: return np.empty((0,2))
    A  = np.array([(d["px"],d["py"]) for d in dets], float)              # all detections
    Sg = np.array([(d["px"],d["py"]) for d in dets if d["n"]>0], float)  # single glyphs -> pitch
    pit,x0,y0 = _fit_lattice(Sg if len(Sg)>=50 else A)
    i=np.round((A[:,0]-x0)/pit).astype(int); j=np.round((A[:,1]-y0)/pit).astype(int)
    c=Counter(zip(i.tolist(), j.tolist()))
    nodes=[(x0+a*pit, y0+b*pit) for (a,b),k in c.items() if k>=minsup]   # survey extent = hit nodes
    return np.array(nodes) if nodes else np.empty((0,2))

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
    solid &= pmb                                             # change-mask: static chart features
    b=nearest(solid)                                         # (route/direction arrows) are excluded
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
    """Returns [(cell_index, speed, dir), ...] for grid cells with a reading this frame.
    SPEED read at the cell (complete, incl. blue/white); DIRECTION inherited from the global
    extractor's validated field (`ref`) — correct even for blue/extra cells since the current
    field is locally coherent. (lat/lon are derived from the cell on load, not stored.)"""
    if ref is None: ref=ex.extract_image(arr,area)
    if ref:
        rxy=np.array([[a["px"],a["py"]] for a in ref]); rdir=np.array([a["dir"] for a in ref])
        rsp=np.array([a["speed"] for a in ref]); tree=cKDTree(rxy)
    out=[]
    for i,(cx,cy) in enumerate(grid):
        s=sample_cell(arr,cm,cx,cy)                       # change-masked -> static features excluded
        if not s: continue
        dd,j=(tree.query([cx,cy]) if ref else (1e9,0))
        # blue is reliable only when the size-gated detector also sees it there; otherwise the
        # cell-read picked up a dynamic depth-sounding -> drop (no phantom 2-2.5 reds).
        if abs(s["speed"]-2.25)<0.1 and not (ref and dd<=20 and abs(rsp[j]-2.25)<0.1): continue
        out.append((i, s["speed"], float(rdir[j]) if ref else 0.0))
    return out
