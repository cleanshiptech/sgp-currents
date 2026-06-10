"""MPA Digital Tidal Atlas current-field extractor.
GIF/array -> georeferenced arrows: (lat, lon, speed_bin_mid_kn, set_dir_degT).
Channels decoupled: speed<-colour (0.5-kn bins), dir<-orientation+head. Size ignored."""
import math, re, os
import numpy as np
from PIL import Image
from scipy import ndimage

# NB: (0,64,128) dark blue (2.0-2.5 kn) is deliberately excluded — on the MPA chart it is
# indistinguishable from printed depth soundings (thousands of them), so every "detection"
# is a false positive (verified: 0 genuine arrows even at peak ebb). Keeping it only painted
# fake red "danger" cells. Max reliably-read band is therefore 1.5-2.0 kn (orange).
LEGEND=[((255,255,255),0.25),((255,255,0),0.75),((128,255,0),1.25),((255,128,0),1.75),
        ((255,0,0),2.75),((128,64,64),3.25),((128,0,0),3.75),((64,0,0),4.25)]
FILL=[c for c,_ in LEGEND]; SPEED={c:s for c,s in LEGEND}
def bin_label(c): s=SPEED[c]; return f"{s-0.25:.1f}-{s+0.25:.1f} kn"

# 6 display bands aligned 1:1 to native 0.5-kn bins: (upper_kn, RGB)
# 0-0.5 is white = "calm" (near-zero / slack), so a white cell never conflicts with a
# coloured one; the ramp then runs green -> yellow -> orange -> red with rising speed.
# Speed ramp for *measured* cells (band[0] 0-0.5 = light blue, a real slow reading with a
# direction). Distinct from NO_READING (grey), used only for grid cells with no recovered
# arrow — which may be genuinely calm OR an arrow we failed to digitise, so never "calm".
DISPLAY_BANDS=[(0.5,(150,190,225)),(1.0,(60,170,90)),(1.5,(225,205,55)),
               (2.0,(240,150,35)),(2.5,(226,59,39)),(99.0,(140,16,16))]
DISPLAY_LABELS=["0–0.5","0.5–1.0","1.0–1.5","1.5–2.0","2.0–2.5","≥2.5"]
NO_READING=(200,206,212)       # grid cell with no recovered arrow (grey; not "calm")
def band_color(speed):
    for up,c in DISPLAY_BANDS:
        if speed<up: return c
    return DISPLAY_BANDS[-1][1]
# operational reference (kn) for the time-series
GREEN_MAX, AMBER_MAX, ROV_LIMIT = 0.8, 1.1, 1.3

GEOREF={"SSP":dict(x_px=[504,1024,1543],x_deg=[103.6997,103.7997,103.8997],
                   y_px=[109,419,729],   y_deg=[1.2400,1.1600,1.0800]),
        "EBA":dict(x_px=[618,1137],      x_deg=[103.9997,104.0997],
                   y_px=[252,872],       y_deg=[1.3200,1.1600])}
# Areas that draw the 0-0.5 kn band in their sea-background colour (only the black outline
# shows) -> recover those calm arrows from the outline. SSP draws 0-0.5 in white, which the
# colour matcher already catches, so it is NOT listed here.
CALM_OUTLINE_AREAS={"EBA"}

def _fit(px,deg):
    m,b=np.polyfit(np.array(px,float),np.array(deg,float),1); return m,b
def georef_funcs(area):
    g=GEOREF[area]; mx,bx=_fit(g["x_px"],g["x_deg"]); my,by=_fit(g["y_px"],g["y_deg"])
    return (lambda x:mx*x+bx),(lambda y:my*y+by),(lambda lon:(lon-bx)/mx),(lambda lat:(lat-by)/my)
def detect_graticule(arr):
    g=arr.mean(2);H,W=g.shape;dark=g<70
    rf=dark[:,200:W-50].mean(1);cf=dark[120:H-30,:].mean(0)
    def pk(f,thr,sep=15):
        idx=np.where(f>thr)[0];o=[]
        for i in idx:
            if o and i-o[-1]<=sep:continue
            o.append(int(i))
        return o
    return pk(rf,.5),pk(cf,.5)
def area_from_name(p):
    m=re.search(r"TA_([A-Z]+)\d{12}\.gif",os.path.basename(p)); return m.group(1) if m else None

def _axis_head(ys,xs):
    x=xs.astype(float)-xs.mean();y=ys.astype(float)-ys.mean()
    u20=(x*x).mean();u02=(y*y).mean();u11=(x*y).mean()
    th=0.5*math.atan2(2*u11,u20-u02);ux,uy=math.cos(th),math.sin(th)
    t=x*ux+y*uy;p=-x*uy+y*ux
    mw=lambda s:(p[s].max()-p[s].min()) if s.sum()>2 else 0
    sign=1 if mw(t>0)>=mw(t<0) else -1
    return sign*ux,sign*uy
def _bearing(dx,dy): return (math.degrees(math.atan2(dx,-dy)))%360
def _elong(ys,xs):
    """Principal-axis elongation (eigenvalue ratio of the pixel covariance).
    Real arrow darts are highly elongated (~15); round chart-text/sounding specks ~1."""
    x=xs.astype(float)-xs.mean();y=ys.astype(float)-ys.mean()
    u20=(x*x).mean();u02=(y*y).mean();u11=(x*y).mean()
    tr=u20+u02;root=math.sqrt(max(((u20-u02)/2)**2+u11*u11,0));l2=tr/2-root
    return (tr/2+root)/l2 if l2>1e-6 else 999.0

def _outline_calm(arr, area, ref, dark_thr=90, amin=8, amax=200, emin=2.5, emax=10,
                  bbox_max=50, bbf_min=0.45, dedupe=13, k=6, rmax=140):
    """Recover 0-0.5 kn arrows whose fill is the sea-background colour (only their dark
    outline is visible). A line-fragment filter rejects chart boundaries/cables: real darts
    are moderately elongated and fill their bbox (>=0.45), while line strokes are thinner
    (lower bbox-fill) and more elongated (>emax). Each kept arrow is classed 0-0.5 and
    oriented against the colour-detected field `ref` (its own head sign is unreliable)."""
    p2lon,p2lat,_,_=georef_funcs(area)
    dark=arr.astype(int).sum(2)<dark_thr; lbl,n=ndimage.label(dark); objs=ndimage.find_objects(lbl)
    rx=np.array([a["px"] for a in ref]); ry=np.array([a["py"] for a in ref]); rd=np.array([a["dir"] for a in ref])
    out=[]
    for kk in range(1,n+1):
        sl=objs[kk-1]; ys,xs=np.where(lbl[sl]==kk); L=len(xs)
        if not(amin<=L<=amax): continue
        e=_elong(ys,xs)
        if e<emin or e>emax: continue
        H=ys.max()-ys.min()+1; Wd=xs.max()-xs.min()+1
        if max(H,Wd)>bbox_max or L/(H*Wd)<bbf_min: continue
        cx,cy=float(xs.mean()+sl[1].start),float(ys.mean()+sl[0].start)
        if len(rx) and ((rx-cx)**2+(ry-cy)**2).min()<dedupe*dedupe: continue   # already a colour arrow
        hx,hy=_axis_head(ys+sl[0].start,xs+sl[1].start); b=_bearing(hx,hy)
        if len(rx):                                          # orient against the colour field
            d=np.hypot(rx-cx,ry-cy); idx=np.argsort(d)[:k]; idx=idx[d[idx]<=rmax]
            if len(idx)>=3:
                lf=math.degrees(math.atan2(np.sin(np.deg2rad(rd[idx])).mean(),np.cos(np.deg2rad(rd[idx])).mean()))%360
                if abs((b-lf+180)%360-180)>90: b=(b+180)%360
        out.append(dict(px=cx,py=cy,lat=float(p2lat(cy)),lon=float(p2lon(cx)),speed=0.25,dir=b,n=L))
    return out

def _resolve_heads(out, head_min=18, k=6, rmax=140):
    """Resolve the 180-deg head ambiguity for small arrows (n<head_min). Their orientation
    *axis* is reliable (~15 deg) but the tip/tail sign flips ~1/3 of the time; the tidal
    field is smooth, so pick whichever direction agrees with the nearby confident arrows."""
    ref=[a for a in out if a["n"]>=head_min]
    if not ref: return out
    rx=np.array([a["px"] for a in ref]); ry=np.array([a["py"] for a in ref])
    rdir=np.deg2rad([a["dir"] for a in ref])
    for a in out:
        if a["n"]>=head_min: continue
        d=np.hypot(rx-a["px"],ry-a["py"]); idx=np.argsort(d)[:k]; idx=idx[d[idx]<=rmax]
        if len(idx)<3: continue
        lf=math.degrees(math.atan2(np.sin(rdir[idx]).mean(),np.cos(rdir[idx]).mean()))%360
        if abs((a["dir"]-lf+180)%360-180)>90: a["dir"]=(a["dir"]+180)%360
    return out

def _merged_cells(arr, area, step=30, amax=260, bbmin=55, bbmax=1000, smin=1.0):
    """Recover fast-channel currents that the single-arrow pass drops. On strong tides the
    long fast darts touch and form one connected colour blob bigger than `amax`, leaving a
    blank in the strait's strongest (most dangerous) flow. Such a clump is tiled into cells
    (band speed + the clump's principal-axis direction; head sign resolved later vs
    neighbours). Restricted to >=1 kn bands so the white/yellow sea-fill can't qualify, and
    to a bbox window that excludes the compact legend swatches and the huge background."""
    p2lon,p2lat,_,_=georef_funcs(area); out=[]
    for c in FILL:
        if SPEED[c]<smin: continue
        m=np.all(arr==np.array(c),axis=2); lbl,n=ndimage.label(m); objs=ndimage.find_objects(lbl)
        for k in range(1,n+1):
            sl=objs[k-1]; ys,xs=np.where(lbl[sl]==k)
            if len(xs)<=amax: continue
            H=ys.max()-ys.min()+1; W=xs.max()-xs.min()+1
            if not(bbmin<max(H,W)<bbmax): continue
            ys=ys+sl[0].start; xs=(xs+sl[1].start)
            x=xs.astype(float)-xs.mean(); y=ys.astype(float)-ys.mean()
            u20=(x*x).mean();u02=(y*y).mean();u11=(x*y).mean(); th=0.5*math.atan2(2*u11,u20-u02)
            br=_bearing(math.cos(th),math.sin(th))
            for px,py in {(int(round(a/step)*step),int(round(b/step)*step)) for a,b in zip(xs,ys)}:
                out.append(dict(px=float(px),py=float(py),lat=float(p2lat(py)),lon=float(p2lon(px)),
                                speed=SPEED[c],dir=br,n=0))
    return out

# amin=8 recovers the smallest slow-current arrows (white 0-0.5, short yellow 0.5-1).
# emin shape-gate rejects round chart specks (text/soundings); _resolve_heads fixes the
# 180-deg head flips that plague tiny darts. _merged_cells fills the fast channel where
# strong arrows merge into over-size blobs. Validated: tiny-arrow orientation ~15 deg median.
def extract_image(arr, area, amin=8, amax=260, bbox_max=50, emin=2.5):
    """arr: HxWx3 uint8 RGB. Returns list of arrow dicts."""
    if area not in GEOREF: raise ValueError(f"No georeference for '{area}'.")
    p2lon,p2lat,_,_=georef_funcs(area); out=[]
    for c in FILL:
        m=np.all(arr==np.array(c),axis=2); lbl,n=ndimage.label(m); objs=ndimage.find_objects(lbl)
        for k in range(1,n+1):
            sl=objs[k-1]; ys,xs=np.where(lbl[sl]==k)
            if not(amin<=len(xs)<=amax): continue
            if _elong(ys,xs)<emin: continue
            ys=ys+sl[0].start; xs=xs+sl[1].start
            if max(ys.max()-ys.min()+1,xs.max()-xs.min()+1)>bbox_max: continue
            hx,hy=_axis_head(ys,xs); cx,cy=float(xs.mean()),float(ys.mean())
            out.append(dict(px=cx,py=cy,lat=float(p2lat(cy)),lon=float(p2lon(cx)),
                            speed=SPEED[c],dir=_bearing(hx,hy),n=len(xs)))
    out=out+_merged_cells(arr,area,amax=amax)     # fill merged fast-channel clumps
    out=_resolve_heads(out)                        # orient small + merged cells vs neighbours
    if area in CALM_OUTLINE_AREAS:                 # add background-colour 0-0.5 arrows (EBA)
        out=out+_outline_calm(arr,area,out)
    return out
def extract_arrows(path, **kw):
    area=area_from_name(path)
    return extract_image(np.asarray(Image.open(path).convert("RGB")), area, **kw), area
def extract_bytes(data, area, **kw):
    import io
    return extract_image(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), area, **kw)

def query_nearest(arrows, lat, lon, max_km=1.0):
    if not arrows: return None
    cl=math.cos(math.radians(lat)); best=None
    for a in arrows:
        d=math.hypot((a["lon"]-lon)*cl*111.32,(a["lat"]-lat)*110.57)
        if best is None or d<best[0]: best=(d,a)
    d,a=best
    if d>max_km: return None
    r=dict(a); r["dist_km"]=d; return r
