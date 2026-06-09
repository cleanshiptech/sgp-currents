"""MPA Digital Tidal Atlas current-field extractor.
GIF/array -> georeferenced arrows: (lat, lon, speed_bin_mid_kn, set_dir_degT).
Channels decoupled: speed<-colour (0.5-kn bins), dir<-orientation+head. Size ignored."""
import math, re, os
import numpy as np
from PIL import Image
from scipy import ndimage

LEGEND=[((255,255,255),0.25),((255,255,0),0.75),((128,255,0),1.25),((255,128,0),1.75),
        ((0,64,128),2.25),((255,0,0),2.75),((128,64,64),3.25),((128,0,0),3.75),((64,0,0),4.25)]
FILL=[c for c,_ in LEGEND]; SPEED={c:s for c,s in LEGEND}
def bin_label(c): s=SPEED[c]; return f"{s-0.25:.1f}-{s+0.25:.1f} kn"

# 6 display bands aligned 1:1 to native 0.5-kn bins: (upper_kn, RGB)
DISPLAY_BANDS=[(0.5,(26,122,58)),(1.0,(124,196,107)),(1.5,(245,224,66)),
               (2.0,(245,154,35)),(2.5,(226,59,39)),(99.0,(140,16,16))]
DISPLAY_LABELS=["0–0.5","0.5–1.0","1.0–1.5","1.5–2.0","2.0–2.5","≥2.5"]
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

def extract_image(arr, area, amin=60, amax=260, bbox_max=50):
    """arr: HxWx3 uint8 RGB. Returns list of arrow dicts."""
    if area not in GEOREF: raise ValueError(f"No georeference for '{area}'.")
    p2lon,p2lat,_,_=georef_funcs(area); out=[]
    for c in FILL:
        m=np.all(arr==np.array(c),axis=2); lbl,n=ndimage.label(m); objs=ndimage.find_objects(lbl)
        for k in range(1,n+1):
            sl=objs[k-1]; ys,xs=np.where(lbl[sl]==k)
            if not(amin<=len(xs)<=amax): continue
            ys=ys+sl[0].start; xs=xs+sl[1].start
            if max(ys.max()-ys.min()+1,xs.max()-xs.min()+1)>bbox_max: continue
            hx,hy=_axis_head(ys,xs); cx,cy=float(xs.mean()),float(ys.mean())
            out.append(dict(px=cx,py=cy,lat=float(p2lat(cy)),lon=float(p2lon(cx)),
                            speed=SPEED[c],dir=_bearing(hx,hy)))
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
