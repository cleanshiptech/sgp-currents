"""Data layer for the online app:
- load_precomputed(): read read-only JSON committed by CI (local 'data/' or raw GitHub URL).
- build_live(): fetch+extract a date entirely in memory (no disk) as a fallback.
Frame JSON schema: {area, date, bounds:{sw,ne}, frames:{HHMM:[{lat,lon,px,py,speed,dir}]}, missing:[]}
"""
import os, json, io, requests
import numpy as np
from PIL import Image
import extractor as ex, fetch as F, grid as G

W,H=1673,896   # MPA frame pixel dimensions
def bounds(area):
    p2lon,p2lat,_,_=ex.georef_funcs(area)
    return dict(sw=[p2lat(H-1),p2lon(0)], ne=[p2lat(0),p2lon(W-1)])

def build(area, day, fetcher=F.fetch_bytes, progress=None):
    """Grid-anchored build (v2). Fetch the day's frames, recover a fixed cell grid from the
    arrow layer (temporal differencing removes static chart clutter), then read each cell's
    band-colour + heading every frame. De-cluttered -> recovers blue (2.0-2.5) and
    white-on-white (0-0.5); stable grid -> jitter-free.

    Compact schema (grid stored ONCE, not per frame):
      {area, date, bounds, grid:[[px,py]...], frames:{HHMM:{s:[band-idx/cell], d:[dir/cell]}}}
      band-idx: 0 = no reading, else index into grid.BAND_MIDS + 1; lat/lon derived on load.
    """
    s=requests.Session(); raw={}
    for i,hhmm in enumerate(F.SLOTS):
        if progress: progress((i+1)/len(F.SLOTS), f"{area} {day} {hhmm}")
        b=fetcher(area,day,hhmm,s)
        if b is not None: raw[hhmm]=np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    slots=sorted(raw); missing=[h for h in F.SLOTS if h not in raw]
    if not slots:
        return dict(area=area,date=day,bounds=bounds(area),grid=[],frames={},missing=missing)
    stack=np.stack([raw[h] for h in slots])
    cm=G.change_mask(stack)
    per=[ex.extract_image(raw[h],area) for h in slots]   # validated detector, once per frame
    gridc=G.build_grid(stack,cm,area,per=per)            # reused for grid union AND direction
    glist=[[int(round(px)),int(round(py))] for px,py in gridc]; N=len(gridc)
    b2i={round(m,2):i+1 for i,m in enumerate(G.BAND_MIDS)}
    frames={}
    for i,h in enumerate(slots):
        sp=[0]*N; di=[0]*N
        for ci,speed,direc in G.extract_frame(raw[h],cm,gridc,area,ref=per[i]):
            sp[ci]=b2i.get(round(speed,2),0); di[ci]=int(round(direc))%360
        frames[h]={"s":sp,"d":di}
    _despike(frames, slots, N)   # drop isolated high-speed spikes (peak-tide chart annotations)
    return dict(area=area,date=day,bounds=bounds(area),grid=glist,frames=frames,missing=missing)

def _despike(frames, slots, N, hi=6, jump=3):
    """The fixed grid gives each cell a time-series. Real fast current ramps smoothly across
    the 30-min frames; a chart annotation (e.g. a peak-tide direction arrow that shares the
    fast-band colour) shows up as an isolated high-speed spike at a normally-calm cell. Drop
    any high band (>=`hi`, ~>=2.75 kn) that exceeds BOTH temporal neighbours by >=`jump` bands."""
    S=np.array([frames[h]["s"] for h in slots])          # (T, N) band indices
    T=len(slots)
    for t in range(T):
        lo=S[t-1] if t>0 else np.zeros(N,int); up=S[t+1] if t<T-1 else np.zeros(N,int)
        spike=(S[t]>=hi) & ((S[t]-np.maximum(lo,up))>=jump)
        for ci in np.nonzero(spike)[0]:
            frames[slots[t]]["s"][ci]=0; frames[slots[t]]["d"][ci]=0

# ---- app-side loading ----
def _expand(dd, area):
    """Compact schema -> the {frames:{HHMM:[arrow dicts]}} the app/render expect."""
    grid=dd["grid"]; p2lon,p2lat,_,_=ex.georef_funcs(area)
    lat=[round(p2lat(py),5) for px,py in grid]; lon=[round(p2lon(px),5) for px,py in grid]
    frames={}
    for h,fr in dd["frames"].items():
        s=fr["s"]; d=fr["d"]; arr=[]
        for i,b in enumerate(s):
            if b:
                px,py=grid[i]
                arr.append({"px":px,"py":py,"speed":G.BAND_MIDS[b-1],"dir":d[i],"lat":lat[i],"lon":lon[i]})
        frames[h]=arr
    return dict(area=dd["area"],date=dd["date"],bounds=dd["bounds"],frames=frames,missing=dd.get("missing",[]))

def load_precomputed(area, day, remote_base=None, local_dir="data"):
    fn=f"{area}_{day}.json"; dd=None
    p=os.path.join(local_dir,fn)
    if os.path.isfile(p):
        dd=json.load(open(p))
    elif remote_base:
        try:
            r=requests.get(remote_base.rstrip("/")+"/"+fn,timeout=20)
            if r.status_code==200 and r.text.strip().startswith("{"): dd=r.json()
        except Exception: dd=None
    if dd is None: return None
    return _expand(dd,area) if "grid" in dd else dd   # expand compact; pass through legacy

def build_live(area, day, progress=None):
    return build(area, day, progress=progress)
