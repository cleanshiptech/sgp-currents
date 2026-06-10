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

def _arrows_compact(arrows):
    # compact JSON: px/py & heading as ints, lat/lon to 4dp (~11m), speed to 0.1kn
    return [{"lat":round(a["lat"],4),"lon":round(a["lon"],4),"px":int(round(a["px"])),
             "py":int(round(a["py"])),"speed":round(a["speed"],1),"dir":int(round(a["dir"]))%360}
            for a in arrows]

def build(area, day, fetcher=F.fetch_bytes, progress=None):
    """Grid-anchored build (v2): fetch the day's frames, recover a fixed cell grid from the
    arrow layer (temporal differencing removes static chart clutter), then read each cell's
    band-colour + heading every frame. Stable grid -> jitter-free; de-cluttered -> recovers
    blue (2.0-2.5) and white-on-white (0-0.5)."""
    s=requests.Session(); raw={}
    for i,hhmm in enumerate(F.SLOTS):
        if progress: progress((i+1)/len(F.SLOTS), f"{area} {day} {hhmm}")
        b=fetcher(area,day,hhmm,s)
        if b is not None: raw[hhmm]=np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    slots=sorted(raw); missing=[h for h in F.SLOTS if h not in raw]
    if not slots:
        return dict(area=area,date=day,bounds=bounds(area),frames={},missing=missing)
    stack=np.stack([raw[h] for h in slots])
    cm=G.change_mask(stack)
    per=[ex.extract_image(raw[h],area) for h in slots]   # validated detector, once per frame
    gridc=G.build_grid(stack,cm,area,per=per)             # reused for the grid union
    frames={h:_arrows_compact(G.extract_frame(raw[h],cm,gridc,area,ref=per[i]))  # ...and direction
            for i,h in enumerate(slots)}
    return dict(area=area,date=day,bounds=bounds(area),frames=frames,missing=missing)

# ---- app-side loading ----
def load_precomputed(area, day, remote_base=None, local_dir="data"):
    fn=f"{area}_{day}.json"
    p=os.path.join(local_dir,fn)
    if os.path.isfile(p):
        return json.load(open(p))
    if remote_base:
        try:
            r=requests.get(remote_base.rstrip("/")+"/"+fn,timeout=20)
            if r.status_code==200 and r.text.strip().startswith("{"): return r.json()
        except Exception: return None
    return None

def build_live(area, day, progress=None):
    return build(area, day, progress=progress)
