"""Data layer for the online app:
- load_precomputed(): read read-only JSON committed by CI (local 'data/' or raw GitHub URL).
- build_live(): fetch+extract a date entirely in memory (no disk) as a fallback.
Frame JSON schema: {area, date, bounds:{sw,ne}, frames:{HHMM:[{lat,lon,px,py,speed,dir}]}, missing:[]}
"""
import os, json, requests
import extractor as ex, fetch as F

W,H=1673,896   # MPA frame pixel dimensions
def bounds(area):
    p2lon,p2lat,_,_=ex.georef_funcs(area)
    return dict(sw=[p2lat(H-1),p2lon(0)], ne=[p2lat(0),p2lon(W-1)])

def _arrows_compact(arrows):
    return [{k:round(a[k],5) if k in ("lat","lon") else round(a[k],1) for k in ("lat","lon","px","py","speed","dir")} for a in arrows]

def build(area, day, fetcher=F.fetch_bytes, progress=None):
    s=requests.Session(); frames={}; missing=[]
    for i,hhmm in enumerate(F.SLOTS):
        if progress: progress((i+1)/len(F.SLOTS), f"{area} {day} {hhmm}")
        b=fetcher(area,day,hhmm,s)
        if b is None: missing.append(hhmm); continue
        frames[hhmm]=_arrows_compact(ex.extract_bytes(b,area))
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
