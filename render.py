"""Render speed-grid + direction-icon overlays (transparent PNG, frame pixel space),
plus date aggregates and point time-series. Glyphs are supersampled for clean edges."""
import math, io, base64
import numpy as np
from PIL import Image, ImageDraw
import extractor as ex

W,H=1673,896; CELL=17; SS=3   # half-cell px; supersample factor for anti-aliasing
DART=[(10,0),(-6,7),(-1,0),(-6,-7)]   # unit dart pointing +x (tip, wing, notch, wing)

def _rot(pts,ang,cx,cy,s):
    c,si=math.cos(ang),math.sin(ang)
    return [((cx+(x*c-y*si)*s), (cy+(x*si+y*c)*s)) for x,y in pts]

def _overlay(items, cell_alpha=120, icon=(35,38,45)):
    big=Image.new("RGBA",(W*SS,H*SS),(0,0,0,0)); d=ImageDraw.Draw(big)
    g=SS                                   # cell pass
    for px,py,speed,deg in items:
        c=ex.band_color(speed); x,y=px*SS,py*SS
        d.rectangle([x-CELL*SS, y-CELL*SS, x+CELL*SS, y+CELL*SS], fill=c+(cell_alpha,))
    for px,py,speed,deg in items:           # icon pass (on top)
        x,y=px*SS,py*SS
        ang=math.atan2(-math.cos(math.radians(deg)),math.sin(math.radians(deg)))  # heading in image coords
        halo=_rot([(p[0]*1.45,p[1]*1.45) for p in DART],ang,x,y,SS)
        dart=_rot(DART,ang,x,y,SS)
        d.polygon(halo,fill=(255,255,255,235))
        d.polygon(dart,fill=icon+(255,))
    return big.resize((W,H),Image.LANCZOS)

def frame_overlay(arrows):
    return _overlay([(a["px"],a["py"],a["speed"],a["dir"]) for a in arrows])

def canonical_nodes(frames):
    dense=max(frames.values(),key=len)
    return [(a["px"],a["py"],a["lat"],a["lon"]) for a in dense]

def aggregate(frames, stat="max", spacing=35):
    half=spacing*0.6; out=[]
    for px,py,lat,lon in canonical_nodes(frames):
        sp=[]; dr=[]
        for arrows in frames.values():
            best=None
            for a in arrows:
                dd=(a["px"]-px)**2+(a["py"]-py)**2
                if best is None or dd<best[0]: best=(dd,a)
            if best and best[0]<=half*half: sp.append(best[1]["speed"]); dr.append(best[1]["dir"])
        if not sp: continue
        sp=np.array(sp)
        if stat=="max":
            i=int(sp.argmax()); val=float(sp[i]); ddeg=dr[i]
        else:
            val=float(sp.mean()); ang=np.deg2rad(dr)
            ddeg=math.degrees(math.atan2(np.sin(ang).mean(),np.cos(ang).mean()))%360
        out.append(dict(px=px,py=py,lat=lat,lon=lon,speed=val,dir=ddeg,n=len(sp)))
    return out

def aggregate_overlay(frames, stat="max"):
    return _overlay([(n["px"],n["py"],n["speed"],n["dir"]) for n in aggregate(frames,stat)])

def point_series(frames, lat, lon, max_km=1.0):
    rows=[]
    for hhmm in sorted(frames):
        r=ex.query_nearest(frames[hhmm],lat,lon,max_km=max_km)
        if r: rows.append((hhmm,r["speed"],r["dir"],r["dist_km"]))
    return rows

def png_data_uri(im):
    b=io.BytesIO(); im.save(b,"PNG"); return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
