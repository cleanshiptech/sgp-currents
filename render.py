"""Render speed-grid + direction-icon overlays (transparent PNG, frame pixel space),
plus date aggregates and point time-series. Glyphs are supersampled for clean edges."""
import math, io, base64
import numpy as np
from PIL import Image, ImageDraw
import extractor as ex

W,H=1673,896; CELL=17; SS=3   # half-cell px; supersample factor for anti-aliasing
# arrow glyph pointing +x: triangular head + rectangular shaft (cleaner than a flat chevron)
DART=[(10,0),(2,-6),(2,-2.4),(-9,-2.4),(-9,2.4),(2,2.4),(2,6)]
CALM=ex.NO_READING+(70,)       # faint grey scaffold for cells with no recovered arrow

def _rot(pts,ang,cx,cy,s):
    c,si=math.cos(ang),math.sin(ang)
    return [((cx+(x*c-y*si)*s), (cy+(x*si+y*c)*s)) for x,y in pts]

def _overlay(items, cell_alpha=140, icon=(28,31,38)):
    """items: (px,py,speed,deg). speed=None -> faint grey 'no reading' cell (no dart);
    any real speed (incl. 0-0.5) -> band-coloured fill plus a haloed direction arrow."""
    big=Image.new("RGBA",(W*SS,H*SS),(0,0,0,0)); d=ImageDraw.Draw(big)
    cell=lambda x,y,f: d.rectangle([x-CELL*SS,y-CELL*SS,x+CELL*SS,y+CELL*SS],fill=f)
    for px,py,speed,deg in items:                 # no-reading scaffold first
        if speed is None: cell(px*SS,py*SS,CALM)
    for px,py,speed,deg in items:                 # measured cells on top
        if speed is not None: cell(px*SS,py*SS,ex.band_color(speed)+(cell_alpha,))
    for px,py,speed,deg in items:                 # darts for every measured cell
        if speed is None: continue
        x,y=px*SS,py*SS
        ang=math.atan2(-math.cos(math.radians(deg)),math.sin(math.radians(deg)))  # heading in image coords
        d.polygon(_rot([(p[0]*1.3,p[1]*1.3) for p in DART],ang,x,y,SS),fill=(255,255,255,235))
        d.polygon(_rot(DART,ang,x,y,SS),fill=icon+(255,))
    return big.resize((W,H),Image.LANCZOS)

def frame_overlay(arrows):
    return _overlay([(a["px"],a["py"],a["speed"],a["dir"]) for a in arrows])

def gridded_frame_overlay(frames, ti, max_px=14):
    """Snapshot over the *full* canonical grid: coloured cell+dart where a current is read
    at time ti, faint grey 'no reading' cell elsewhere — so the survey field is always
    visible while blanks stay honestly unknown (not painted 'calm')."""
    nodes=canonical_nodes(frames); cur=frames.get(ti,[])
    if cur:
        cx=np.array([a["px"] for a in cur]); cy=np.array([a["py"] for a in cur])
    items=[]
    for px,py,lat,lon in nodes:
        if cur:
            dd=(cx-px)**2+(cy-py)**2; j=int(dd.argmin())
            if dd[j]<=max_px*max_px:
                a=cur[j]; items.append((px,py,a["speed"],a["dir"])); continue
        items.append((px,py,None,0.0))   # no reading -> grey cell, no dart
    return _overlay(items)

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
