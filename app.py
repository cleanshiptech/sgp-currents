"""MPA Current Time-Series (online). Pick a point -> current through the day.
No disk cache: reads CI-precomputed JSON (repo 'data' branch via raw URL) and falls
back to in-memory live fetch. Speed = 6-band filled grid; direction = arrows."""
import os, math, json, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import altair as alt, folium
from streamlit_folium import st_folium
import extractor as ex, render as R, data as D, grid as G

st.set_page_config(page_title="MPA Current Time-Series", layout="wide")
STATIONS={"SSP-A":(1.1963,103.6815),"SSP-B":(1.1893,103.6934),"SSP-C":(1.1818,103.7032),
          "SSP-D":(1.1715,103.7134),"SSP-ADCP":(1.1571,103.7402),
          "EBA-A":(1.2870,104.0020),"EBA-B":(1.3000,104.0680)}
# Greyscale Positron first: neutral basemap so the speed colours read true.
TILES={"Carto Positron":("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
       "Carto Voyager":("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
       "Esri Ocean":("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}","Esri — Oceans (no tiles past z13)")}

@st.cache_data(show_spinner=False, ttl=3600)   # re-check repo hourly for fresh CI data
def get_precomputed(area,day,remote): return D.load_precomputed(area,day,remote_base=remote)

@st.cache_data(show_spinner=False)             # MPA designated anchorages (Port Regs, 2nd Schedule)
def load_anchorages():
    p=os.path.join(os.path.dirname(os.path.abspath(__file__)),"anchorages.json")
    try: return json.load(open(p))
    except Exception: return []
ANCH=load_anchorages()

def _overlap():                                # split lon so SSP/EBA tile without double-plotting
    bs=D.bounds("SSP"); be=D.bounds("EBA")
    lo=max(bs["sw"][1],be["sw"][1]); hi=min(bs["ne"][1],be["ne"][1])
    if lo>=hi: return None,None,None
    ssp_w=(bs["sw"][1]+bs["ne"][1])/2 < (be["sw"][1]+be["ne"][1])/2
    return (lo+hi)/2, ("W" if ssp_w else "E"), ("E" if ssp_w else "W")
SPLIT,SSP_KEEP,EBA_KEEP=_overlap()

def nearest_station(lat,lon):
    best=None
    for n,(la,lo) in STATIONS.items():
        d=math.hypot((la-lat)*110.6,(lo-lon)*111.3*math.cos(math.radians(lat)))
        if best is None or d<best[0]: best=(d,n)
    return best

# ---- TV kiosk mode: ?kiosk=1 -> chrome-free, auto-looping today→+6-day animation ----
_KIOSK_HTML = """<!doctype html><html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;background:#eef1f3;font-family:system-ui,Arial,sans-serif;overflow:hidden}
 #top{position:absolute;top:0;left:0;right:0;height:58px;z-index:1000;display:flex;align-items:center;
      gap:18px;padding:0 18px;background:rgba(22,50,79,.95);color:#fff;box-sizing:border-box}
 #clock{font-size:30px;font-weight:700;white-space:nowrap}
 #legend{flex:1;text-align:center;font-size:17px}
 #legend b{margin-right:8px}
 #legend span{padding:3px 10px;margin:2px;border-radius:4px;font-weight:600;display:inline-block}
 #title{font-size:16px;opacity:.85;white-space:nowrap}
 #map{position:absolute;top:58px;left:0;right:0;bottom:0}
 .anchlbl{background:none!important;border:0!important;box-shadow:none!important;color:#1b1c66;font-weight:700;font-size:10px;text-shadow:0 0 3px #fff,0 0 3px #fff}
 #anchleg{position:absolute;bottom:14px;right:14px;z-index:1000;background:rgba(255,255,255,.92);padding:8px 12px;border-radius:8px;font-size:11px;line-height:1.3;color:#1b1c66;max-height:88vh;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.18)}
 #anchleg b{display:inline-block;width:20px;text-align:right;margin-right:6px}
 #anchleg .h{font-weight:700;margin-bottom:3px;border-bottom:1px solid #ccd;padding-bottom:2px}
 #bar{position:absolute;bottom:0;left:0;height:5px;background:#2e8b57;z-index:1001;transition:width .25s linear}
</style></head><body>
<div id="top">
 <span id="clock">--</span>
 <span id="legend"><b>Speed (kn):</b> __LEGEND__</span>
 <span id="title">SGP Currents &middot; live 7-day loop</span>
</div>
<div id="map"></div>
<div id="bar"></div>
<div id="anchleg"><div class="h">Anchorages</div>__ANCHLEG__</div>
<script>
var D=__PAYLOAD__;
var BLANK='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
var map=L.map('map',{zoomControl:false,attributionControl:false,dragging:false,scrollWheelZoom:false,doubleClickZoom:false,boxZoom:false,keyboard:false,touchZoom:false,tap:false});
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{maxZoom:18}).addTo(map);
var ssp=D.sspb?L.imageOverlay(BLANK,[D.sspb.sw,D.sspb.ne],{opacity:.92}).addTo(map):null;
var eba=D.ebab?L.imageOverlay(BLANK,[D.ebab.sw,D.ebab.ne],{opacity:.92}).addTo(map):null;
(D.anch||[]).forEach(function(a,i){L.polygon(a.poly,{color:'#1b1c66',weight:1.2,fill:false,opacity:.55})
 .addTo(map).bindTooltip(String(i+1),{permanent:true,direction:'center',className:'anchlbl'});});
function fit(){map.invalidateSize();map.fitBounds([D.sw,D.ne],{padding:[28,28]});}
fit();setTimeout(fit,400);setTimeout(fit,1200);window.addEventListener('resize',fit);
var i=0,n=D.frames.length;
function show(){var f=D.frames[i];
 if(ssp)ssp.setUrl(f.ssp||BLANK);
 if(eba)eba.setUrl(f.eba||BLANK);
 document.getElementById('clock').textContent=f.label;
 document.getElementById('bar').style.width=((i+1)/n*100)+'%';
 i=(i+1)%n;}
show();setInterval(show,Math.round(1000/D.fps));
setTimeout(function(){location.reload();},3*3600*1000);
</script></body></html>"""

@st.cache_data(show_spinner="Building 7-day loop…", ttl=10800)   # cache 3h; JS reloads to roll the window
def _kiosk_frames(days_tuple, step, remote):
    out=[]; sspb=ebab=None
    for ds in days_tuple:
        dssp=get_precomputed("SSP",ds,remote); deba=get_precomputed("EBA",ds,remote)
        hs=(dssp or {}).get("frames",{}); he=(deba or {}).get("frames",{})
        if not hs and not he: continue
        if dssp: sspb=dssp["bounds"]
        if deba: ebab=deba["bounds"]
        for ti in sorted(set(hs)|set(he))[::step]:
            osp=R.gridded_frame_overlay(hs,ti) if ti in hs else None
            oeb=R.gridded_frame_overlay(he,ti) if ti in he else None
            if SPLIT and osp is not None: osp=R.clip_lon(osp,D.bounds("SSP"),SPLIT,SSP_KEEP)
            if SPLIT and oeb is not None: oeb=R.clip_lon(oeb,D.bounds("EBA"),SPLIT,EBA_KEEP)
            su=R.kiosk_uri(osp) if osp is not None else None
            eu=R.kiosk_uri(oeb) if oeb is not None else None
            if su or eu:
                out.append({"label":f"{dt.datetime.strptime(ds,'%Y%m%d'):%a %d %b}  {ti[:2]}:{ti[2:]}","ssp":su,"eba":eu})
    return out,sspb,ebab

def _render_kiosk(qp):
    from streamlit.components.v1 import html as _html
    days=int(qp.get("days",7)); step=int(qp.get("step",2)); fps=float(qp.get("fps",2)); H=int(qp.get("h",1040))
    remote=st.secrets.get("DATA_BASE","") if hasattr(st,"secrets") else ""
    today=(dt.datetime.utcnow()+dt.timedelta(hours=8)).date()
    days_tuple=tuple((today+dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(days))
    st.markdown("<style>#MainMenu,header,footer,[data-testid='stToolbar']{display:none!important}"
                ".block-container{padding:0!important;max-width:100%!important}"
                ".stApp{overflow:hidden}"
                "iframe{height:100vh!important;width:100%!important;display:block;border:0}</style>",
                unsafe_allow_html=True)   # force the map iframe to fill the actual TV viewport
    frames,sspb,ebab=_kiosk_frames(days_tuple,step,remote)
    if not frames: st.error("No data archived yet for the next 7 days."); return
    sws=[b["sw"] for b in (sspb,ebab) if b]; nes=[b["ne"] for b in (sspb,ebab) if b]
    sw=[min(s[0] for s in sws),min(s[1] for s in sws)]; ne=[max(n[0] for n in nes),max(n[1] for n in nes)]
    def _txt(rgb): return "#111" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2])>150 else "#fff"
    legend="".join(f"<span style='background:rgb{ex.DISPLAY_BANDS[i][1]};color:{_txt(ex.DISPLAY_BANDS[i][1])}'>{ex.DISPLAY_LABELS[i]}</span>" for i in range(6))
    payload=json.dumps({"frames":frames,"sspb":sspb,"ebab":ebab,"sw":sw,"ne":ne,"fps":fps,"anch":ANCH})
    anchleg="".join(f"<div><b>{i+1}</b>{a['name'].replace(' Anchorage','')}</div>" for i,a in enumerate(ANCH))
    _html(_KIOSK_HTML.replace("__PAYLOAD__",payload).replace("__LEGEND__",legend).replace("__ANCHLEG__",anchleg),
          height=H,scrolling=False)

def _qp():
    try: return {k:(v[-1] if isinstance(v,list) else v) for k,v in dict(st.query_params).items()}
    except Exception: return {k:(v[0] if isinstance(v,list) else v) for k,v in st.experimental_get_query_params().items()}

if str(_qp().get("kiosk",""))=="1":
    _render_kiosk(_qp()); st.stop()

# ---- Anchorage planner: ETA -> 48h per-anchorage current, ranked by hours below threshold ----
def _pip(poly, lat, lon):                      # vectorised ray-casting point-in-polygon
    vy=np.array([p[0] for p in poly]); vx=np.array([p[1] for p in poly]); n=len(poly)
    ins=np.zeros(len(lat),bool); j=n-1
    for i in range(n):
        c=((vy[i]>lat)!=(vy[j]>lat)) & (lon < (vx[j]-vx[i])*(lat-vy[i])/((vy[j]-vy[i])+1e-12)+vx[i])
        ins^=c; j=i
    return ins

@st.cache_data(show_spinner="Computing anchorage currents…", ttl=3600)
def plan_series(eta_iso, hours, aggmode, remote):
    """For each anchorage, the aggregated (mean/max) current at each 30-min step over the window.
    Returns (series{code:[kn|None]}, step_iso[], approx{code:True}). Cells inside each polygon;
    anchorages smaller than the grid fall back to the nearest cell (flagged approx)."""
    start=dt.datetime.fromisoformat(eta_iso)
    steps=[start+dt.timedelta(minutes=30*k) for k in range(int(hours)*2)]
    days=sorted({s.strftime("%Y%m%d") for s in steps})
    cent={a["code"]:(float(np.mean([p[0] for p in a["poly"]])),float(np.mean([p[1] for p in a["poly"]]))) for a in ANCH}
    grids={}
    for area in ("SSP","EBA"):
        for day in days:
            dd=D.load_compact(area,day,remote)
            if not dd or not dd.get("grid"): continue
            g=np.array(dd["grid"],float); p2lon,p2lat,_,_=ex.georef_funcs(area)
            clat=np.array([p2lat(y) for y in g[:,1]]); clon=np.array([p2lon(x) for x in g[:,0]])
            idx={a["code"]:np.where(_pip(a["poly"],clat,clon))[0].tolist() for a in ANCH}
            grids[(area,day)]=(dd,idx,clat,clon)
    if not grids: return {},[],{}
    approx={}
    for a in ANCH:                              # nearest-cell fallback for tiny anchorages
        c=a["code"]
        if any(grids[k][1][c] for k in grids): continue
        cy,cx=cent[c]; best=None
        for (area,day),(dd,idx,clat,clon) in grids.items():
            d2=(clat-cy)**2+(clon-cx)**2; mi=int(d2.argmin())
            if best is None or d2[mi]<best[0]: best=(d2[mi],area)
        if best:
            home=best[1]
            for (area,day),(dd,idx,clat,clon) in grids.items():
                if area==home: idx[c]=[int(((clat-cy)**2+(clon-cx)**2).argmin())]
            approx[c]=True
    series={a["code"]:[] for a in ANCH}
    for s in steps:
        day=s.strftime("%Y%m%d"); hhmm=s.strftime("%H%M")
        for code in series:
            sp=[]
            for area in ("SSP","EBA"):
                k=(area,day)
                if k not in grids: continue
                dd,idx,_,_=grids[k]; fr=dd["frames"].get(hhmm)
                if not fr: continue
                for ci in idx[code]:
                    b=fr["s"][ci]; sp.append(G.BAND_MIDS[b-1] if b>0 else 0.25)
            series[code].append(round(float(np.mean(sp) if aggmode=="mean" else max(sp)),3) if sp else None)
    return series,[s.isoformat() for s in steps],approx

def _render_planner(eta_d, eta_h, win, thr, aggmode, remote):
    st.subheader("⚓ Anchorage planner")
    if not ANCH: st.error("No anchorage data loaded."); return
    eta=dt.datetime.combine(eta_d,dt.time(int(eta_h)))
    am="max" if "worst" in aggmode else "mean"
    series,steps_iso,approx=plan_series(eta.isoformat(),int(win),am,remote)
    if not steps_iso or not any(any(v is not None for v in s) for s in series.values()):
        st.warning("No archived current data for that window yet. The forecast archive covers the "
                   "current month (and next once MPA publishes it) — try an ETA within that range."); return
    steps=[dt.datetime.fromisoformat(t) for t in steps_iso]; nmap={a["code"]:a["name"] for a in ANCH}
    rows=[]
    for code,s in series.items():
        av=[v for v in s if v is not None]
        if not av: continue
        hrs=0.5*sum(1 for v in av if v<thr); mean=sum(av)/len(av)
        rows.append((hrs,mean,max(av),code))
    rows.sort(key=lambda r:(-r[0],r[1],r[2]))   # most hours below thr, then calmest mean, then lowest peak
    allcalm=rows and all(abs(h-win)<1e-6 for h,*_ in rows)
    st.caption(f"ETA **{eta:%a %d %b %Y %H:%M}** SGT → **{win} h**. Ranked by hours the **{aggmode}** current "
               f"stays below **{thr:.1f} kn**, ties broken by lowest mean.  ~ = nearest-cell estimate (anchorage < grid).")
    if rows:
        h0,mn0,pk0,c0=rows[0]
        if allcalm:
            st.info(f"🟢 **Calm window** — every anchorage stays below {thr:.1f} kn for the whole {win} h "
                    f"(it's a neap/slack period). Ranked by lowest mean current; calmest: **{nmap[c0]}** "
                    f"({c0}) — mean {mn0:.2f} kn, peak {pk0:.2f} kn.")
        else:
            st.success(f"Most attractive: **{nmap[c0]}** ({c0}) — **{h0:.1f} h** ({h0/win:.0%}) below "
                       f"{thr:.1f} kn, mean {mn0:.2f}, peak {pk0:.2f} kn.")
    df=pd.DataFrame([{"#":i+1,"Anchorage":nmap[c]+(" ~" if approx.get(c) else ""),"Code":c,
                      f"Hrs <{thr:.1f}":round(h,1),"% win":round(100*h/win),"Mean kn":round(mn,2),"Peak kn":round(pk,2)}
                     for i,(h,mn,pk,c) in enumerate(rows)])
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.download_button("Download ranking (CSV)",df.to_csv(index=False).encode(),
                       f"anchorage_plan_{eta:%Y%m%d_%H%M}.csv","text/csv")
    long=[{"Anchorage":nmap[c],"t":t,"kn":v} for c,s in series.items() for t,v in zip(steps,s) if v is not None]
    if long:
        L=pd.DataFrame(long); order=[nmap[c] for *_,c in rows]
        hi=max(thr*1.5, math.ceil(L["kn"].max()))   # adapt upper bound so calm days still show variation
        hm=alt.Chart(L).mark_rect().encode(
            x=alt.X("t:T",title="SGT (window from ETA)"),
            y=alt.Y("Anchorage:N",sort=order,title=None),
            color=alt.Color("kn:Q",title="kn",scale=alt.Scale(domain=[0,thr,hi],range=["#2c7bb6","#ffffbf","#d7191c"])),
            tooltip=[alt.Tooltip("Anchorage:N"),alt.Tooltip("t:T",format="%a %H:%M"),alt.Tooltip("kn:Q")]
        ).properties(height=max(220,18*len(rows)))
        st.caption(f"Blue = calm, yellow ≈ {thr:.1f} kn, red = above. Each row is an anchorage; left→right is the window from ETA.")
        st.altair_chart(hm,use_container_width=True)

with st.sidebar:
    st.header("SGP Currents")
    sgt_today=(dt.datetime.utcnow()+dt.timedelta(hours=8)).date()
    remote=st.secrets.get("DATA_BASE","") if hasattr(st,"secrets") else ""    # config, not UI
    xlsx=st.secrets.get("MPA_XLSX","") if hasattr(st,"secrets") else ""
    mode=st.radio("Mode",["Live map","Anchorage planner"]) if ANCH else "Live map"
    st.markdown("---")
    if mode=="Anchorage planner":
        eta_d=st.date_input("Vessel ETA (date)",sgt_today,
                            min_value=sgt_today-dt.timedelta(days=7),
                            max_value=sgt_today+dt.timedelta(days=55))
        eta_h=st.slider("ETA hour (SGT)",0,23,0)
        win=st.select_slider("Window (hours)",[24,48,72],value=48)
        thr=st.slider("Calm threshold (kn)",0.5,3.0,1.5,0.5)
        aggmode=st.radio("Per-anchorage current",["typical (mean)","worst-case (max)"])
    else:
        area="SSP + EBA"; sel_areas=["SSP","EBA"]   # combined view only
        date=st.date_input("Date",sgt_today,
                           min_value=sgt_today-dt.timedelta(days=40),
                           max_value=sgt_today+dt.timedelta(days=62))
        day=date.strftime("%Y%m%d")
        loaded=[(a,dd) for a in sel_areas if (dd:=get_precomputed(a,day,remote)) and dd.get("frames")]
        if loaded:
            st.caption(f"✓ {date:%a %d %b %Y} · SSP + EBA")
        else:
            st.warning(f"No data for **{date:%d %b %Y}** yet. The archive covers the current month "
                       f"(it updates automatically) — try a date in {sgt_today:%B}.")
        if st.button("↻ Refresh"): get_precomputed.clear(); st.rerun()
        st.markdown("---")
        basemap=st.selectbox("Basemap",list(TILES))
        view=st.radio("Map layer",["Snapshot (time)","Aggregate: max","Aggregate: mean"])
        opacity=st.slider("Overlay opacity",0.3,1.0,0.85)
        show_anch=st.checkbox(f"Anchorage areas ({len(ANCH)})",value=True) if ANCH else False

if mode=="Anchorage planner":
    _render_planner(eta_d,eta_h,win,thr,aggmode,remote); st.stop()

if not loaded:
    st.title("MPA Current Time-Series")
    st.info(f"No archived data for {date:%d %b %Y}. Pick a date in the current month — the "
            "archive fills automatically and extends into next month once MPA publishes it."); st.stop()

# combined bounds + union of frame times across the selected area(s)
allsw=[dd["bounds"]["sw"] for _,dd in loaded]; allne=[dd["bounds"]["ne"] for _,dd in loaded]
sw=[min(s[0] for s in allsw),min(s[1] for s in allsw)]; ne=[max(n[0] for n in allne),max(n[1] for n in allne)]
times=sorted(set().union(*[set(dd["frames"]) for _,dd in loaded]))
tcount=lambda ti: sum(len(dd["frames"].get(ti,[])) for _,dd in loaded)     # arrows across all areas
daymax=max((tcount(t) for t in times),default=0)

hcol,bcol=st.columns([5,1])
hcol.subheader(f"{area} · {date:%d %b %Y}")
hcol.caption("precomputed · click the map to drop a worksite")
if "pt" in st.session_state and bcol.button("Clear pin"): del st.session_state["pt"]; st.rerun()

left,right=st.columns([3,2])
with left:
    if view=="Snapshot (time)":
        ti=st.select_slider("Time (SGT)",times,value=times[len(times)//2])
        overlays=[(a,R.gridded_frame_overlay(dd["frames"],ti),dd["bounds"]) for a,dd in loaded if ti in dd["frames"]]
        cap=f"{area} {day} {ti} · {tcount(ti)} arrows"
        if daymax and tcount(ti)<0.25*daymax:
            st.info("⚓ **Slack water** — currents near zero, so most cells are calm (≤0.5 kn) with no direction. Slide to a flood/ebb time for the working current.")
    else:
        stat="max" if "max" in view else "mean"
        overlays=[(a,R.aggregate_overlay(dd["frames"],stat),dd["bounds"]) for a,dd in loaded]
        cap=f"{area} {day} · {stat} over day"
    url,attr=TILES[basemap]
    m=folium.Map(location=[(sw[0]+ne[0])/2,(sw[1]+ne[1])/2],zoom_start=11 if len(loaded)>1 else 12,tiles=url,attr=attr)
    for a,ov,bnd in overlays:
        if SPLIT: ov=R.clip_lon(ov,bnd,SPLIT, SSP_KEEP if a=="SSP" else EBA_KEEP)
        folium.raster_layers.ImageOverlay(R.png_data_uri(ov),bounds=[bnd["sw"],bnd["ne"]],opacity=opacity).add_to(m)
    if show_anch:
        for a in ANCH:
            folium.Polygon([tuple(p) for p in a["poly"]],color="#2b2c7c",weight=1.4,
                           fill=True,fill_color="#3b3bbf",fill_opacity=0.05,
                           tooltip=f"{a['name']} · {a['code']}").add_to(m)
    nodes=[n for _,dd in loaded for n in R.canonical_nodes(dd["frames"])]   # fit to the data extent (closer)
    if nodes:
        la=[n[2] for n in nodes]; lo2=[n[3] for n in nodes]; pad=0.012
        m.fit_bounds([[min(la)-pad,min(lo2)-pad],[max(la)+pad,max(lo2)+pad]])
    elif len(loaded)>1: m.fit_bounds([sw,ne])
    if "pt" in st.session_state: folium.Marker(st.session_state["pt"]).add_to(m)
    st.caption(cap)
    ev=st_folium(m,height=470,use_container_width=True,returned_objects=["last_clicked"])
    if ev and ev.get("last_clicked"): st.session_state["pt"]=[ev["last_clicked"]["lat"],ev["last_clicked"]["lng"]]
    # 6-band legend (dark text on light swatches, light on dark — stays readable)
    def _txt(rgb): return "#111" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2])>150 else "#fff"
    def _chip(rgb,lab): return f"<span style='background:rgb{rgb};padding:2px 8px;margin:2px;color:{_txt(rgb)};border:1px solid #cbd2d9;border-radius:3px'>{lab}</span>"
    chips="".join(_chip(ex.DISPLAY_BANDS[i][1],ex.DISPLAY_LABELS[i]) for i in range(6))
    st.markdown("**Speed (kn):** "+chips,unsafe_allow_html=True)
    st.caption("Cells without an arrow are **≤0.5 kn (calm)** — too weak to resolve a direction. "
               "All currents ≥0.5 kn are detected, so a calm cell isn't hiding stronger flow.")

with right:
    st.subheader("Current at point")
    if "pt" not in st.session_state: st.info("Click the map to choose a worksite."); st.stop()
    plat,plon=st.session_state["pt"]; st.caption(f"{plat:.4f} N, {plon:.4f} E")
    # pick whichever loaded area has the nearest arrow to the point
    best=None
    for a,dd in loaded:
        s=R.point_series(dd["frames"],plat,plon,max_km=1.0)
        if s:
            mind=min(r[3] for r in s)
            if best is None or mind<best[0]: best=(mind,a,s)
    if best is None: st.warning("No arrows within 1 km (land or outside field)."); st.stop()
    _,parea,ser=best
    if len(loaded)>1: st.caption(f"area: {parea}")
    df=pd.DataFrame(ser,columns=["time","speed","dir","dist_km"])
    df["t"]=pd.to_datetime(df["time"],format="%H%M").dt.strftime("%H:%M")
    c1,c2,c3=st.columns(3)
    c1.metric("Max (binned)",f"{df.speed.max():.1f} kn"); c2.metric("Mean",f"{df.speed.mean():.1f} kn")
    c3.metric(f"% ≥ {ex.ROV_LIMIT}",f"{100*(df.speed>=ex.ROV_LIMIT).mean():.0f}%")
    bandrows=pd.DataFrame({"y0":[0,.5,1,1.5,2,2.5],"y1":[.5,1,1.5,2,2.5,3.2],
        "c":[f"rgb{ex.DISPLAY_BANDS[i][1]}" for i in range(6)]})
    bg=alt.Chart(bandrows).mark_rect(opacity=.18).encode(y="y0:Q",y2="y1:Q",
        color=alt.Color("c:N",scale=None,legend=None))
    base=alt.Chart(df).encode(x=alt.X("t:N",title="SGT"))
    err=base.mark_area(opacity=.25,color="#333").encode(y=alt.Y("lo:Q",title="speed (kn)"),y2="hi:Q") \
        .transform_calculate(lo="datum.speed-0.25",hi="datum.speed+0.25")
    line=base.mark_line(point=True,color="#111").encode(y="speed:Q")
    limit=alt.Chart(pd.DataFrame({"y":[ex.ROV_LIMIT]})).mark_rule(strokeDash=[5,4],color="red").encode(y="y:Q")
    st.altair_chart((bg+err+line+limit).properties(height=240),use_container_width=True)
    st.altair_chart(alt.Chart(df).mark_line(point=True,color="#555").encode(
        x=alt.X("t:N",title="SGT"),y=alt.Y("dir:Q",title="set °T",scale=alt.Scale(domain=[0,360]))
        ).properties(height=140),use_container_width=True)
    dkm,stn=nearest_station(plat,plon)
    if dkm<0.4 and os.path.isfile(xlsx):
        try:
            xa=pd.read_excel(xlsx,sheet_name="Current Data"); d=dt.datetime.strptime(day,"%Y%m%d")
            es=xa[(xa.Area==parea)&(xa.Month==d.month)&(xa.Day==d.day)&(xa.Station==stn.split("-")[1])]
            if not es.empty:
                es=es.assign(t=es.Hour_SGT.map(lambda h:f"{int(h):02d}:00"))
                st.caption(f"dashed = exact MPA value at {stn} ({dkm*1000:.0f} m)")
                st.altair_chart((line+alt.Chart(es).mark_line(strokeDash=[4,3],color="black").encode(
                    x="t:N",y="Speed_knots:Q")).properties(height=200),use_container_width=True)
        except Exception: pass
    st.caption("Speed colour/bin-derived (±0.25 kn band). Red dashed = ROV limit.")
