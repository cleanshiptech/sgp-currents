"""MPA Current Time-Series (online). Pick a point -> current through the day.
No disk cache: reads CI-precomputed JSON (repo 'data' branch via raw URL) and falls
back to in-memory live fetch. Speed = 6-band filled grid; direction = arrows."""
import os, math, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import altair as alt, folium
from streamlit_folium import st_folium
import extractor as ex, render as R, data as D

st.set_page_config(page_title="MPA Current Time-Series", layout="wide")
STATIONS={"SSP-A":(1.1963,103.6815),"SSP-B":(1.1893,103.6934),"SSP-C":(1.1818,103.7032),
          "SSP-D":(1.1715,103.7134),"SSP-ADCP":(1.1571,103.7402),
          "EBA-A":(1.2870,104.0020),"EBA-B":(1.3000,104.0680)}
TILES={"Esri Ocean":("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}","Esri — Oceans"),
       "Carto Positron":("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
       "Carto Voyager":("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO")}

@st.cache_data(show_spinner=False, ttl=3600)   # re-check repo hourly for fresh CI data
def get_precomputed(area,day,remote): return D.load_precomputed(area,day,remote_base=remote)
@st.cache_data(show_spinner="Fetching + digitising frames…")
def get_live(area,day):
    try:
        return D.build_live(area,day)
    except Exception as e:
        st.session_state["live_error"]=str(e); return None

def nearest_station(lat,lon):
    best=None
    for n,(la,lo) in STATIONS.items():
        d=math.hypot((la-lat)*110.6,(lo-lon)*111.3*math.cos(math.radians(lat)))
        if best is None or d<best[0]: best=(d,n)
    return best

with st.sidebar:
    st.header("Data")
    area=st.selectbox("Area",["SSP","EBA"])
    sgt_today=(dt.datetime.utcnow()+dt.timedelta(hours=8)).date()
    date=st.date_input("Date",sgt_today); day=date.strftime("%Y%m%d")
    remote=st.secrets.get("DATA_BASE","") if hasattr(st,"secrets") else ""
    remote=st.text_input("Precomputed data base URL (optional)",remote)
    xlsx=st.text_input("MPA spreadsheet (optional)","MPA_TidalCurrent_2026_Jan-Jun_compressed.xlsx")
    if area not in ex.GEOREF:
        st.warning(f"{area} needs georeference calibration — upload one {area} frame to add it."); st.stop()
    day_data=get_precomputed(area,day,remote)
    src="precomputed" if day_data else None
    if not day_data:
        st.info("No precomputed data for this date.")
        if st.button(f"Build live: {area} {day}"):
            day_data=get_live(area,day); src="live (in-memory)"
            if not day_data:
                st.error(f"Live build failed: {st.session_state.get('live_error','no frames for this date (future or unavailable).')}")
    if day_data: st.caption(f"source: {src} · {len(day_data['frames'])}/48 frames")
    st.markdown("---")
    basemap=st.selectbox("Basemap",list(TILES))
    view=st.radio("Map layer",["Snapshot (time)","Aggregate: max","Aggregate: mean"])
    opacity=st.slider("Overlay opacity",0.3,1.0,0.85)

if not day_data or not day_data["frames"]:
    st.title("MPA Current Time-Series")
    st.info("Choose area/date. If not precomputed, click **Build live** in the sidebar."); st.stop()
frames=day_data["frames"]; sw=day_data["bounds"]["sw"]; ne=day_data["bounds"]["ne"]; times=sorted(frames)

hcol,bcol=st.columns([5,1])
hcol.subheader(f"{area} · {date:%d %b %Y}")
hcol.caption(f"{src} · {len(frames)}/48 frames · click the map to drop a worksite")
if "pt" in st.session_state and bcol.button("Clear pin"): del st.session_state["pt"]; st.rerun()

left,right=st.columns([3,2])
with left:
    if view=="Snapshot (time)":
        ti=st.select_slider("Time (SGT)",times,value=times[len(times)//2]); overlay=R.frame_overlay(frames[ti]); cap=f"{area} {day} {ti}"
    else:
        stat="max" if "max" in view else "mean"; overlay=R.aggregate_overlay(frames,stat); cap=f"{area} {day} · {stat} over day"
    url,attr=TILES[basemap]
    m=folium.Map(location=[(sw[0]+ne[0])/2,(sw[1]+ne[1])/2],zoom_start=12,tiles=url,attr=attr)
    folium.raster_layers.ImageOverlay(R.png_data_uri(overlay),bounds=[sw,ne],opacity=opacity).add_to(m)
    if "pt" in st.session_state: folium.Marker(st.session_state["pt"]).add_to(m)
    st.caption(cap)
    ev=st_folium(m,height=470,use_container_width=True,returned_objects=["last_clicked"])
    if ev and ev.get("last_clicked"): st.session_state["pt"]=[ev["last_clicked"]["lat"],ev["last_clicked"]["lng"]]
    # 6-band legend
    chips="".join(f"<span style='background:rgb{ex.DISPLAY_BANDS[i][1]};padding:2px 8px;margin:2px;color:#fff;border-radius:3px'>{ex.DISPLAY_LABELS[i]}</span>" for i in range(6))
    st.markdown("**Speed (kn):** "+chips,unsafe_allow_html=True)

with right:
    st.subheader("Current at point")
    if "pt" not in st.session_state: st.info("Click the map to choose a worksite."); st.stop()
    plat,plon=st.session_state["pt"]; st.caption(f"{plat:.4f} N, {plon:.4f} E")
    ser=R.point_series(frames,plat,plon,max_km=1.0)
    if not ser: st.warning("No arrows within 1 km (land or outside field)."); st.stop()
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
            es=xa[(xa.Area==area)&(xa.Month==d.month)&(xa.Day==d.day)&(xa.Station==stn.split("-")[1])]
            if not es.empty:
                es=es.assign(t=es.Hour_SGT.map(lambda h:f"{int(h):02d}:00"))
                st.caption(f"dashed = exact MPA value at {stn} ({dkm*1000:.0f} m)")
                st.altair_chart((line+alt.Chart(es).mark_line(strokeDash=[4,3],color="black").encode(
                    x="t:N",y="Speed_knots:Q")).properties(height=200),use_container_width=True)
        except Exception: pass
    st.caption("Speed colour/bin-derived (±0.25 kn band). Red dashed = ROV limit.")
