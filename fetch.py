"""Fetch MPA tidal-atlas current frames. In-memory (for the live app) and to-disk
(for the CI data builder)."""
import requests
BASE="https://digitalport-service.mpa.gov.sg/ta/ImageServlet"
SLOTS=[f"{h:02d}{m:02d}" for h in range(24) for m in (0,30)]
def frame_id(area,day,hhmm): return f"TA_{area}{day}{hhmm}.gif"
def fetch_bytes(area, day, hhmm, session=None):
    """Return GIF bytes, or None if MPA has no frame for that slot."""
    s=session or requests
    r=s.get(BASE,params={"id":frame_id(area,day,hhmm)},headers={"Accept":"image/gif"},timeout=30)
    if r.content[:3]!=b"GIF":
        if r.content[:3]==b"<!D" or r.content[:5]==b"<html":
            raise RuntimeError("Got disclaimer HTML, not a GIF (header/session issue).")
        return None
    return r.content
