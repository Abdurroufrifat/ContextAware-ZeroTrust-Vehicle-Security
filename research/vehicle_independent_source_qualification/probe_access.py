from pathlib import Path
import urllib.request, json, hashlib, re, time
ROOT=Path(__file__).resolve().parent
SOURCES={
"msc_drive":"https://drive.google.com/drive/folders/1wCoiC4WzlgyLCSZMaYEdFcTqjOc0IkGQ",
"ntu_onedrive":"https://1drv.ms/f/c/8fe60b7ad7d4d18e/IgA1-ozCMat0TrFABcJPHWPOATE6AVal5oLrE9TPc1_ZC3M?e=rUMokv",
"hercules_download":"https://sites.google.com/view/herculesdataset/download",
"hercules_system":"https://sites.google.com/view/herculesdataset/system",
"hercules_license":"https://sites.google.com/view/herculesdataset/contact",
"msc_platform":"https://mscrad4r.github.io/platform/",
"ntu_readme":"https://raw.githubusercontent.com/junzhang2016/NTU4DRadLM/main/README.md",
}
out={}
for key,url in SOURCES.items():
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"CO-ACT-CPS-research-source-screen/1.0"})
        with urllib.request.urlopen(req,timeout=25) as r:
            b=r.read(4*1024*1024+1)
            if len(b)>4*1024*1024: raise ValueError("metadata size cap")
            meta={"url":url,"final_url":r.url,"http_status":r.status,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"content_type":r.headers.get("Content-Type")}
        (ROOT/(key+".html" if key!="ntu_readme" else key+".md")).write_bytes(b)
        txt=b.decode("utf-8",errors="replace")
        meta["interesting_strings"]=list(dict.fromkeys(re.findall(r'.{0,60}(?:Urban|urban|\.bag|\.zip|\.csv|RTK|Doppler|doppler|AF_initDataCallback|drive/folders|Sign in|sign in).{0,180}',txt)))[:35]
        out[key]=meta
    except Exception as e: out[key]={"url":url,"error":str(e)}
    print(key,json.dumps(out[key]),flush=True)
(ROOT/"access_probe.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
