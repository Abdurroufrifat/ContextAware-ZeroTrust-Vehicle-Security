from pathlib import Path
import re
R=Path(__file__).resolve().parent
for key in ["msc_drive","ntu_onedrive"]:
 s=(R/(key+".html")).read_text(encoding="utf-8")
 for pat in [r'data-id="[^"]+"[^>]{0,400}',r'.{0,80}application/vnd.google-apps.folder.{0,130}',r'.{0,80}(?:URBAN|urban|CALIB|calib|authkey|api.onedrive|signIn).{0,140}']:
  print(key,pat)
  for x in re.findall(pat,s)[:20]:print(x)
