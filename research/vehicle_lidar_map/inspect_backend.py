import hashlib,json,urllib.request,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
url="https://files.pythonhosted.org/packages/8e/34/d853cb135ed47455f76f4a8b0812070a95de9df246776267b717f891b383/kiss_icp-1.3.0-cp312-cp312-win_amd64.whl"
expected="b3c57195d31f375650ec4e0bb8a8f3ea4d9c6ec6296e8eb3fbbf1cf82d5f2809"
(R/"runtime").mkdir(exist_ok=True)
p=R/"runtime"/url.rsplit("/",1)[1]
urllib.request.urlretrieve(url,p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
with zipfile.ZipFile(p) as z:
 for n in z.namelist():
  if n.endswith(".py") and "/datasets/" not in n and "/tools/" not in n:
   print("\nFILE",n,"\n",z.read(n).decode())
  if "LICENSE" in n.upper(): (R/"BACKEND_LICENSE.txt").write_bytes(z.read(n))
(R/"backend_pin.json").write_text(json.dumps(dict(url=url,sha256=expected,version="1.3.0",wheel=str(p)),indent=2))
