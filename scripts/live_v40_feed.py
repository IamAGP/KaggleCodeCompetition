"""Forward numeric Runpod progress to the owner-private Site. Secret arrives on stdin only."""
import json, pathlib, subprocess, sys, time, urllib.request, termios

if sys.stdin.isatty():
    settings=termios.tcgetattr(sys.stdin.fileno());settings[3]&=~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,settings)
print('Ready for private Site service credential; stdin echo disabled.',flush=True)
token=json.loads(sys.stdin.readline())['token']
remote="""import json,pathlib,time
p=pathlib.Path('/workspace/v40/results'); runs=[]
keys={'event','epoch','step','steps','ce','contrastive','loss','elapsed_seconds','rows_completed'}
for fold in [1,2]:
 for arm in ['dense_ce','blocks_ce','dense_supcon','blocks_supcon']:
  d=p/arm/f'fold-{fold}';events=[]
  if (d/'progress.jsonl').exists():
   for line in (d/'progress.jsonl').read_text().splitlines():
    try: item=json.loads(line)
    except json.JSONDecodeError: continue
    events.append({k:v for k,v in item.items() if k in keys})
  result=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else None
  runs.append({'arm':arm,'fold':fold,'events':events,'result':result})
print(json.dumps({'experiment':'V40','observed_at':time.time(),'runs':runs}))
"""
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','/Users/adithyagiridharan/.ssh/id_ed25519_runpod_new','-p','16343','root@213.173.109.7','python -']
out=pathlib.Path('artifacts/improvement-v40');deadline=json.loads((out/'control.json').read_text())['deadline_unix']
while time.time()<deadline:
    try:
        r=subprocess.run(command,input=remote,text=True,capture_output=True,timeout=35,check=True)
        data=json.loads(r.stdout);body=json.dumps(data).encode()
        request=urllib.request.Request('https://name-that-llm-lab-notebook.jollyray514.chatgpt.site/api/live',data=body,method='PUT',headers={'Content-Type':'application/json','OAI-Sites-Authorization':'Bearer '+token})
        with urllib.request.urlopen(request,timeout=25) as response:
            receipt=json.load(response)
        assert receipt['saved'] and receipt['observed_at']==data['observed_at']
        (out/'live-telemetry.json').write_bytes(body)
        complete=sum(bool(x['result'] and x['result']['completed']) for x in data['runs'])
        print(json.dumps({'published_at':data['observed_at'],'completed_fits':complete,'bytes':len(body)}),flush=True)
        if complete==8: break
    except Exception as e:
        # Exception type only: HTTP errors and command details must never leak credentials.
        print(json.dumps({'feed_error_type':type(e).__name__,'time':time.time()}),flush=True)
    time.sleep(20)
