"""Local, non-secret watchdog state I/O; control-plane actions use official MCP."""
import json
from pathlib import Path
import sys,time
path=Path('artifacts/improvement-v40/control.json')
state=json.loads(path.read_text())
command=sys.argv[1]
if command=='arm':
    state['watchdog_verified']=True;state['deadline_unix']=time.time()+9000
elif command=='register':
    payload=json.load(sys.stdin)
    assert state['pod_id'] is None
    state['pod_id']=payload['id'];state['created_unix']=time.time()
elif command=='heartbeat':
    state['heartbeat_unix']=time.time()
elif command=='stopped':
    state['stopped_unix']=time.time();state['closed']=True
elif command=='close':state['closed']=True
elif command!='read':raise ValueError(command)
if command!='read':path.write_text(json.dumps(state,indent=2))
print(json.dumps(state))
