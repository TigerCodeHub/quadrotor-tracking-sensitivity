"""Per-host scan: did the SUCCESSFUL attempt of each run load the Dryden/Wind plugin?  Run host-side (stdlib only):
   python3 scan_plugin_failures.py <runs_root> <exp_dir> [<exp_dir>...]  -> CSV on stdout: host,exp,run_id,success_attempt,plugin_fail,gazebo_bytes
plugin_fail = 1 iff the success attempt's gazebo.log contains 'Failed to load system plugin'."""
import csv, json, os, socket, sys
root, dirs = sys.argv[1], sys.argv[2:]
host = socket.gethostname().split(".")[0]
w = csv.writer(sys.stdout)
for d in dirs:
    base = os.path.join(root, d)
    if not os.path.isdir(base):
        continue
    for run in sorted(os.listdir(base)):
        rs = os.path.join(base, run, "run_status.json")
        try:
            st = json.load(open(rs))
        except Exception:
            continue
        succ = [a.get("attempt") for a in st.get("attempts", []) if a.get("outcome") == "success"]
        if not succ:
            continue
        a = succ[-1]
        gl = os.path.join(base, run, "attempts", "attempt_%03d" % int(a), "gazebo.log")
        try:
            txt = open(gl, errors="ignore").read()
            fail = int("Failed to load system plugin" in txt); nb = len(txt)
        except Exception:
            fail, nb = -1, -1
        w.writerow([host, d, run, a, fail, nb])
