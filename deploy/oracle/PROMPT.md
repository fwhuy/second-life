# Deployment prompt — Second Life AI on Oracle Cloud

Copy everything below the line into Claude Cowork / Claude Code, replacing every
`<PUBLIC_IP>` with the instance's public IP address. That is the only value you
need to supply.

---

Deploy a Flask + PyTorch web app onto an Oracle Cloud VM and serve it over HTTPS.

## My details

- VM public IP: `<PUBLIC_IP>`
- SSH: `ssh -i ~/.ssh/oracle_secondlife ubuntu@<PUBLIC_IP>` (Ubuntu 24.04, ARM64)
- Domain: I don't own one. Setting up a free one is part of your job — see step 8.

Project root on this Mac: `/Users/huy/Desktop/NYU Shanghai AI Course/Image Classifcation`

## What the app is

A bilingual waste-classification site. A React frontend (plain files, no build
step) posts uploaded photos to a Flask backend, which runs a fine-tuned ResNet-50
and returns the real softmax output. It currently runs only on my laptop; I want
it publicly reachable.

Two directories matter:

- `website/` — `app.py` (Flask), `index.html`, `support.js`, `vendor/` (React UMD),
  `ood_bank.npz` (5.6 MB), and `checkpoints/baseline_resnet50/fold0/best.pth`
  (**180 MB** — gitignored, so it exists only on this Mac; it must be copied to
  the server, not cloned).
- `model/src/` — the training repo's inference code, which `app.py` imports as
  `src.*`. On the server it must sit at `<app_root>/model/src/` with `website/`
  as its sibling, because `app.py` resolves it as `HERE.parent / "model"`.

`app.py` already accepts `--device cpu`, `--host`, `--port`, `--checkpoint`, and
`--ood-bank`, and reads `HOST`/`PORT` from the environment. It needs no changes
to run on a server.

## Steps

**1. Verify the machine.** SSH in and confirm architecture (`uname -m` should be
`aarch64`), RAM (`free -h`), and disk. Report back before installing anything.

**2. Install Python dependencies** into a virtualenv at `/opt/second-life/.venv`.

IMPORTANT — this is ARM64, not x86. Do **not** use
`--index-url https://download.pytorch.org/whl/cpu`; that index is for x86_64. On
aarch64 the default PyPI wheel for `torch` is already CPU-only. Install plainly:

```
pip install torch torchvision timm numpy pandas scikit-learn scipy pillow PyYAML flask
```

Preferred pins (matching what the model was validated under) are in
`deploy/hf-space/requirements.txt`: torch 2.13.0, torchvision 0.28.0, timm
1.0.28, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.17.1, pillow
12.3.0, PyYAML 6.0.3. If any pin has no aarch64 wheel, fall back to the nearest
available version and say which ones you changed — do not silently drift.

**3. Copy the app across** with `rsync` or `scp`: `website/` (excluding `.venv`
and `__pycache__`) and `model/src/`, into `/opt/second-life/`. Verify the
checkpoint arrived intact by comparing `sha256sum` on both ends — a truncated
180 MB transfer is the most likely silent failure here.

**4. Smoke-test before daemonising.** Run the app by hand and confirm both
endpoints work:

```
/opt/second-life/.venv/bin/python /opt/second-life/website/app.py --device cpu --port 5001
curl -s localhost:5001/api/model
curl -s -F "image=@/opt/second-life/website/../bottle.jpeg" localhost:5001/api/identify
```

`/api/model` must report `resnet50.tv2_in1k`, `ood_bank_size: 2166`, and
`guarded: true`. If the OOD bank fails to load, the app still starts but with the
closed-set guard silently off — check the startup log for "OOD guard ON".

**5. Replace the dev server.** `app.py` ends with Flask's built-in
`app.run()`, which prints a production warning and handles one request at a time.
Install `waitress` and swap that final call for `waitress.serve(app, host=...,
port=..., threads=4)`, keeping everything above it unchanged — the model globals
are populated in `main()` before serving, so a WSGI factory would break them.

**6. systemd service** at `/etc/systemd/system/second-life.service` so it starts
on boot and restarts on failure. Run it as a non-root user, bound to
`127.0.0.1:5001` only (Caddy will be the public face). Enable and start it.

**7. Firewall — both layers, and this is the single most common failure.** Oracle
Ubuntu images ship iptables rules blocking everything except SSH, *in addition*
to the cloud-side Security List. Opening one and not the other produces a site
that hangs with no error.

Cloud side: the VCN subnet's Security List needs ingress rules for TCP 80 and 443
from `0.0.0.0/0`. I may not have added these — check, and if they're missing tell
me exactly where to click, since you likely can't reach my Oracle console.

Instance side:

```
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

Confirm the rules survive a reboot.

**8. Get a free domain.** A domain is required for HTTPS, and HTTPS is required
because the site's live-camera capture uses `getUserMedia`, which browsers block
on plain `http://`. Two routes — try the first, fall back to the second rather
than getting stuck:

*Preferred — DuckDNS* (`second-life-ai.duckdns.org`, a nicer name for a poster).
Register at duckdns.org, which needs a GitHub/Google sign-in. If you have browser
access, do it and tell me which account you used; if a login wall blocks you,
stop and ask me rather than burning time. Once registered, point it at the VM:

```
curl "https://www.duckdns.org/update?domains=<SUBDOMAIN>&token=<TOKEN>&ip=<PUBLIC_IP>"
```

*Fallback — sslip.io*, which needs **no registration at all**. Any hostname of
the form `<dashed-ip>.sslip.io` already resolves to that IP — for `152.67.1.2`,
the hostname is `152-67-1-2.sslip.io`. Let's Encrypt issues certificates for it
normally. Uglier, but it works instantly and cannot fail on an account problem.

Whichever you use, verify before continuing: `dig +short <hostname>` must return
the VM's public IP.

**9. Caddy for automatic HTTPS.** Install from the official apt repo. Caddyfile
(substitute whichever hostname step 8 produced):

```
<HOSTNAME> {
    reverse_proxy 127.0.0.1:5001
    request_body {
        max_size 25MB
    }
}
```

The larger body limit matters — users upload phone photos, which routinely exceed
Caddy's default. Caddy will obtain a Let's Encrypt certificate over HTTP-01,
which requires port 80 to be publicly reachable, so do this after step 7.

**10. Verify from the outside**, not from the server:

- `curl -sI https://<HOSTNAME>` returns 200 with a valid certificate.
- `curl -s https://<HOSTNAME>/api/model` returns the model JSON.
- Uploading a photo through the site returns a real prediction, and the dark
  "Demo version" banner at the top of the page is **gone**. That banner is driven
  by a `/api/model` probe on page load, so its disappearance is the real proof
  the backend is live.
- `sudo systemctl reboot`, wait, and confirm the site comes back by itself.

Report what you changed from this plan and anything that failed. Do not tell me
it works without showing the actual command output.
