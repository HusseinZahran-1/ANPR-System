# Workspace layout and relocation

| Location | Contents |
| --- | --- |
| app/ | FastAPI, camera and dashboard, detector, OCR and validators |
| scripts/ | Local training, evaluation and dataset phase commands |
| tools/ | Reusable data intake, verification and measurement helpers |
| tests/ | Project test suite |
| docs/ | Maintained deployment and integration guides |
| data/raw/ | Immutable input inventory |
| data/prepared/, data/processed/, data/reviewed/ | Derived splits, local capture inbox and review quarantine |
| models/ | Private checkpoints and training run outputs |
| logs/ | Local experiment provenance and historical reports |
| publication/ | Local publication checkouts and receipts |

Recreate a moved virtual environment with Python 3.12. Use the checked-in
requirements and launcher. Set private model paths in `.env` and use relative
paths from the project root or explicit absolute overrides.

Camera archives and review stores use `CAMERA_ARCHIVE_DIR` and
`CAMERA_REVIEW_DIR`. Defaults retain the existing project directories.
Archive records store relative image locations and SHA-256 checksums.
Review records refer to the filename in the review store. Preserve subdirectory
structure when relocating either store. Old model paths in archived inference
metadata identify the original run.

For manual camera operation use the PowerShell launcher with the verified
listener address and port. `HOST` and `PORT` apply by default, and
`-BindAddress` and `-Port` override both the listener and its advertised
defaults. A concrete bind address is advertised as-is. For wildcard binds,
multi-interface hosts or proxies, provide `CAMERA_PUBLIC_HOST`, and set
`CAMERA_PUBLIC_PORT` and `CAMERA_PUBLIC_TRANSPORT` when the external listener
differs. HTTPS requires an independently configured TLS listener or proxy.
Disable proxy access logs for the tokenized ISAPI2 route. The manual launcher
disables its own access logging.

Use `/live` and enter the camera key from the private deployment environment.
**Show camera settings** reveals the tokenized path in the authenticated
session. A developer bridge sends multipart `file` and optional `camera_id`
to `/camera/events` with `X-Camera-Key`. The operator provisions that key
privately. License plate uploads from another backend use the dedicated
[inference API](INFERENCE_API.md).

Historical phase inputs, experiment directories and versioned reports remain in
place because scripts and manifests refer to them by name. Review references
and verify asset hashes before relocating a phase. Keep public Git history in
a separate checkout, with data and local settings excluded.

## Start from Windows after a move

Keep the private `.env`, model files and `data/` folders with the app,
launcher and requirements. The data directory contains capture history,
daily image/XML exports, review files, and any issued-key or camera-settings
databases. Preserve its directory structure. Stop the running receiver
before moving the live folder.

Open PowerShell at the new location. For a direct Ethernet deployment, use
the receiving computer's current IPv4 address:

```powershell
Set-Location -LiteralPath 'D:\Projects\ANPR_System'
.\run_mvp.ps1 -BindAddress 192.0.2.10 -Port 8000 -RequireModels
```

Replace the example address with the receiver's actual interface address.
Open that address on port 8000 for the workspace. The filesystem location
does not determine the LAN address. If the receiving address changes,
update any advertised camera destination override and the camera's target.

A copied environment can work on the same computer when its base Python
installation still exists. The launcher uses the Python executable next to
the checkout. Use `python -m pip` via that executable for package commands.
On another computer, or if the copied environment fails, preserve the old
`.venv` under a different name and rebuild with Python 3.12. The CUDA setup
used for the active local model profile is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install setuptools==78.1.0 wheel
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements-vehicle.txt
```

For CPU operation install the CPU PyTorch wheels, use
`requirements-public.txt` and `requirements-authority.txt` instead of
`requirements.txt`, then install the optional vehicle requirements above
and set `DEVICE=cpu` in the private configuration.
