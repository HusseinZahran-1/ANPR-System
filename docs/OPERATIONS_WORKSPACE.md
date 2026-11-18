# Operations workspace

The FastAPI server hosts the company workspace on the configured LAN listener.
Open `/` for service status and navigation to image analysis, live captures,
the integration API console and camera settings. The workspace and live view
keep the fixed camera upload URL visible. A configured alternate bridge URL
appears separately in the authenticated live view.

The live dashboard remembers a sign-in for up to 30 days using an HTTP-only
browser cookie. Choose **Disconnect** to clear the session on that browser.
Changing the camera key requires current viewers to sign in again. Hardware
camera and bridge uploads continue to authenticate with their existing methods.

## Operator settings

Open `/settings` and unlock it with `SITE_ADMIN_KEY` from the private service
configuration. The settings API uses `X-Site-Admin-Key`. The site administrator,
camera upload key and inference keys each retain their own scope.

- `POST /camera/events` is the stable bridge endpoint. The bridge alias is
  optional and accepts uploads at `/camera/bridge/<alias>`. Saving a new alias
  replaces the previous alias immediately. The stable path continues to work.
- Public host, port and transport affect the address advertised to camera
  clients. A proxy or HTTPS listener must already exist at any advertised
  external address. Blank overrides inherit the deployment configuration.
- Enter a new camera API key only when the camera bridge and viewers are ready
  to switch. Saving it replaces the active `X-Camera-Key` without a restart.
  The service stores a digest and the settings page never returns the key.
  Direct ISAPI2 pushes continue to authenticate with their existing path token.
- `CAMERA_SETTINGS_DB` selects the private SQLite settings store. Include the
  store in operator backups to retain overrides and the active key digest.

The inference [API console](INFERENCE_API.md) is available at `/api`. See
[image and XML exports](CAPTURE_EXPORTS.md) for the storage layout.
