const el = id => document.getElementById(id);
const statusText = el('settings-status');
const endpoint = '/ops/camera-settings';

async function request(options = {}) {
  const response = await fetch(endpoint, {
    cache: 'no-store',
    ...options,
    headers: {'X-Site-Admin-Key': el('admin-key').value.trim(), ...options.headers},
  });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Check the settings fields.');
  return body;
}

function show(settings) {
  const values = settings.overrides;
  el('public-host').value = values.public_host;
  el('public-host').placeholder = settings.effective.host || 'Receiver IP or DNS name';
  el('public-port').value = values.public_port ?? '';
  el('public-port').placeholder = settings.effective.port;
  el('transport').value = values.transport ?? '';
  el('bridge-alias').value = values.bridge_alias;
  el('new-camera-key').value = '';
  el('bridge-url').textContent = settings.effective.bridge_url ?? settings.effective.bridge_path;
  el('editable').hidden = false;
  el('destination').hidden = false;
}

el('unlock').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    show(await request());
    statusText.className = 'status-good';
    statusText.textContent = 'Camera settings loaded.';
  } catch (error) {
    el('editable').hidden = true;
    el('destination').hidden = true;
    statusText.className = 'status-bad';
    statusText.textContent = error.message;
  }
});

el('save-settings').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button');
  button.disabled = true;
  const cameraKey = el('new-camera-key').value.trim();
  try {
    show(await request({
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        public_host: el('public-host').value.trim(),
        public_port: el('public-port').value ? Number(el('public-port').value) : null,
        transport: el('transport').value || null,
        bridge_alias: el('bridge-alias').value.trim(),
        camera_key: cameraKey || null,
      }),
    }));
    statusText.className = 'status-good';
    statusText.textContent = cameraKey ? 'Settings saved. Update clients to use the new camera API key.' : 'Settings saved.';
  } catch (error) {
    statusText.className = 'status-bad';
    statusText.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
