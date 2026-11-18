/* The operator key and newly issued credentials stay in this page session. */
const get = id => document.getElementById(id);
const managementForm = get('generate');
const requestForm = get('try-request');
const managementStatus = get('manage-status');
const requestStatus = get('request-status');
const newKey = get('new-key');
const responseOutput = get('response');

function adminHeaders() {
  return { 'X-Inference-Admin-Key': get('admin').value.trim() };
}

async function readResponse(response) {
  if (response.status === 204) return null;
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
  }
  return body;
}

async function refreshKeys() {
  const items = await readResponse(await fetch('/v1/keys', {headers: adminHeaders(), cache: 'no-store'}));
  const list = get('key-list');
  list.replaceChildren();
  for (const item of items) {
    const row = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = item.label;
    const revoke = document.createElement('button');
    revoke.type = 'button';
    revoke.className = 'secondary';
    revoke.textContent = 'Revoke';
    revoke.setAttribute('aria-label', `Revoke ${item.label}`);
    revoke.addEventListener('click', async () => {
      if (!window.confirm(`Revoke the key for ${item.label}?`)) return;
      revoke.disabled = true;
      try {
        await readResponse(await fetch(`/v1/keys/${encodeURIComponent(item.id)}`,
                                      {method: 'DELETE', headers: adminHeaders()}));
        if (newKey.dataset.id === item.id) {
          if (get('api-key').value === newKey.textContent) get('api-key').value = '';
          newKey.hidden = true;
          newKey.textContent = '';
        }
        managementStatus.textContent = 'Key revoked.';
        await refreshKeys();
      } catch (error) {
        managementStatus.textContent = error.message;
      } finally {
        revoke.disabled = false;
      }
    });
    row.append(label, revoke);
    list.append(row);
  }
}

managementForm.addEventListener('submit', async event => {
  event.preventDefault();
  const button = managementForm.querySelector('[type=submit]');
  button.disabled = true;
  managementStatus.textContent = 'Generating key…';
  try {
    const generated = await readResponse(await fetch('/v1/keys', {
      method: 'POST',
      headers: {...adminHeaders(), 'Content-Type': 'application/json'},
      body: JSON.stringify({label: get('label').value}),
    }));
    newKey.textContent = generated.key;
    newKey.dataset.id = generated.id;
    newKey.hidden = false;
    get('api-key').value = generated.key;
    managementStatus.textContent = 'Key generated. Copy it now. The service displays it only once.';
    await refreshKeys();
  } catch (error) {
    managementStatus.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

get('load').addEventListener('click', async () => {
  if (!get('admin').reportValidity()) return;
  try {
    await refreshKeys();
    managementStatus.textContent = 'Issued keys loaded.';
  } catch (error) {
    managementStatus.textContent = error.message;
  }
});

requestForm.addEventListener('submit', async event => {
  event.preventDefault();
  const button = requestForm.querySelector('[type=submit]');
  button.disabled = true;
  requestStatus.textContent = 'Analyzing image…';
  responseOutput.hidden = true;
  const payload = new FormData();
  payload.set('file', get('scene').files[0]);
  if (get('save-xml').checked) payload.set('save_xml', 'true');
  try {
    const result = await readResponse(await fetch('/v1/predict', {
      method: 'POST',
      headers: {'X-Inference-Key': get('api-key').value.trim()},
      body: payload,
    }));
    responseOutput.textContent = JSON.stringify(result, null, 2);
    responseOutput.hidden = false;
    requestStatus.textContent = result.export ? 'Analysis complete. Image and XML saved.' : 'Analysis complete.';
  } catch (error) {
    requestStatus.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

get('example').textContent = [
  'curl --fail-with-body',
  '  -H "X-Inference-Key: YOUR_KEY"',
  '  -F "file=@scene.jpg;type=image/jpeg"',
  '  -F "save_xml=true"',
  `  ${location.origin}/v1/predict`,
].join(' \\\n');
