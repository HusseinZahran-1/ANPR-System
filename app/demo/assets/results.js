/* Shared result fields and vehicle overlays for the upload and live views. */
(() => {
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function fields(items) {
    const list = element('dl', undefined, 'result-fields');
    for (const [label, value, className] of items) {
      const item = element('div', undefined, className);
      item.append(element('dt', label), element('dd', value));
      list.append(item);
    }
    return list;
  }

  function plateFields(ocr) {
    const parts = ocr?.plate_parts;
    const full = parts?.full_text || ocr?.text || '';
    const ambiguous = !parts?.plate_code && !parts?.plate_number && /^[0-9]{6}$/.test(full);
    const awaitingReview = ambiguous ? 'Needs review' : 'Unresolved';
    const result = element('div');
    result.append(fields([
      ['Full plate reading', full || 'Unreadable', 'full-plate'],
      ['Registration code', parts?.plate_code ?? awaitingReview, 'plate-part'],
      ['Plate number', parts?.plate_number ?? awaitingReview, 'plate-part'],
    ]));
    if (ambiguous) {
      const options = [1, 2].map(cut => `${full.slice(0, cut)} + ${full.slice(cut)}`);
      result.append(element('p', `Possible code + number: ${options.join(' or ')}. Check the plate image.`,
                            'plate-split-note'));
    }
    return result;
  }

  function vehicleSection(result) {
    if (!result?.vehicle_status || result.vehicle_status === 'disabled') return null;
    const section = element('section', undefined, 'vehicle-results');
    section.append(element('h3', 'Vehicle details'));
    const vehicles = result.vehicles ?? [];
    if (!vehicles.length) {
      const detail = result.vehicle_status === 'unavailable'
        ? 'Vehicle analysis is unavailable.'
        : 'No vehicle detected.';
      section.append(element('p', detail, 'vehicle-note'));
      return section;
    }

    vehicles.forEach((vehicle, index) => {
      const card = element('div', undefined, 'vehicle-card');
      const plateRefs = (vehicle.plate_indices ?? []).map(i => i + 1);
      const association = plateRefs.length
        ? ` · Linked to plate ${plateRefs.join(', ')}`
        : ' · No plate linked';
      card.append(element('div', `Vehicle ${index + 1}${association}`, 'vehicle-note'));
      card.append(fields([
        ['Type', vehicle.vehicle_type || 'Unknown'],
        ['Color', vehicle.color ?? 'Unknown'],
        ['Make', vehicle.make ?? 'Unknown'],
      ]));
      section.append(card);
    });
    const note = result.vehicle_status === 'detection_only'
      ? 'Basic detection only. Vehicle attributes are unavailable.'
      : 'Vehicle attributes are estimates. Verify them against the image.';
    section.append(element('p', note, 'vehicle-note'));
    return section;
  }

  function drawVehicles(ctx, vehicles, scale) {
    ctx.save();
    ctx.strokeStyle = '#ffca75';
    ctx.fillStyle = '#ffca75';
    ctx.lineWidth = 2 * scale;
    ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.font = `bold ${12 * scale}px Segoe UI`;
    ctx.textBaseline = 'top';
    for (const [index, vehicle] of (vehicles ?? []).entries()) {
      const [x1, y1, x2, y2] = vehicle.box;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(`Vehicle ${index + 1}`, x1 + 3 * scale, y1 + 3 * scale);
    }
    ctx.restore();
  }

  window.anprResults = { plateFields, vehicleSection, drawVehicles };
})();
