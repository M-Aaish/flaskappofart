/*
 * geometrize UI bridge v3
 *  – keeps sliders in sync
 *  – keeps shapeTypes array EXACTLY equal to ticked boxes every time
 *  – calls sync whenever you load / reset / run
 */
document.addEventListener('DOMContentLoaded', () => {
  const main = window.geomMain;
  if (!main) { console.error('geomMain not found'); return; }

  /* ----- map HTML id -> engine code ----- */
  const SHAPE_IDS = {
    rectangles: 0,
    rotatedrectangles: 1,
    triangles: 2,
    ellipses: 3,
    rotatedellipses: 4,
    circles: 5,
    lines: 6,
    quadraticbeziers: 7
  };

  /* ----- synchronise once ----- */
  const syncShapeTypes = () => {
    main.shapeTypes.length = 0;                    // wipe
    Object.entries(SHAPE_IDS).forEach(([id, code]) => {
      const el = document.getElementById(id);
      if (el?.checked) main.shapeTypes.push(code);
    });
    if (main.shapeTypes.length === 0) {            // never leave it empty
      main.shapeTypes.push(2);                     // default to triangles
    }
  };

  /* run on page load */
  syncShapeTypes();

  /* ----- hook every checkbox ----- */
  Object.keys(SHAPE_IDS).forEach(id => {
    document.getElementById(id)?.addEventListener('change', syncShapeTypes);
  });

  /* ----- re-sync when image or reset buttons are used ----- */
  ['openimageinput', 'resetbutton', 'randomimagebutton']
    .forEach(id => document.getElementById(id)
      ?.addEventListener('click', () => {          // slight delay → after file picker closes
        setTimeout(syncShapeTypes, 50);
      }));

  /* ----- sliders (unchanged) ----- */
  [
    ['shapeopacity',             v => main.shapeOpacity             = v],
    ['initialbackgroundopacity', v => { main.initialBackgroundOpacity = v; main.onTargetImageChanged(); }],
    ['randomshapesperstep',      v => main.candidateShapesPerStep    = v],
    ['shapemutationsperstep',    v => main.shapeMutationsPerStep     = v]
  ].forEach(([id, setter]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const apply = () => setter(+el.value || 0);
    el.addEventListener('input',  apply);
    el.addEventListener('change', apply);
    apply();                      // initial push
  });
});
