
var ASSET = 'users/YOUR_USER/Jeddah_LST_Dataset_2023_raw';
var pts = ee.FeatureCollection(ASSET);

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var built = wc.eq(50).rename('wc_built_share');
var bare = wc.eq(60).rename('wc_bare_share');
var win = ee.Reducer.mean();
var shares = built.addBands(bare)
  .reduceNeighborhood({reducer: win, kernel: ee.Kernel.square(15, 'meters')})
  .rename(['wc_built_share', 'wc_bare_share']);

var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
  .divide(10000).clamp(0, 1).rename('ghsl_built_frac');

var stack = wc.rename('worldcover').addBands(shares).addBands(ghsl);

var out = stack.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 10})
  .map(function (f) {
    var c = f.geometry().coordinates();
    var has = f.propertyNames();
    return ee.Feature(null, {
      'lon': ee.Algorithms.If(has.contains('lon'), f.get('lon'), c.get(0)),
      'lat': ee.Algorithms.If(has.contains('lat'), f.get('lat'), c.get(1)),
      'worldcover': f.get('worldcover'),
      'wc_built_share': f.get('wc_built_share'),
      'wc_bare_share': f.get('wc_bare_share'),
      'ghsl_built_frac': f.get('ghsl_built_frac')
    });
  });

Export.table.toDrive({
  collection: out, description: 'Jeddah_landcover', folder: 'negi', fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'worldcover', 'wc_built_share', 'wc_bare_share', 'ghsl_built_frac']
});
