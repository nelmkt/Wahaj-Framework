// Wahaj stage 1 (satellite panel) for Seoul: summer land surface temperature.
// Descriptive only: no matching, model or gate has been run for Seoul.
// Run in the Google Earth Engine Code Editor.
var seoul = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME', 'Seoul'));
function prep(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0)).and(qa.bitwiseAnd(1 << 4).eq(0));
  var lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST');
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  var ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  return lst.addBands(ndvi).updateMask(clear);
}
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(seoul)
  .filter(ee.Filter.calendarRange(2023, 2025, 'year'))
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .filter(ee.Filter.lt('CLOUD_COVER', 60))
  .map(prep);
var comp = col.median().clip(seoul);
var nd = comp.select('NDVI');
var vis = {min: 26, max: 46, palette: ['313695', '4575b4', '74add1', 'abd9e9', 'fee090', 'fdae61', 'f46d43', 'd73027', 'a50026']};
Map.centerObject(seoul, 11);
Map.addLayer(comp.select('LST'), vis, 'Seoul summer LST (C)');
var mean = function (mask) {
  return comp.select('LST').updateMask(mask).reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.count(), '', true),
    geometry: seoul.geometry(), scale: 30, maxPixels: 1e10});
};
print('Scenes used', col.size());                         // 40 on 24 Nov 2026
print('Low vegetation, 0 < NDVI < 0.2', mean(nd.gt(0).and(nd.lt(0.2))));   // 40.3 C
print('Dense vegetation, NDVI > 0.6', mean(nd.gt(0.6)));                   // 32.5 C
print('Water, NDVI < 0', mean(nd.lt(0)));                                  // 26.9 C
