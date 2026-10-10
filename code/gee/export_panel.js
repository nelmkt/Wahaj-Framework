var G = (function () {
  var YEARS = [2014, 2015, 2018, 2019, 2024, 2025], MONTHS = [5, 6, 7, 8, 9];
  var RECT = [39.0, 21.2, 39.4, 21.8], CELL_M = 90, CONTROL_FRACTION = 0.10;
  var DW_PRE = ['2015-06-01', '2015-10-01'], DW_POST = ['2024-05-01', '2025-10-01'], NB_RADIUS_M = 500;
  function build() {
    var rect = ee.Geometry.Rectangle(RECT);
    var land = rect.intersection(ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
      .filter(ee.Filter.eq('country_na', 'Saudi Arabia')).geometry(), ee.ErrorMargin(1));
    var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(rect).filter(ee.Filter.lt('CLOUD_COVER', 20));
    var proj30 = col.first().select('SR_B4').projection();
    var cp = col.map(function (img) {
      var qa = img.select('QA_PIXEL'), sat = img.select('QA_RADSAT');
      var m = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 2).eq(0)).and(qa.bitwiseAnd(1 << 3).eq(0))
        .and(qa.bitwiseAnd(1 << 4).eq(0)).and(sat.eq(0));
      var o = img.select('SR_B.*').multiply(0.0000275).add(-0.2);
      return ee.Image(ee.Image.cat([o.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'),
        o.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI'), o.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI'),
        img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST'),
        img.select('ST_EMIS').multiply(0.0001).rename('EMIS')]).updateMask(m).copyProperties(img, ['system:time_start']));
    });
    var comps = {}, counts = {};
    YEARS.forEach(function (y) {
      var monthly = MONTHS.map(function (mth) {
        var c = cp.filter(ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(mth, mth, 'month'));
        counts[y + '-0' + mth] = c.size();
        return c.median().set('n', c.size());
      });
      comps[y] = ee.ImageCollection.fromImages(monthly).filter(ee.Filter.gt('n', 0)).mean().reproject(proj30);
    });
    var water = null;
    YEARS.forEach(function (y) { var w = comps[y].select('NDWI').gt(0); water = water === null ? w : water.or(w); });
    var ok = water.unmask(0).focal_max({radius: 300, units: 'meters'}).not();
    var nd = {}; YEARS.forEach(function (y) { nd[y] = comps[y].select('NDVI'); });
    var greened = nd[2014].lt(0.15).and(nd[2015].lt(0.15)).and(nd[2024].gte(0.30)).and(nd[2025].gte(0.30)).and(ok);
    var never = ok; YEARS.forEach(function (y) { never = never.and(nd[y].lt(0.15)); });
    var dist = greened.selfMask().fastDistanceTransform(64).sqrt().multiply(30).unmask(99999).rename('dist_greened_m');
    var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(rect).filter(ee.Filter.calendarRange(5, 9, 'month')).select('built');
    function to30(im) { return im.reproject(proj30.atScale(10)).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64}).reproject(proj30); }
    var built_pre = to30(dw.filterDate(DW_PRE[0], DW_PRE[1]).mean()).rename('built_pre');
    var built_post = to30(dw.filterDate(DW_POST[0], DW_POST[1]).mean()).rename('built_post');
    var notGreen = greened.unmask(0).not(), kern = ee.Kernel.circle(NB_RADIUS_M, 'meters');
    var nb_pre = built_pre.updateMask(notGreen).reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kern, skipMasked: false}).reproject(proj30).rename('nb_built_pre');
    var nb_post = built_post.updateMask(notGreen).reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kern, skipMasked: false}).reproject(proj30).rename('nb_built_post');
    var elev = ee.Image('USGS/SRTMGL1_003').rename('elev');
    var gh = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2015').select('built_surface').divide(1e4);
    var ghsl_own = gh.reproject(proj30).rename('ghsl_2015');
    var ghsl_nb = gh.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kern}).reproject(proj30).rename('ghsl_nb_2015');
    var pre_bare = nd[2014].lt(0.15).and(nd[2015].lt(0.15)).and(ok).unmask(0).rename('pre_bare_frac');
    var late = greened.and(nd[2018].lt(0.15)).and(nd[2019].lt(0.15)).unmask(0).rename('late_frac');
    var bands30 = [greened.unmask(0).rename('greened_frac'), never.unmask(0).rename('never_frac'), pre_bare, late, ok.unmask(0).rename('dry_frac')];
    YEARS.forEach(function (y) {
      bands30.push(comps[y].select('NDVI').rename('ndvi_' + y), comps[y].select('LST').rename('lst_' + y), comps[y].select('EMIS').rename('emis_' + y), comps[y].select('NDBI').rename('ndbi_' + y));
    });
    var stack30 = ee.Image.cat(bands30.concat([built_pre, built_post, nb_pre, nb_post, ghsl_own, ghsl_nb, elev])).reproject(proj30);
    var cell = proj30.scale(3, 3);
    var mean90 = stack30.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024}).reproject(cell);
    var dmin = dist.reproject(proj30).reduceResolution({reducer: ee.Reducer.min(), maxPixels: 64}).reproject(cell);
    var g = mean90.select('greened_frac'), nv = mean90.select('never_frac');
    var rnd = ee.Image.random(42).reproject(cell);
    var cls = ee.Image(0).where(nv.eq(1).and(dmin.gt(300)).and(rnd.lt(CONTROL_FRACTION)), 3)
      .where(nv.eq(1).and(dmin.lte(150)), 2).where(g.gte(1 / 9 - 1e-6).and(mean90.select('pre_bare_frac').gte(1 - 1e-6)), 1).rename('cls').reproject(cell);
    cls = cls.where(cls.eq(0).and(rnd.lt(CONTROL_FRACTION)).and(mean90.select('dry_frac').gte(1 - 1e-6)), 4).rename('cls');
    var out = mean90.addBands(dmin).addBands(cls).addBands(rnd.rename('rnd')).updateMask(cls.gt(0)).clip(land);
    return {out: out, land: land, cell: cell, counts: counts, greened: greened, cls: cls, mean90: mean90};
  }
  return {build: build, RECT: RECT, CELL_M: CELL_M};
})();

var GB = G.build();
var panel = GB.out.sample({region: GB.land, projection: GB.cell, geometries: true, tileScale: 16});
Export.table.toDrive({collection: panel, description: 'panel', fileFormat: 'CSV'});
print('scenes per year-month', ee.Dictionary(GB.counts));
