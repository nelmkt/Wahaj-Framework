import csv
import json
import math
from pathlib import Path

"""Independent standard-library arithmetic checks for the reviewed package."""

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'tables'
CONFIG = json.loads((TABLES / 'results.json').read_text(encoding='utf-8'))['config']

def rows(name):
    with (TABLES / name).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def n(x):
    return float(x)

def eq(a, b, label, tol=1e-6):
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f'{label}: {a} != {b}')

water = {r['level']: r for r in rows('water.csv')}
central = n(water['central']['m3_per_ha_yr'])
for r in water.values():
    eq(n(r['m3_per_ha_yr']), 10000*n(r['depth_m']), 'water depth')

decision = rows('decision_summary.csv')
for r in decision:
    if not r['basis'].startswith('measured:'):
        continue
    water_calc = n(r['pixels_mean']) * .09 * central
    eq(n(r['water_m3']), water_calc, f"water {r['option_id']}")
    eq(n(r['m3_per_degree']), water_calc/abs(n(r['cooling_C'])), f"ratio {r['option_id']}")
    eq(n(r['scenario_low']), n(r['m3_per_degree_low'])*n(water['low']['depth_m'])/n(water['central']['depth_m']), 'scenario low')
    eq(n(r['scenario_high']), n(r['m3_per_degree_high'])*n(water['high']['depth_m'])/n(water['central']['depth_m']), 'scenario high')
print('decision-summary measured rows checked', sum(r['basis'].startswith('measured:') for r in decision))

look = {r['option_id']: r for r in decision}
scenario = rows('decision_scenarios.csv')
for r in scenario:
    d = look[r['option_id']]
    expected = n(d['m3_per_degree']) * n(water[r['water_case']]['depth_m']) / n(water['central']['depth_m'])
    eq(n(r['m3_per_degree']), expected, 'scenario water')
    factors = CONFIG['kwh_per_m3'][r['source']]
    intensity = factors[{'low':0,'central':1,'high':2}[r['energy_case']]]
    eq(n(r['MWh_per_degree']), expected*intensity/1000, 'scenario energy')
    eq(n(r['MWh_per_degree_low'])/n(r['m3_per_degree_low']), n(r['MWh_per_degree'])/n(r['m3_per_degree']), 'scenario energy low')
    eq(n(r['MWh_per_degree_high'])/n(r['m3_per_degree_high']), n(r['MWh_per_degree'])/n(r['m3_per_degree']), 'scenario energy high')
print('scenario rows',len(scenario))

for r in rows('decision_pairwise.csv'):
    a,b=look[r['option_A']],look[r['option_B']]
    eq(n(r['water_intensity_A_over_B_at_point_tie']), n(b['m3_per_degree'])/n(a['m3_per_degree']), 'pair tie')
print('pairwise rows',len(rows('decision_pairwise.csv')))

for r in rows('logic_primary_area_mixing.csv')+rows('followup_area_mixing.csv'):
    eq(n(r['difference_C']),n(r['measured_C'])-n(r['benchmark_C']),'mixing diff')
print('mixing rows',len(rows('logic_primary_area_mixing.csv'))+len(rows('followup_area_mixing.csv')))

for r in rows('test_by_dose.csv')+rows('test_support_coasts.csv'):
    if r.get('difference_C') and r.get('model_C') and r.get('measured_C'):
        eq(n(r['difference_C']),n(r['model_C'])-n(r['measured_C']),'model diff dose')
    if r.get('difference_px_C') and r.get('model_px_C') and r.get('measured_px_C'):
        eq(n(r['difference_px_C']),n(r['model_px_C'])-n(r['measured_px_C']),'model diff coast')

for r in rows('test_region_holdouts.csv'):
    eq(n(r['difference_C']),n(r['model_C'])-n(r['measured_C']),'holdout diff')

intervals_checked = 0
for file in TABLES.glob('*.csv'):
    if file.stem.endswith('_INVALID'):
        continue
    rr=rows(file.name)
    if not rr: continue
    cols=rr[0].keys()
    for lo in cols:
        if '_lo_' not in lo and not lo.endswith('_lo_C'): continue
        hi=lo.replace('_lo_','_hi_') if '_lo_' in lo else lo.replace('_lo_C','_hi_C')
        if hi not in cols: continue
        base=lo.replace('_lo_','_').replace('_lo_C','_C')
        if base not in cols: continue
        for r in rr:
            if not(r[lo] and r[hi] and r[base]): continue
            try: low,point,high=map(n,(r[lo],r[base],r[hi]))
            except ValueError: continue
            if low>point or point>high: raise AssertionError(f'interval {file.name} {lo}: {low},{point},{high}')
            intervals_checked += 1
print('interval containments checked', intervals_checked)

simple_center = {
    'decision_bare_places_model.csv': 'cooling_C',
    'decision_by_coast.csv': 'cooling_px_C',
    'followup_area_mixing.csv': 'difference_C',
    'logic_primary_area_mixing.csv': 'difference_C',
    'followup_coastal_difference.csv': 'difference_C',
    'logic_primary_coastal_difference.csv': 'difference_C',
    'logic_panel_proximity_proxy.csv': 'per_pixel_C',
    'gee_common_month_primary.csv': 'per_pixel_C',
    'gee_common_month_primary_v2.csv': 'per_pixel_C',
    'gee_common_month_primary_v6.csv': 'per_pixel_C',
    'gee_exact_isolation_effects_from_raw_v4.csv': 'per_pixel_C',
    'gee_pixel_centered_isolation_effects_v7.csv': 'per_pixel_C',
    'gee_isolation_dose1_comparison_v5_corrected.csv': 'per_pixel_C',
    'gee_neighbor_count_gradient_v7.csv': 'per_pixel_C',
    'gee_neighbor_count_gradient_lst_tercile_v7_PRINTFAIL.csv': 'per_pixel_C',
    'gee_neighbor_count_gradient_lst_tercile_v7_verified.csv': 'per_pixel_C',
}
simple_outside = []
simple_checked = 0
for file in TABLES.glob('*.csv'):
    if file.stem.endswith('_INVALID'):
        continue
    data = rows(file.name)
    if not data or not {'lo_C','hi_C'} <= data[0].keys():
        continue
    center = simple_center.get(file.name, 'estimate_C')
    if center not in data[0]:
        raise AssertionError(f'No center mapping for {file.name}')
    for index,row in enumerate(data, start=2):
        if not (row['lo_C'] and row['hi_C'] and row[center]):
            continue
        lo,point,hi = map(n,(row['lo_C'],row[center],row['hi_C']))
        simple_checked += 1
        if not lo <= point <= hi:
            simple_outside.append((file.name,index,lo,point,hi))
print('generic interval containments checked', simple_checked)
print('generic interval points outside bounds', simple_outside)

for file_name in ('gee_common_month_paired.csv', 'gee_common_month_paired_v2.csv'):
    for index, row in enumerate(rows(file_name), start=2):
        if not row['new_minus_old_C']:
            continue
        old, new, change = map(n, (row['old_per_pixel_C'], row['new_per_pixel_C'], row['new_minus_old_C']))
        if abs((new - old) - change) > 1e-9:
            raise AssertionError(f'paired difference {file_name}:{index}')
        lo, hi = map(n, (row['paired_lo_C'], row['paired_hi_C']))
        if not lo <= change <= hi:
            raise AssertionError(f'paired interval {file_name}:{index}')

meta=json.loads((ROOT/'code'/'gee'/'panel_meta.json').read_text(encoding='utf-8'))
print('ET years',meta.get('reference_et_mm_per_year'))
print('rows:',{'primary':len(rows('logic_primary_class_estimates.csv')),'matched':len(rows('measured_dose.csv')),'holdouts':len(rows('test_region_holdouts.csv'))})
for setting in {'outside the built-up area','built-up surroundings'}:
    pop=[r for r in rows('population_retention.csv') if r['setting']==setting]
    if pop:
        print('retention',setting, 'classified',sum(int(r['classified']) for r in pop),'matched',sum(int(r['matched']) for r in pop),'caliper0.5',sum(int(r['caliper_0.5_sd']) for r in pop))
    sup=[r for r in rows('test_support_counts.csv') if r['setting']==setting]
    if sup:
        print('support',setting,sum(int(r['n_supported']) for r in sup),'/',sum(int(r['n_matched']) for r in sup))
for r in rows('logic_primary_class_estimates.csv'):
    if r['setting']=='outside the built-up area' and r['dose_class']=='all' and r['specification']=='pre-treatment-only strata':
        print('primary slope',r['per_pixel_C'],'n',r['n_matched'])
for r in rows('logic_block_summary.csv'):
    if r['setting']=='outside the built-up area' and r['dose_class']=='all' and r['specification']=='pre-treatment-only strata':
        print('block leverage',r['largest_block_dose_squared_share'],'n blocks',r['n_deleted_blocks'],'delete range',r['deletion_min_C'],r['deletion_max_C'])
for r in rows('logic_sio_integrity.csv'):
    eq(n(r['depth_m']),n(r['source_total_m3'])/n(r['irrigated_area_ha'])/10000,'SIO depth')
print('SIO rows',len(rows('logic_sio_integrity.csv')),'outside band',sum(r['inside_assumed_depth_band']=='False' for r in rows('logic_sio_integrity.csv')))
print('missing May',meta['scenes_per_year_month']['2014-05'])
