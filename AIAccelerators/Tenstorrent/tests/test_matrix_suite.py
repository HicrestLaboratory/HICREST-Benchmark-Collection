import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matrix_suite import shapes, cases, case_id, BASE
from parser import parse_stdout


class CampaignTest(unittest.TestCase):
    def test_exact_shapes_and_operation_dimensions(self):
        all_shapes = shapes()
        self.assertEqual(len(all_shapes), 15)
        self.assertIn(('square',4096,4096), all_shapes)
        self.assertIn(('square',8192,8192), all_shapes)
        self.assertIn(('short_wide',262144,5376), all_shapes)
        campaign = list(cases())
        self.assertEqual(len(campaign), 840)
        self.assertEqual(len({case_id(c) for c in campaign}), 840)
        for c in campaign:
            self.assertIn((c['shape_family'],c['N'],c['K']), all_shapes)
            self.assertIn(c['M'], (1,) if c['operation']=='gemv' else (64,512,2048))
        scheduled = json.loads((BASE/'jobs.yaml').read_text())
        self.assertTrue(scheduled['sequential'])
        self.assertEqual({j['tag'] for j in scheduled['jobs'][0]['config_jobs']}, {case_id(c) for c in campaign})
        for c,j in zip(campaign, scheduled['jobs'][0]['config_jobs']):
            dims = f"{c['M']} {c['K']} {c['N']}" if c['operation'] in ('gemm','gemv') else f"{c['M']} {c['N']}"
            self.assertIn('-- '+dims+' --op',j['command'])

    def test_logical_gemv_work_and_padding(self):
        c = next(c for c in cases() if c['operation']=='gemv')
        stdout = 'HICREST_RESULT '+json.dumps({'device_time_us':10, 'M':32, 'N':c['N'], 'K':c['K']})
        row = parse_stdout(stdout,case_id(c))
        self.assertEqual(row['M'],1)
        self.assertEqual(row['reported_M'],32)
        self.assertAlmostEqual(row['throughput'],2*c['N']*c['K']/10000)
        self.assertEqual(row['throughput_unit'],'GFLOP/s')

    def test_reject_bad_results_and_ignore_legacy(self):
        tag = case_id(next(cases()))
        self.assertIsNone(parse_stdout('', 'tt_old_square_s4096'))
        for stdout in ('', 'HICREST_RESULT {"device_time_us":0}', 'HICREST_RESULT {"device_time_us":10,"N":1}'):
            with self.assertRaises(ValueError):
                parse_stdout(stdout,tag)

    def test_nonlinear_work(self):
        c = next(c for c in cases() if c['operation']=='gelu')
        row = parse_stdout('HICREST_RESULT {"device_time_us":10,"M":null}',case_id(c))
        self.assertEqual(row['throughput_unit'],'Gelem/s')
        self.assertAlmostEqual(row['throughput'],c['M']*c['N']/10000)


if __name__ == '__main__':
    unittest.main()
