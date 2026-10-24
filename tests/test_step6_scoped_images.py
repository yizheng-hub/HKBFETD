import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('step6_candidate', SRC / '6.geometry_ai_arbitration.py')
step6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step6)


def frame():
    return gpd.GeoDataFrame([
        {'BUILDINGSTRUCTUREID':'1','Final_Main_Class':'Unknown_未知类别','Final_Sub_Class':'Unknown_未知类别','geometry':box(0,0,10,10)},
        {'BUILDINGSTRUCTUREID':'2','Final_Main_Class':'Commercial_商业类别','Final_Sub_Class':'Office_办公室','geometry':box(10,0,20,10)},
        {'BUILDINGSTRUCTUREID':'3','Final_Main_Class':'Commercial_商业类别','Final_Sub_Class':'Retail_零售','geometry':box(20,0,30,10)},
    ], crs='EPSG:2326').set_index('BUILDINGSTRUCTUREID')


def pairs(indexed, image_root):
    rows = []
    for host_id in ('2','3'):
        fragment, host = indexed.loc['1'], indexed.loc[host_id]
        context = step6.context_hash(fragment, host)
        _, render_context = step6.geometry_render_context(fragment, host, indexed.crs)
        base = pd.Series({'fragment_id':'1','host_id':host_id,'Context_SHA256':context,
                          'Render_Context_SHA256':render_context})
        rows.append({'fragment_id':'1','host_id':host_id,'Context_SHA256':context,
            'Render_Context_SHA256':render_context,'distance_m':0.0,
            **step6.isolated_image_paths(base, image_root)})
    return pd.DataFrame(rows)


def fake_renderer(pair, indexed):
    Path(pair.plain_image_path).write_bytes(b'plain-image:' + str(pair.Context_SHA256).encode())
    Path(pair.overlay_image_path).write_bytes(b'overlay-image:' + str(pair.Render_Context_SHA256).encode())
    context = step6.image_context_payload(pair, indexed)
    return {'acquired_at_utc':'2026-10-24T00:00:00+00:00','provider':step6.IMAGE_PROVIDER,
        'zoom':step6.IMAGE_ZOOM,'crs':str(indexed.crs),'tile_bounds':context['tile_bounds'],
        'fragment_geometry_sha256':context['fragment_geometry_sha256'],
        'host_geometry_sha256':context['host_geometry_sha256'],
        'render_context_sha256':pair.Render_Context_SHA256}


class ScopedImageTests(unittest.TestCase):
    def test_images_require_explicit_scope_before_pipeline_io(self):
        called = []
        with patch.object(step6, 'read_table', side_effect=lambda path: called.append(path)):
            with self.assertRaisesRegex(ValueError, 'requires --scope-id-file or --pair-whitelist'):
                step6.main(images=True)
        self.assertEqual(called, [])

    def test_images_require_isolated_run_root(self):
        called = []
        with patch.object(step6, 'BASE_DIR', r'D:\project'), \
             patch.object(step6, 'RUN_ROOT', r'D:\project'), \
             patch.object(step6, 'read_table', side_effect=lambda path: called.append(path)):
            with self.assertRaisesRegex(ValueError, 'isolated run root'):
                step6.main(images=True, scope_id_file='scope.csv')
        self.assertEqual(called, [])

    def test_default_candidate_inventory_does_not_recover_or_create_image_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / 'run'
            archive_root = root / 'frozen_output'
            legacy_data = root / 'data' / 'evidence' / 'legacy_vision_images'
            image_root = run_root / 'intermediate_files' / 'v2' / 'geometry_correction_candidates'
            with patch.object(step6, 'DATA_DIR', str(root / 'data')), \
                 patch.object(step6, 'RUN_ROOT', str(run_root)), \
                 patch.object(step6, 'INTERMEDIATE_DIR', str(run_root / 'intermediate_files' / 'v2')), \
                 patch.object(step6, 'LEGACY_OUTPUT_DIR', str(archive_root)):
                candidates = step6.generate_candidates(frame().reset_index(), image_root=image_root, isolated=False)
            self.assertEqual(len(candidates), 1)
            self.assertFalse(legacy_data.exists())
            self.assertFalse((run_root / 'intermediate_files' / 'v2' / 'legacy_vision_images').exists())
            self.assertFalse(image_root.exists())
            self.assertTrue(candidates.iloc[0].plain_image_path.startswith(str(image_root)))

    def test_fresh_run_root_recovers_zip_without_scoped_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / 'fresh_run'
            data_root = root / 'protected_data'
            frozen_output = root / 'frozen_output'
            frozen_output.mkdir()
            archive = frozen_output / 'Validation_Materials.zip'
            payloads = {}
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zipped:
                for number in range(43):
                    pair = f'{1000+number}_vs_{2000+number}'
                    for kind in ('plain','overlay'):
                        name = f'Validation_Materials/Vision_Check/{pair}_{kind}.png'
                        payload = f'historical-pixels-{pair}-{kind}'.encode()
                        payloads[Path(name).name] = payload
                        zipped.writestr(name, payload)
            original_zip_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            protected_dir = data_root / 'evidence' / 'legacy_vision_images'
            protected_dir.mkdir(parents=True)
            protected_file = protected_dir / 'do_not_touch.png'
            protected_file.write_bytes(b'protected')
            protected_sha = hashlib.sha256(protected_file.read_bytes()).hexdigest()
            intermediate = run_root / 'intermediate_files' / 'v2'
            image_root = intermediate / 'geometry_correction_candidates'
            with patch.object(step6, 'RUN_ROOT', str(run_root)), \
                 patch.object(step6, 'DATA_DIR', str(data_root)), \
                 patch.object(step6, 'INTERMEDIATE_DIR', str(intermediate)), \
                 patch.object(step6, 'LEGACY_OUTPUT_DIR', str(frozen_output)):
                recovered = step6.recover_validation_images()
                images = list(recovered.glob('*.png'))
                self.assertEqual(len(images), 86)
                self.assertEqual(len({p.name.rsplit('_',1)[0] for p in images}), 43)
                self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), original_zip_sha)
                self.assertEqual(hashlib.sha256(protected_file.read_bytes()).hexdigest(), protected_sha)
                self.assertEqual(protected_file.read_bytes(), b'protected')
                manifest = json.loads(next(recovered.glob('recovery_manifest_*.json')).read_text(encoding='utf-8'))
                self.assertEqual(len(manifest['entries']), 86)
                self.assertTrue(all(row['archive_sha256'] == original_zip_sha for row in manifest['entries']))
                self.assertTrue(all(row['context_status'] == 'context_unverified_historical_only' for row in manifest['entries']))
                for name, payload in payloads.items():
                    self.assertEqual(hashlib.sha256((recovered / name).read_bytes()).hexdigest(), hashlib.sha256(payload).hexdigest())

                # A matching historical basename does not become a current scoped image.
                (recovered / '1_vs_2_plain.png').write_bytes(b'old-basename-only')
                (recovered / '1_vs_2_overlay.png').write_bytes(b'old-basename-only-overlay')
                candidates = step6.generate_candidates(frame().reset_index(), image_root=image_root, isolated=True)
                row = candidates.iloc[0]
                self.assertNotEqual(Path(row.plain_image_path).name, '1_vs_2_plain.png')
                self.assertTrue(Path(row.plain_image_path).is_relative_to(run_root))
                self.assertFalse(Path(row.image_context_path).exists())
                self.assertFalse(step6.sidecar_is_valid(row, frame()))

    def test_pair_whitelist_filters_before_generation_and_binds_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            indexed = frame()
            candidates = pairs(indexed, Path(temporary))
            chosen = candidates.iloc[[1]].copy()
            whitelist = [chosen.iloc[0].to_dict()]
            filtered = step6.filter_candidates(candidates, pair_whitelist=whitelist)
            self.assertEqual(filtered[['fragment_id','host_id']].values.tolist(), [['1','3']])
            with self.assertRaisesRegex(ValueError, 'Context_SHA256 mismatch'):
                step6.filter_candidates(candidates, pair_whitelist=[{**whitelist[0], 'Context_SHA256':'0'*64}])

    def test_generation_limit_counts_failed_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            indexed = frame()
            candidates = pairs(indexed, Path(temporary))
            calls = []
            def failing_renderer(pair, unused_frame):
                calls.append((pair.fragment_id, pair.host_id))
                raise RuntimeError('mock tile failure')
            result, attempts = step6.run_scoped_image_jobs(candidates, indexed, Path(temporary),
                image_limit=1, renderer=failing_renderer)
            self.assertEqual(attempts, 1)
            self.assertEqual(calls, [('1','2')])
            self.assertEqual(result.Image_Generation_Status.tolist(), ['failed_RuntimeError','limit_not_attempted'])

    def test_sidecar_reuse_geometry_and_text_context_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexed = frame()
            candidate = pairs(indexed, root).iloc[0]
            first, created = step6.ensure_isolated_images(candidate, indexed, root, renderer=fake_renderer)
            self.assertTrue(created)
            self.assertTrue(step6.sidecar_is_valid(first, indexed))
            original_paths = (first.plain_image_path, first.overlay_image_path)
            again, created = step6.ensure_isolated_images(first, indexed, root,
                renderer=lambda *_: self.fail('valid sidecar must reuse'))
            self.assertFalse(created)
            self.assertEqual((again.plain_image_path, again.overlay_image_path), original_paths)

            changed = indexed.copy()
            changed.loc['2','Final_Sub_Class'] = 'Different function'
            fragment, host = changed.loc['1'], changed.loc['2']
            new_context = step6.context_hash(fragment, host)
            _, same_render = step6.geometry_render_context(fragment, host, changed.crs)
            changed_pair = candidate.copy()
            changed_pair['Context_SHA256'] = new_context
            changed_pair['Render_Context_SHA256'] = same_render
            changed_pair = changed_pair.drop(labels=['plain_image_path','overlay_image_path','image_context_path'])
            for key, value in step6.isolated_image_paths(changed_pair, root).items():
                changed_pair[key] = value
            updated, created = step6.ensure_isolated_images(changed_pair, changed, root, renderer=fake_renderer)
            self.assertTrue(created)
            self.assertNotEqual(updated.Context_SHA256, first.Context_SHA256)
            self.assertEqual(updated.Render_Context_SHA256, first.Render_Context_SHA256)
            self.assertTrue(Path(first.plain_image_path).read_bytes().startswith(b'plain-image:'))
            self.assertTrue(Path(updated.image_context_path).is_file())

    def test_mismatched_sidecar_regenerates_once_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexed = frame()
            candidate = pairs(indexed, root).iloc[0]
            first, _ = step6.ensure_isolated_images(candidate, indexed, root, renderer=fake_renderer)
            sidecar = Path(first.image_context_path)
            saved = sidecar.read_text(encoding='utf-8').replace(first.Context_SHA256, 'f'*64)
            sidecar.write_text(saved, encoding='utf-8')
            old = Path(first.plain_image_path).read_bytes()
            calls = []
            def counted(pair, unused_frame):
                calls.append(1)
                return fake_renderer(pair, unused_frame)
            regenerated, created = step6.ensure_isolated_images(first, indexed, root, renderer=counted)
            self.assertTrue(created)
            self.assertEqual(len(calls), 1)
            self.assertNotEqual(regenerated.plain_image_path, first.plain_image_path)
            self.assertEqual(Path(first.plain_image_path).read_bytes(), old)
            self.assertTrue(step6.sidecar_is_valid(regenerated, indexed))
            resumed = step6.generate_candidates(indexed.reset_index(), image_root=root, isolated=True).iloc[0]
            self.assertEqual(resumed.plain_image_path, regenerated.plain_image_path)
            self.assertEqual(resumed.overlay_image_path, regenerated.overlay_image_path)
            self.assertTrue(step6.sidecar_is_valid(resumed, indexed))
            _, created = step6.ensure_isolated_images(resumed, indexed, root,
                renderer=lambda *_: self.fail('resume should reuse the context-matched alternate path'))
            self.assertFalse(created)

    def test_renderer_uses_one_mocked_basemap_capture_for_both_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexed = frame()
            pair = pairs(indexed, root).iloc[0].copy()
            for key, value in step6.isolated_image_paths(pair, root).items():
                pair[key] = value
            calls, saves = [], []
            class FakeAxes:
                def set_xlim(self, *args): pass
                def set_ylim(self, *args): pass
                def axis(self, *args): pass
            class FakeFigure:
                def savefig(self, path, **kwargs):
                    saves.append(str(path))
                    Path(path).write_bytes(f'mock-png-{len(saves)}'.encode())
            fake_plt = types.SimpleNamespace(subplots=lambda **kwargs:(FakeFigure(),FakeAxes()), close=lambda fig:None)
            fake_matplotlib = types.SimpleNamespace(use=lambda backend:None, pyplot=fake_plt)
            with patch.object(step6, '_add_basemap', side_effect=lambda ax, crs, bounds: calls.append((str(crs),tuple(bounds)))), \
                 patch.object(gpd.GeoDataFrame, 'plot', lambda *args, **kwargs:None), \
                 patch.dict(sys.modules, {'matplotlib':fake_matplotlib,'matplotlib.pyplot':fake_plt}):
                result = step6.render_pair_images(pair, indexed)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(saves), 2)
            self.assertTrue(Path(pair.plain_image_path).is_file())
            self.assertTrue(Path(pair.overlay_image_path).is_file())
            self.assertEqual(result['provider'], step6.IMAGE_PROVIDER)
            self.assertEqual(len(result['tile_bounds']), 4)


if __name__ == '__main__':
    unittest.main()
