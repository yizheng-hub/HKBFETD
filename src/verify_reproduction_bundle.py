"""Verify every frozen input and source in the release archives without extraction."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'v2'


def archive_hash(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((OUT / 'pipeline_run_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(OUT / 'reproduction_inputs.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate archived input'
        assert set(archive.namelist()) == set(manifest['inputs']), 'Input inventory differs'
        for name, expected in manifest['inputs'].items():
            assert archive.getinfo(name).file_size == expected['bytes'], name
            assert archive_hash(archive, name) == expected['sha256'], name
    source_files = list((ROOT / 'src').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    source_files += [p for p in (ROOT / 'ctl').glob('*') if p.is_file()]
    source_files += [ROOT / name for name in ['README.md', 'LICENSE', 'environment.yml', 'requirements.txt']]
    with zipfile.ZipFile(OUT / 'v2_source_snapshot.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate archived source'
        for path in source_files:
            assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), str(path)
    result = {'frozen_inputs_verified': len(manifest['inputs']), 'source_files_verified': len(source_files),
              'archive_crc_checked_by_streaming_reads': True, 'source_matches_current_checkout': True}
    (OUT / 'reproduction_bundle_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
