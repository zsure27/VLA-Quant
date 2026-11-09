"""Repair the known Decord 0.6.0 wheel's packaging tag, without changing code.

Does not install anything. Decode compatibility must be tested independently.
"""
import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

SOURCE_SHA256 = '51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA256
    assert not args.destination.exists(), 'Keep previous derived wheels unchanged'
    with zipfile.ZipFile(args.source) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    wheel = 'decord-0.6.0.dist-info/WHEEL'
    record = 'decord-0.6.0.dist-info/RECORD'
    original = dict(data)
    lines = data[wheel].decode().splitlines()
    assert sum(line.startswith('Tag:') for line in lines) == 1
    data[wheel] = ('\n'.join('Tag: py3-none-manylinux2010_x86_64'
                             if line.startswith('Tag:') else line for line in lines) + '\n').encode()
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    for name, body in data.items():
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b'=').decode()
            writer.writerow([name, 'sha256=' + digest, len(body)])
    writer.writerow([record, '', ''])
    data[record] = stream.getvalue().encode()
    assert sorted(name for name in data if data[name] != original[name]) == [record, wheel]
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in data.items():
            archive.writestr(name, body)
    print('PASS_METADATA_ONLY_REPACK', hashlib.sha256(args.destination.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
