# --
# Download extra in-domain audio for the 10 target species from Xeno-canto.
#
# Purpose: expand the tiny 2.2k-clip train set with audio the *Perch teacher*
# can later label (teacher-labeled distillation, the highest-ceiling lever for a
# small-data regime). We query each species precisely by genus + epithet and
# write into per-species folders that mirror the dataset's class layout, so the
# existing Perch export (experiments/perch/export_embeddings.py) and the feature
# pipeline can consume the result with no path changes.
#
# Domain caveat: Xeno-canto is mostly *focal* recordings (single bird, close
# mic), while the task is *passive soundscape* -> expect domain shift. The
# species label here is only an organizational prior; the actual training
# targets will come from running Perch over these clips. Format note: XC serves
# mp3 of arbitrary length -> a later step converts to wav and slices to 3 s
# windows before feature extraction (not done here; this script only fetches).
#
# Requires the `xenocanto-api` package and an API key (mandatory since
# 2025-10-10). Put XENO_CANTO_API_KEY in a .env file, or pass --api-key.
#   pip install xenocanto-api          # provides the `xcapi` module
#
# Examples:
#   # preview only (writes metadata_only.csv per species, no audio):
#   python experiments/data/download_xeno_canto.py --metadata-only
#   # download A-quality recordings, capped per species:
#   python experiments/data/download_xeno_canto.py --quality A --max-per-species 200
#   # one species, Spain only, 5-60 s clips:
#   python experiments/data/download_xeno_canto.py --species "Tawny Owl" --country Spain --length 5-60

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.data.species import by_labels, LABELS


def parse_args():
  p = argparse.ArgumentParser(description='Download Xeno-canto audio for the target species.')
  p.add_argument('--out', default='./data/xeno_canto', help='output base dir (per-species subfolders)')
  p.add_argument('--species', nargs='+', default=None, help='subset of labels (default: all 10)')
  p.add_argument('--group', default='birds', help='Xeno-canto taxonomic group')
  p.add_argument('--quality', default='A', help='quality rating, e.g. A or ">C" (XC operator syntax)')
  p.add_argument('--country', default=None, help='restrict to a country (e.g. Spain)')
  p.add_argument('--sound-type', default=None, help='e.g. song, call (XC sound_type)')
  p.add_argument('--length', default=None, help='length filter in seconds, XC syntax (e.g. 5-60, ">10")')
  p.add_argument('--max-per-species', type=int, default=0, help='cap recordings per species (0 = no cap)')
  p.add_argument('--metadata-only', action='store_true', help='preview only: write metadata_only.csv, no audio')
  p.add_argument('--redownload', action='store_true', help='re-fetch even if already downloaded')
  p.add_argument('--api-key', default=None, help='Xeno-canto API key (else read XENO_CANTO_API_KEY / .env)')
  return p.parse_args()


def build_query(QueryBuilder, sp, args):
  """precise per-species query: group + genus + epithet + filters."""
  q = QueryBuilder().group(args.group).genus(sp.genus).species(sp.species)
  if args.quality:    q = q.quality(args.quality)
  if args.country:    q = q.country(args.country)
  if args.sound_type: q = q.sound_type(args.sound_type)
  if args.length:     q = q.length(args.length)
  return q.build()


def main():
  args = parse_args()
  from xcapi.query import QueryBuilder
  from xcapi.client import XenoCantoClient
  from xcapi.downloader import Downloader

  client = XenoCantoClient(api_key=args.api_key) if args.api_key else XenoCantoClient()
  targets = by_labels(args.species)
  out_base = Path(args.out)
  print('Xeno-canto fetch | {} species | quality={} country={} length={} | out={} | {}'.format(
      len(targets), args.quality, args.country, args.length, out_base,
      'METADATA ONLY' if args.metadata_only else 'downloading audio'))

  summary = []
  for sp in targets:
    tag = '{} ({})'.format(sp.label, sp.scientific)
    try:
      # search returns List[Dict]; max_results stops paging early (no post-slice)
      recordings = client.search(build_query(QueryBuilder, sp, args),
                                 max_results=(args.max_per_species or None))
      n = len(recordings)
      # one Downloader per species -> per-class folder mirroring the dataset
      dl = Downloader(output_dir=str(out_base / sp.label))
      if args.metadata_only:
        dl.save_metadata_only(recordings)
        print('  [meta] {:<28} found {}'.format(tag, n))
      else:
        dl.download_recordings(recordings, redownload=args.redownload)
        print('  [ ok ] {:<28} {} recordings'.format(tag, n))
      summary.append((sp.label, n, 'ok'))
    except Exception as e:                                  # don't abort the whole sweep
      print('  [FAIL] {:<28} {}'.format(tag, e))
      summary.append((sp.label, None, 'FAIL: {}'.format(e)))

  ok = [s for s in summary if s[2] == 'ok']
  total = sum(s[1] for s in ok if s[1] is not None)
  print('\nDone: {}/{} species ok | ~{} recordings under {}'.format(len(ok), len(targets), total, out_base))
  for label, n, status in summary:
    if status != 'ok': print('  {} -> {}'.format(label, status))


if __name__ == '__main__':
  main()
