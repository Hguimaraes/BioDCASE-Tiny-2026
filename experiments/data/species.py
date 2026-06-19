# --
# Canonical species table for the BioDCASE-Tiny 2026 task (the 10 target birds).
#
# Shared by every external-audio collector (Xeno-canto now, iNaturalist next) so
# folder names, scientific names and English names stay consistent across sources
# and line up with the dataset's class folders. `label` is the dataset class
# folder name verbatim -> downloaded audio drops straight into the existing
# Perch export / feature pipeline (class-subfolder layout).
#
# `Background` (class 0) is intentionally absent: it is ambient/noise, not a
# species, and is sourced separately (soundscapes), not from species queries.

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
  label: str        # dataset class folder name (English common name as labelled)
  genus: str        # scientific genus       -> Xeno-canto gen:
  species: str      # scientific species epithet -> Xeno-canto sp:
  english: str      # Xeno-canto English name (for english_name() queries / iNat)

  @property
  def scientific(self):
    return '{} {}'.format(self.genus, self.species)


# 10 target species (label == dataset folder name exactly).
SPECIES = [
  Species('Common Chaffinch',         'Fringilla',   'coelebs',      'Common Chaffinch'),
  Species('Common Chiffchaff',        'Phylloscopus','collybita',    'Common Chiffchaff'),
  Species('Eurasian Blackbird',       'Turdus',      'merula',       'Eurasian Blackbird'),
  Species('Eurasian Blackcap',        'Sylvia',      'atricapilla',  'Eurasian Blackcap'),
  Species('Eurasian Blue Tit',        'Cyanistes',   'caeruleus',    'Eurasian Blue Tit'),
  Species('Great Spotted Woodpecker', 'Dendrocopos', 'major',        'Great Spotted Woodpecker'),
  Species('Great Tit',                'Parus',       'major',        'Great Tit'),
  Species('Mallard',                  'Anas',        'platyrhynchos','Mallard'),
  Species('Song Thrush',              'Turdus',      'philomelos',   'Song Thrush'),
  Species('Tawny Owl',                'Strix',       'aluco',        'Tawny Owl'),
]

LABELS = [s.label for s in SPECIES]


def by_labels(labels=None):
  """subset of SPECIES by label (None -> all). raises on an unknown label."""
  if not labels:
    return list(SPECIES)
  known = {s.label: s for s in SPECIES}
  missing = [l for l in labels if l not in known]
  if missing:
    raise ValueError('unknown species label(s): {} | known: {}'.format(missing, LABELS))
  return [known[l] for l in labels]
