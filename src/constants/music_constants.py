ALLOWED_SIGNATURES = [3, 4]

CHROMATIC_SCALE_SIZE = 12

NOTE_INDEX_SHIFT = 3

TICKS_PER_BEAT = 48

BASE_TIME_SIGNATURE = 4

GLOBAL_LENGHT_LIMIT = 1728 # 36 тактов в 4/4, 48 тактов в 3/4

GENRES = ['blues', 'metal', 'rock']

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MODE_INTERVALS = {
    '': [4, 3],
    'm': [3, 4],
    'dim': [3, 3],
    'aug': [4, 4],
    'sus2': [2, 5],
    'sus4': [5, 2],
    '5': [7],
    '6': [4, 3, 2],
    'm6': [3, 4, 2],
    '7': [4, 3, 3],
    'm7': [3, 4, 3],
    'maj7': [4, 3, 4],
    'mmaj7': [3, 4, 4]
}

FLAT_TO_SHARP = {
    'Db': 'C#',
    'Eb': 'D#',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#'
}

NOTE_STATES = {
     "pause": 0,
     "attack": 1,
     "slide": 2,
     "hammer-on": 3,
     "pull-off": 4,
     "bend": 5,
     "sustain": 6
}