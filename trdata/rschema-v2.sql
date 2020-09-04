-- sqlite3

DROP TABLE IF EXISTS samples;
DROP TABLE IF EXISTS backgrounds;

-- labels: comma sep list of whatever labels you want to apply (be as granular
-- as possible to assign many labels).  negative samples prefix with n_.
-- hierarchy of categories, start out general and append more specific.

-- pad
-- tom
-- snare
-- kick
-- click (sharp percussive, non-tonal spike, e.g. metronome click or nylon tip against table)
-- chirp (tonal click, e.g. metronome, clave, or short percussive note from any instrument)
-- bell
-- ride
-- crash
-- hat

-- n_foo
-- n_say_hi
-- n_etc

-- important note: all samples have percussive onset at t=0 in file.
-- filepath is relative to $REPO_BASE, same dir where repo.db lives.


-- training/test examples
CREATE TABLE samples (  -- use sqlite3 rowid instead of explicit primary key
  filepath text not null,
  labels text not null, -- comma separated
  effects boolean not null default FALSE, -- apply effects or not (set to 0 for phone-mic'd acoustic samples)
  unique (filepath)
);

CREATE TABLE backgrounds (
  filepath text not null,
  labels text not null, -- comma separated
  effects boolean not null default FALSE, -- apply effects or not (set to 0 for phone-mic'd acoustic samples)
  unique (filepath)
);

