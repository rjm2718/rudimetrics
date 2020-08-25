DROP TABLE IF EXISTS samples;
DROP TABLE IF EXISTS backgrounds;
DROP TABLE IF EXISTS labels;


-- label == category
CREATE TABLE IF NOT EXISTS labels (
  label char(4) primary key,
  description varchar(255)
);

-- labeled training/test examples.  All files references here are already
-- converted to 44.1khz, 24bit float. Onset starts at begining of sample.
-- relative file paths.
CREATE TABLE IF NOT EXISTS samples (
  id int auto_increment primary key,
  filepath varchar(255) not null,
  label char(4) not null,
  channels integer not null default 1,
  length_sec float not null,
  added timestamp default now(),
  unique (filepath),
  foreign key (label) references labels(label)
);

CREATE TABLE IF NOT EXISTS backgrounds (
  id int auto_increment primary key,
  filepath varchar(255) not null,
  label char(2) not null,
  length_sec float not null,
  added timestamp default now(),
  unique (filepath),
  foreign key (label) references labels(label)
);



-- label codes:
--  c?, s?, k?, t?, p?  (cymbal, snare, kick, tom, pad, etc etc etc)
--  x?, y?, z? etc etc ... 
--
-- n_?? generic label for negatives, but otherwise up to application to
-- choose which are positive or negative samples
insert into labels values ('pd1', 'practice pad, short not-very-loud thuds');
insert into labels values ('pd2', 'practice pad, more percussive (or nylon tip)');
insert into labels values ('snr', 'snares');
insert into labels values ('clk', 'clicks');
insert into labels values ('tnc', 'tonal clicks/chirps');
insert into labels values ('kck', 'kick drums');
insert into labels values ('tml', 'tom - low');
insert into labels values ('tmm', 'tom - medium');
insert into labels values ('tmh', 'tom - high');
insert into labels values ('hhc', 'high-hat closed');

-- negatives
insert into labels values ('n_xx', '');
