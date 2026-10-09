# HBM-Usage(MB) "used" of one Ascend chip, from an `npu-smi info` table on stdin.
#
#   npu-smi info | awk -v id=0 -f scripts/ascend/npu_hbm_used_mib.awk
#
# Prints the MiB in use on NPU `id`, or NOTHING AT ALL on any table shape it does
# not recognise, so the caller can refuse to launch instead of believing a number.
# Printing 0 on a layout change is the one outcome that must never happen: the
# launcher's "only take free cards" guard then passes on a full card.
#
# The layout this reads (npu-smi 24.1.0.3).  An NPU row carries the id and the chip
# name; the chip row below it carries the memory, and its last pipe-delimited field
# holds TWO "used / total" pairs -- Memory-Usage first, HBM second:
#
#   | 0     910B3               | OK            | 113.7       33                0    / 0             |
#   | 0                         | 0000:C1:00.0  | 0           0    / 0          65521/ 65536         |
#
# The earlier version of this parser split that field on its first "/" and so read
# the Memory-Usage pair, which is "0 / 0" on this driver: every card reported 0 MiB
# in use, including one holding 65521 MiB.  (The version before that concatenated
# every digit on the row and reported an idle card as 100000 MiB.)  Hence the shape
# checks below: this file would rather say nothing than be wrong a third time.

# The header fixes which of the two pairs is HBM.  Checking only that the string
# "HBM-Usage(MB)" appears somewhere would still read the first pair, so a driver that
# printed the columns the other way round would hand back Memory-Usage -- 0 on this
# driver -- for a full card.
/HBM-Usage\(MB\)/ {
  memory_at = index($0, "Memory-Usage(MB)")
  hbm_at = index($0, "HBM-Usage(MB)")
  if (memory_at > 0 && hbm_at > memory_at) { header = 1 } else { bad = 1; exit }
}

# The per-process table further down repeats chip ids; nothing after it is a chip row.
/Process id/ { exit }

# The chip row of the NPU we are looking for.
want == 1 {
  # It must look like one: chip index in $2, Bus-Id in $4.
  if ($2 !~ /^[0-9]+$/ || $4 !~ /^[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+$/) { bad = 1; exit }
  count = split($0, field, "|")
  segment = field[count - 1]
  pairs = 0
  while (match(segment, /[0-9]+ *\/ *[0-9]+/)) {
    pair[++pairs] = substr(segment, RSTART, RLENGTH)
    segment = substr(segment, RSTART + RLENGTH)
  }
  # Exactly Memory-Usage and HBM-Usage.  One pair means the columns moved.
  if (pairs != 2) { bad = 1; exit }
  split(pair[2], hbm, "/")
  gsub(/ /, "", hbm[1])
  gsub(/ /, "", hbm[2])
  if (hbm[1] !~ /^[0-9]+$/ || hbm[2] !~ /^[0-9]+$/) { bad = 1; exit }
  # A total of 0, or more in use than exists, means this is not the HBM pair.
  if (hbm[2] + 0 <= 0 || hbm[1] + 0 > hbm[2] + 0) { bad = 1; exit }
  used = hbm[1]
  want = 2
  next
}

# A multi-chip NPU has a second chip row, and this parser only knows how to read one.
want == 2 {
  if ($2 ~ /^[0-9]+$/ && $4 ~ /:/) { bad = 1; exit }
  want = 3
}

# String comparison, not awk's numeric one for two strnum operands: "00" is then
# not card 0.  The launcher refuses such an id before calling this, but anyone
# running the parser by hand gets the same answer rather than a silent canonicalisation.
$1 == "|" && ($2 "") == (id "") && $3 ~ /910/ {
  if (seen++) { bad = 1; exit }   # the same id twice: not a table we understand
  want = 1
  next
}

END {
  if (!bad && header && want >= 2 && used ~ /^[0-9]+$/) {
    print used + 0
  }
}
