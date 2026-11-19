#!/bin/sh
# basic checks of `lgor defects` on small models (run from lgor_v2.3: sh examples/defects/run_all.sh)
T=""
command -v timeout >/dev/null 2>&1 && T="timeout 2400"
for f in Q1Fermat2 Q1Mirror G44441 G66222 A19Z33Mirror20 G4_2_2_2_2_1 G2_2_2_2_2_2 G2_2_2_2_1_1_1; do
  start=$(date +%s)
  $T python3 -m lgor defects examples/defects/$f.lgo -j 2 > examples/defects/$f.log 2>&1
  echo "$f exit=$? $(( $(date +%s) - start ))s"
done
