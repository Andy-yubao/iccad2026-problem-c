#!/usr/bin/env bash
# 2026-11-11 experiments consolidation: same-volume rename of 29 retired
# experiment directories into history/20261111_experiments_consolidation.
# E119/E120, README.md and __init__.py stay; __pycache__ is deleted separately.
set -u
SRC="/e/project/ICCAD/2026/problemC/experiments"
DST="/e/project/ICCAD/2026/problemC/history/20261111_experiments_consolidation/experiments"
mkdir -p "$DST" || exit 1
DIRS="E98_loss_weight_learning
E101_final_c5_solver
E102_reference_image_generation
E103_offline_loss_alignment
E104_scaling32k
E105_gated_prenorm_placer
E106_postprocessor_aware_loss
E107_geometry_posttrain
E108_random_reference40k
E109_deep_supervision_shared
E110_deep_supervision_heads
E111_progressive_depth
E112_deep_geometry_budget
E113_deep_head_budget
E114_supervision_placement
E115_first_head_supervision
E116_first_draft_comparison
E117_node26
E118_first12_supervision
E121_residual_update
E122_global2_scaling16
E123_node12_scaling16
E124_residual_scaling16
E125_global2_ref04
E126_global2_refsplit
E127_node12_ref04
E128_node12_refsplit
E129_global2_s0ref
E130_node12_s0ref"
fail=0
moved=0
for d in $DIRS; do
  if [ ! -d "$SRC/$d" ]; then
    echo "MISSING_SRC	$d"
    fail=$((fail+1))
    continue
  fi
  if [ -e "$DST/$d" ]; then
    echo "DST_EXISTS	$d"
    fail=$((fail+1))
    continue
  fi
  if mv "$SRC/$d" "$DST/$d"; then
    if [ -d "$DST/$d" ] && [ ! -e "$SRC/$d" ]; then
      echo "MOVED	$d"
      moved=$((moved+1))
    else
      echo "MOVE_UNVERIFIED	$d"
      fail=$((fail+1))
    fi
  else
    echo "MOVE_FAILED	$d"
    fail=$((fail+1))
  fi
done
echo "SUMMARY	moved=$moved	failed=$fail"
exit 0
