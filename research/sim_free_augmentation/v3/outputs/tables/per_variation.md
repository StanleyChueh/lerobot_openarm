| variation | family | ep0 | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pos_right_up | A | a/a | a/a | ✗/a | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✗/✗ | a/a | a/a |
| pos_left_down | A | a/a | a/✓ | a/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/✓ |
| pos_forward | A | a/a | a/a | a/a | a/a | ✓/✓ | ✓/✓ | a/a | a/a | a/a | a/a |
| approach_above | A | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✗/✗ | ✓/✓ | ✓/✓ |
| approach_side | A | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| orient_tilt20 | A | a/a | a/✓ | a/a | a/a | a/a | ✓/✓ | ✓/✓ | ✗/✗ | a/a | a/a |
| orient_yaw30 | A | a/a | a/✓ | ✗/✗ | a/✓ | a/a | ✓/✓ | a/✓ | ✓/✓ | a/✓ | a/a |
| timing_early | A | ✓/✓ | ✓/✓ | ✗/✗ | ✓/✓ | a/a | ✗/✗ | ✓/✓ | ✓/✓ | ✗/✗ | ✓/✓ |
| timing_late | A | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✗/✗ | ✓/✓ |
| grasp_spin30 | B | a/✓ | a/✓ | a/✓ | a/✓ | a/a | a/✓ | ✓/✓ | ✗/✓ | a/✓ | ✗/✓ |
| grasp_higher | B | ✓/✓ | ✓/✓ | ✓/✗ | ✓/✓ | a/✗ | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | ✓/a |
| rot_handover_nominal | B | a/a | a/a | a/a | a/a | ✓/a | ✓/a | ✓/✓ | ✓/a | a/a | a/a |
| rot_handover_early6 | B | a/a | a/a | ✗/✗ | ✗/a | ✓/a | ✗/✗ | a/a | ✓/a | ✗/✗ | a/a |
| rot_handover_late6 | B | a/a | a/a | a/a | ✗/a | ✓/a | ✓/a | ✓/✓ | ✓/a | a/a | a/a |
| init_x+3 | C | a/a | a/a | ✓/✓ | ✓/✓ | a/a | ✗/✗ | a/a | a/a | ✓/✓ | ✓/✓ |
| init_x-3 | C | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| init_y+3 | C | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✗/✗ | ✗/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| init_y-3 | C | ✓/✓ | a/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| init_x+6 | C | a/a | a/a | ✓/✓ | ✓/✓ | ✗/✗ | ✗/✗ | a/a | ✗/✗ | ✓/✓ | ✗/✗ |
| init_x-6 | C | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ | ✓/✓ | ✓/✓ | a/a | ✓/✓ |
| init_y+6 | C | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ | ✓/✓ | ✓/✓ |
| init_y-6 | C | a/a | ✗/✗ | ✓/✓ | ✓/✓ | a/a | a/a | ✓/✓ | ✓/✓ | a/a | a/a |
| init_x+5_pos_right_up | D | a/a | a/a | ✗/a | a/a | ✗/✗ | ✗/✗ | a/a | ✗/✗ | a/a | ✗/✗ |
| init_y+5_grasp_higher | D | ✓/✓ | ✓/✓ | ✓/✗ | ✓/✓ | a/✗ | ✗/✗ | ✗/✗ | ✗/✗ | ✓/✓ | ✓/✓ |
| init_x-5_orient_tilt20 | D | a/a | a/✓ | a/a | a/a | a/a | ✓/✓ | ✓/✓ | ✗/✗ | a/a | a/a |

G+/C per cell: ✓ accepted and passes every export check, a = accepted but fails an export check (image QA / image lag), ✗ rejected at every scale.
