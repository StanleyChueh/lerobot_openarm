# v4B Task 1 - measured augmentation classes of accepted v3.1 G+ candidates

| v3 family | position_only | handover_only | both | other |
|---|---|---|---|---|
| A_original_geometry | 0 | 69 | 0 | 17 |
| B_original_contact | 0 | 43 | 0 | 0 |
| C_new_initial_position | 68 | 0 | 0 | 0 |
| D_new_position_plus_handover | 0 | 0 | 23 | 0 |

In the frozen `sfa_v31_Gp_matched_evidence_complete` dataset, by source episode:

| class | ep0 | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| position_only | 5 | 5 | 8 | 8 | 0 | 3 | 0 | 5 | 6 | 6 | 46 |
| handover_only | 4 | 4 | 3 | 3 | 1 | 4 | 0 | 4 | 2 | 2 | 27 |
| both | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 6 |
| other | 2 | 1 | 0 | 2 | 0 | 2 | 0 | 2 | 2 | 2 | 13 |

`other` candidates (not used):

- ep0 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0618, handover dq 0.0140, can 0.0 mm, grip shift 0
- ep0 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0283, handover dq 0.0086, can 0.0 mm, grip shift 0
- ep1 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0010, handover dq 0.0217, can 0.0 mm, grip shift 0
- ep3 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0062, handover dq 0.0384, can 0.0 mm, grip shift 0
- ep3 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0033, handover dq 0.0273, can 0.0 mm, grip shift 0
- ep4 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0634, handover dq 0.0159, can 0.0 mm, grip shift 0
- ep4 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0318, handover dq 0.0101, can 0.0 mm, grip shift 0
- ep5 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0063, handover dq 0.0279, can 0.0 mm, grip shift 0
- ep5 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0032, handover dq 0.0195, can 0.0 mm, grip shift 0
- ep6 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0509, handover dq 0.0141, can 0.0 mm, grip shift 0
- ep6 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0253, handover dq 0.0103, can 0.0 mm, grip shift 0
- ep7 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0466, handover dq 0.0131, can 0.0 mm, grip shift 0
- ep7 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0223, handover dq 0.0085, can 0.0 mm, grip shift 0
- ep8 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0645, handover dq 0.0152, can 0.0 mm, grip shift 0
- ep8 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0317, handover dq 0.0108, can 0.0 mm, grip shift 0
- ep9 approach_above__G+__s1.0: init 0.0 mm, approach dq 0.0769, handover dq 0.0163, can 0.0 mm, grip shift 0
- ep9 approach_side__G+__s1.0: init 0.0 mm, approach dq 0.0384, handover dq 0.0107, can 0.0 mm, grip shift 0

Identity candidate == recorded source state: max |diff| 0.00e+00
