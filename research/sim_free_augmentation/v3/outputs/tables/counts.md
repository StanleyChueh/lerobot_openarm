| family | condition | variations x episodes | accepted (own scale) | at full scale | export checks pass | matched pairs (both conditions, common scale, both pass) |
|---|---|---|---|---|---|---|
| A_original_geometry | G+ | 90 | 81 | 72 | 45 |  |
| A_original_geometry | C | 90 | 82 | 70 | 53 | 46 |
| B_original_contact | G+ | 50 | 43 | 25 | 20 |  |
| B_original_contact | C | 50 | 45 | 36 | 18 | 10 |
| C_new_initial_position | G+ | 80 | 68 | 54 | 49 |  |
| C_new_initial_position | C | 80 | 69 | 54 | 51 | 50 |
| D_new_position_plus_handover | G+ | 30 | 21 | 16 | 8 |  |
| D_new_position_plus_handover | C | 30 | 20 | 16 | 8 | 7 |
| all | both | 250 per condition | G+ 213, C 216 | G+ 167, C 176 | G+ 122, C 130 | 113 |

| condition | check failing on a never-accepted case | count |
|---|---|---|
| G+ | can_penetration | 19 |
| C | can_penetration | 19 |
| G+ | joint_speed | 17 |
| C | joint_speed | 14 |
| G+ | joint_limits | 13 |
| C | joint_limits | 11 |
| C | labels | 11 |
| G+ | targets | 10 |
| C | targets | 10 |
| G+ | ik | 9 |
| C | ik | 9 |
| G+ | labels | 8 |
| G+ | attach6d_translation | 8 |
| G+ | grasp_feasibility_R | 8 |
| C | attach6d_translation | 8 |
| C | grasp_feasibility_R | 8 |
| G+ | joint_accel | 6 |
| C | joint_accel | 6 |
| G+ | attach6d_tilt | 3 |
| G+ | table_clearance | 3 |
| C | table_clearance | 3 |
| C | attach6d_tilt | 2 |
| G+ | action_jump | 1 |
| C | action_jump | 1 |

| condition | export-time failure among accepted candidates | count |
|---|---|---|
| G+ | image_qa | 65 |
| C | image_qa | 63 |
| G+ | receiver image lag -0.75 | 19 |
| C | receiver image lag -0.75 | 16 |
| C | receiver image lag -1.0 | 5 |
| G+ | receiver image lag -1.0 | 2 |
| G+ | giver image lag -8.0 | 2 |
| C | giver image lag -8.0 | 2 |
| G+ | receiver image lag -0.5 | 2 |
| C | giver image lag 0.0 | 1 |
| G+ | receiver image lag 5.25 | 1 |
| G+ | giver image lag -0.5 | 1 |
| C | giver image lag -0.5 | 1 |
| G+ | receiver image lag 0.25 | 1 |
| C | receiver image lag -0.5 | 1 |
