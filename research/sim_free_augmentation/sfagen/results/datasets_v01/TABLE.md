| Repo (`ethanCSL/...`) | Demos | Episodes / frames | Coverage /40 | max speed / accel ratio | body decode | wrist valid (L / R) | Gate |
|---|---|---|---|---|---|---|---|
| [`openarm_pringles_lerobot_real_to_sim_1_3cam_v01`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_1_3cam_v01) | [3] | 25 / 7,328 | 16 | 1.09 / 1.50 | 40.29 dB | 0.40 / 0.40 | pass |
| [`openarm_pringles_lerobot_real_to_sim_2_3cam_v01`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_2_3cam_v01) | [3, 8] | 51 / 15,934 | 24 | 1.08 / 1.49 | 39.58 dB | 0.36 / 0.36 | pass |
| [`openarm_pringles_lerobot_real_to_sim_4_3cam_v01`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_4_3cam_v01) | [1, 3, 5, 8] | 111 / 32,668 | 27 | 1.02 / 1.50 | 39.85 dB | 0.41 / 0.41 | pass |
| [`openarm_pringles_lerobot_real_to_sim_10_3cam_v01`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v01) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 269 / 84,587 | 31 | 1.23 / 1.50 | 39.83 dB | 0.37 / 0.37 | pass |
| [`openarm_pringles_lerobot_real_to_sim_1_3cam_wristrender_v01`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_1_3cam_wristrender_v01) | [3] | 25 / 7,328 | 16 | 1.09 / 1.50 | 40.29 dB | 1.00 / 0.82 | pass |
| `openarm_pringles_lerobot_real_to_sim_2_3cam_wristrender_v01` (not published) | [3, 8] | 51 / 15,934 | 24 | 1.08 / 1.49 | 39.58 dB | 1.00 / 0.81 | FAIL: wrist_decode_psnr (min 29.1 dB < 30) |
| `openarm_pringles_lerobot_real_to_sim_4_3cam_wristrender_v01` (not published) | [1, 3, 5, 8] | 111 / 32,668 | 27 | 1.02 / 1.50 | 39.85 dB | 1.00 / 0.82 | FAIL: wrist_decode_psnr (min 29.0 dB < 30) |
| `openarm_pringles_lerobot_real_to_sim_10_3cam_wristrender_v01` (not published) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 269 / 84,587 | 31 | 1.23 / 1.50 | 39.83 dB | 1.00 / 0.81 | FAIL: wrist_decode_psnr (min 28.6 dB < 30) |

`_wristrender` = rendered-wrist **ablation** (the registered wrist gate failed for Pringles); otherwise identical to the main repo. Mask wrist cameras where `wrist_view_valid == 0`.