# PyTorch Iteration Notes

## 2026-05-25 PFM PyTorch Port

- Implemented a PyTorch equivalent of the current LibTorch PFM model:
  `Backbone`, `SparseHead`, `DenseHead`, `DescriptorMatcher`, and
  `PlanetaryGraphMatcher`.
- Added strict loading from the current LibTorch checkpoint state dict.
- Added PyTorch-state save/load for fast resume.
- Added descriptor-map fine-tuning from synthetic `warp_a_to_b` correspondences.
- Found and fixed a PyTorch-side training blocker: learned sparse descriptors
  overflowed to an infinite channel norm and normalized to all-zero vectors.
  Stable channel normalization now rescales by channel max before L2 normalize.

## Held-Out Descriptor Retrieval

Evaluation set: deterministic tail split, 16 held-out cached synthetic pairs from
`img/Rotate` + `img/CompoundViewpoint`, 128 sampled correspondences per pair.

| state | loss | top1 | top5 | top10 | mean rank | pos | neg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable_libtorch_baseline | 4.707478 | 0.1196 | 0.3110 | 0.4204 | 27.83 | 0.974736 | 0.946956 |
| teacher_500 | 4.415898 | 0.1313 | 0.3574 | 0.5093 | 18.48 | 0.974263 | 0.908605 |
| teacher025_probe | 4.254856 | 0.1265 | 0.3340 | 0.4761 | 21.19 | 0.961782 | 0.843237 |
| teacher1_1000 | 4.242686 | 0.1270 | 0.3335 | 0.4766 | 20.71 | 0.962153 | 0.845631 |

Current best for matching candidates: `teacher_500`, because it improves top-k
retrieval and mean rank the most.

Checkpoint:
`runs/pytorch_pfm_finetune_2026-05-25_stable_teacher_500step/pytorch_pfm_state.pt`
