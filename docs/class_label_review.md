# Class Label Review

This file records the observed Labelme labels from `SM_norm` and `NZB_dataset`.
The current `configs/class_map.sm_nzb.json` keeps pinyin labels unchanged to avoid incorrect class merging.

Before paper-grade experiments, confirm the English display name, whether any labels should be merged, and whether the class belongs to the same defect taxonomy.

| Raw Label | Current Canonical Class | Count | Source Dataset | Needs Review |
| --- | --- | ---: | --- | --- |
| feigen | feigen | 1721 | SM_norm | yes |
| duanzhen | duanzhen | 473 | SM_norm | yes |
| songdiaojing | songdiaojing | 431 | NZB_dataset | yes |
| daisha | daisha | 212 | NZB_dataset | yes |
| bubian | bubian | 178 | NZB_dataset | yes |
| duanjing | duanjing | 74 | NZB_dataset | yes |
| weisuo | weisuo | 39 | NZB_dataset | yes |
| zhifeihua | zhifeihua | 39 | NZB_dataset | yes |
| shuangjing | shuangjing | 20 | NZB_dataset | yes |
| milu | milu | 17 | NZB_dataset | yes |
| quewei | quewei | 16 | NZB_dataset | yes |
| fumao | fumao | 12 | NZB_dataset | yes |
| podong | podong | 12 | NZB_dataset | yes |
| suoquan | suoquan | 1 | NZB_dataset | yes |
| youwu | youwu | 1 | NZB_dataset | yes |

## Review Questions

1. Are `duanzhen` and `duanjing` separate defect classes, or should they be merged?
2. Should rare classes such as `suoquan` and `youwu` remain as independent classes, be merged into a parent class, or be excluded from training until more samples are available?
3. Which English class names should appear in the paper tables and prompts?
4. Which classes are visually confusable with each other?
5. For each class, what are the domain causes, treatment actions, and risk notes?
