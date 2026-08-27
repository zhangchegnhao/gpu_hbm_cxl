# cycle-v1固定分界消融

## 实验问题

上一阶段`sieve-cycle-v1`在全部96次层执行中都选择16个GPU专家和33个PIM专家。
本消融比较动态枚举与固定16专家热点前缀，判断当前合成Trace是否真正触发了动态
放置。

两种策略使用相同的精确schema-v3表、相同的20 us Sieve调度开销和相同的端到端
事件图。唯一差异是：

- `sieve-cycle-v1`枚举每层全部热点前缀；
- `sieve-fixed-16-cycle-v1`只读取固定长度16的精确候选。

## 结果

| 策略 | 总时延(ms) | GPU/PIM专家/层 | GPU/PIM token总数 |
|---|---:|---:|---:|
| sieve-fixed-16-cycle-v1 | 10.144665 | 16 / 33 | 2976 / 3168 |
| sieve-cycle-v1 | 10.144665 | 16 / 33 | 2976 / 3168 |

两个策略的96行`layers.csv`逐字节相同。固定候选的单层路径仍为：

```text
GPU path = 47.121984 us memory + 3.130023 us compute = 50.252007 us
PIM path = 50.844768 us
objective = 20 us scheduler + max(path) = 70.844768 us
```

## 解释

当前合成Trace虽然在相邻step大幅轮换专家ID，但全部96个层批次共享同一种负载
计数signature。精确时序和调度目标由活跃专家负载计数决定，因此每层最优前缀均为
16。动态搜索在这个输入上没有产生相对固定分界的收益。

这个结果不能说明动态放置无效，只能说明旧合成Trace无法验证动态放置。下一步必须
使用真实Qwen3 Router Trace，检查不同层和step的负载signature是否变化，并重复同一
固定/动态消融。

结果位于`results/full_decode_cycle_v1_fixed_split_ablation`。
