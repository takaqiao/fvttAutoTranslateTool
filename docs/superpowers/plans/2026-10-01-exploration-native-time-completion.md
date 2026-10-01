# 探索收尾：原时间提供者完成接口

本项沿已批准设计的全Party被动恢复与真实整夜休息来源要求实施。静态调用链和精确来源在output/exploration-quality-goal-20260930/stage1/stage2-time-provider-contract-proposal.md；现有回调丢失Promise，纯外层await不能证明完成。

## 第一段：Patreon精确源码接缝

只为固定Patreon3.2.29原src/index.js（SHA89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9）准备可逆隔离副本补丁。原文件、manifest、当前QA runtime、安装候选、旧世界和报告不改。补丁作者只在本goal输出中创建新的patcher、patched-source副本、原字节归档、diff／hash清单和有界测试；安装另由root处理。

- 保留原Mn和Rn调用各一次，以及原筛选、predicate、resolveValue、真实RNG、治疗计算、写入参数、规则和写入发起顺序。登记每个已经调用的原生写入Promise，等同一Promise结束，不重新调用或补算。
- 必须先核Rn中的decrease真实返回语义；不能把forEach或decrease的undefined当终结。无法证明的分支明确拒绝完成，不用延时或静默期。
- 原updateWorldTime callback把原handler的terminalPromise和实际调用清单交出。提供窄的有版本observer注册及disposer，只有原callback发出一次调用；不开新的推进／治疗入口，不改变Hooks全局调度，不按toString模糊删除listener。
- 实际调用元数据至少包括provider/version/baseSourceSHA、invocationId、原worldTime/delta/options、实际activeGM身份、原成员／规则／roll与文档写入。Promise留私有内存，持久证明只用JSON投影与准确来源ID。无规则／满HP也须真实原循环结束后的no-op证明。
- 具体observer导出位置和exact事件形状先保存最小接口清单，再生成补丁；保留既有module API，不覆盖其他成员。source/hash不匹配、重复接缝、重复listener、未知写入或错误来源均阻断，不自动修改未知版本。

必要失败测试直接运行固定原函数的窄源码及新接缝，比较未打补丁／打补丁的规则选择、RNG调用、原写入参数和发起顺序。gate最后骰子和写入Promise证明settle不能早返；含全Party重复成员、未选患者、predicate false、字符串／数值、满HP、rejection、重复观察／dispose及精确版本拒绝。不加载运行整份依赖bundle，不触发世界操作。保存RED和GREEN，代码测试不冒称真实世界验收。

## 第二段：本模组adapter及单点集成

接缝独立核验后，T作者拥有time-effects.mjs、新窄Patreon完成adapter和相关测试；I作者root拥有runtime／capabilities的注入及准确handler清单。beforeAdvance只绑定本次私有checkpoint，原clock仍唯一推进；settle等待同次原handler全部终结并核其原native options和身份。刷新、未知／拒绝／迟到响应保留事实，不补治疗或第二次advance。Stage1的clock／revision执行权算法不重写。

## 第三段：Rest调用来源

另起固定Calendaria1.4.2／PF2e8.5.1的窄源码接缝：真正Rest发出每角色Hook时附同次调用来源，原wm把同次元数据沿原gated／trigger传到唯一native advance options，并交出实际Promise。包装public PF2e Rest只保存原this／args／return与真实调用边界；不借用公开calendar advance另加时间。ordinary／cinematic、群体、取消／disabled、非primaryGM、重叠与未知来源都保存真实行为。不把500ms debounce当幂等或完成证明。

每段先有准确patch清单、原字节归档和独立审查，再由root安装到新隔离QA。现S2世界的未知执行域必须先按原生关闭／锁释放／库存保全后才安装下一候选。最终发行仅包含自有适配代码与必要补丁工具／来源清单，不把完整付费依赖副本收入仓库或模组ZIP；正式部署按既有授权由root单点执行，保留原依赖字节与可逆资料。

最终实际场景10须证明原handler真实随机与每个原文档ACK全部完成后才开始下一治疗；Rest为实际调用source marker与一次原时间推进。没有这些正向证据不能称本项完成。
