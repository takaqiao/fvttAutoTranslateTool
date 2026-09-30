# 0.9.20 发布工具只读复核

本复核只读取本地工具与上一版证据，执行本地 fixture/ZIP 验证测试；未执行 capture、SSH、公开下载、发版、部署或生产写入。源目录为 `C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/release-tools`。新远端目录固定为 `/root/fvtt-patch-backups/automation-release-0920-20260930`，没有复用 `.19` 的 `0919` 事务目录。

## 结论与证据

| 检查 | 已证明范围 |
|---|---|
| 前版清单 | `prior-release-manifest.json` 与上一版 `automation-native-20260930/release-tools/release/release-manifest.json` 原始字节相等，148 个文件；源提交 `f22980350ec176a7b6df69bb68d5062733b7d484`，版本/tag 均为 `.19`。清单 SHA256 `a879028ae6529459b0964fec04bc185d84a4c372ce50cbca451666c8a60ba77b`。 |
| 新版本参数 | prepare、lib、capture、freeze、deploy、transaction 的转换统一为 `0.9.19 → 0.9.20`，新 tag `pf2e-third-party-automation-v0.9.20`。模块根仅 `/root/fvtt14-data/Data/modules/pf2e-third-party-automation`。 |
| 本地工具测试 | `node --test .../transaction.test.mjs`：17/17，零跳过（工具输出 `16efce`）；`python .../artifact-checks.test.py`：7/7（`a7ab3c`）；9 个 `.mjs` 的 `node --check` 全通过（`35e3f8`）。测试的 fixture 会在工具目录内创建临时文件并清理，不访问生产。 |
| 验收计数门槛 | 初读 `verify-evidence.mjs:20` 为 1970；root 随后已改为 1996，二次读取确认。本复核没有改其文件。完整模块 1996/1996 是 root 的既有结果，非本复核重新执行。最后热路径修复若新增测试，root 应绑定新的完整日志并同步该 exact-count 条件；freeze/deploy 的证据下限为 >=1996。 |
| 后续补强复核 | root 补了 freeze evidence gate 与真实 latest HTTP 校验。曾发现 freeze 五个 inputs 与 deploy 旧四个 key 不兼容；root 已修 strict5keys，并补部署侧 evidence verified/同commit/version/文件数/测试全过 gate。现有两个 key-array 直接参数对账通过（`066f5e`），9mjs再次 syntax 全过，transaction SHA 未变，17/17 证据仍适用。 |
| 尚未证明 | 新 `.20` capture、公开资产、冻结 plan、远端安装和 HTTP 提供的实际字节均须由后续授权交付取得；本地绿色测试不能替代这些回执。 |

## 停服门槛与保护边界

- `capture.mjs:8,17–19` 读取主服务 `30001` 的 API active=false，并在原生进程检查 ready=false、world=null、startup world=null、port=30001、原模块 `.19`、unlocked、零 world activity users。capture 不关闭世界、不重启服务。freeze `:11–13` 要求上述基线与公开附件证明，且基线未超过 30 分钟；`:7–10` 另绑定同提交的验收证据。
- `deploy.mjs:19–25` 再核 Setup API、受保护字节、原模块完整文件集合和原生状态。实际换目录的同步 inspector 事务还以 `transaction.mjs:24–43` 核 PID/startTicks、Core14.368、startup/disk options、PF2e8.5.1、game.active/world/ready、activity users 及 active game.users、模块锁与 lock 文件。备份完成后 `:101–104` 在首次 rename 前再次 guard；不依靠一次旧 capture 决定安全。
- `capture.mjs:10–15` 保存八个模块的完整文件图：av-v14-hotfix、pf2e-bob-animations、pf2e-bob-companion、wayfinder、sundry、sequencer、pf2e-toolbelt、pf2e-reaction。完整世界树同时加入 protectedFiles 与 protectedTrees。上一 `.19` 冻结 plan 是 170 个世界文件、193 个受保护文件；新 capture 应以当时完整实际集合为准，不用静态 170 排除新文件。
- `capture.mjs:7,19` 与 `deploy.mjs:7,9,27–29` 保护 PM2 id2/id4 的 PID、启动 ticks、status 和启动定义 SHA，另保护 options、系统、Core、原生字体、audio 与 IWR 启动适配文件。世界、模块新增/删除文件也会造成完整 file-map 比较失败。
- “主世界未加载、零用户”绑定主服务 `30001`，没有宣称 id4 的独立世界也被关闭。两个服务的身份/定义保持不变；本流程没有调用 PM2 restart 或自动 shutdown。
- `transaction.mjs:88–92` 的最低 mandatory-module 列表只有六项，但当前 capture/freeze 实际传入完整八项，guard `:39` 逐项保护其全部文件；Toolbelt 版本另由 `:43` 校验。实际交付必须确认冻结 plan 的 protectedModules 确有八项，不能只凭最低名单推定八项齐全。

## 具体调用顺序（仅供后续授权交付，未在本复核执行）

1. root 完成候选与 `.20` manifest，记录纳入提交/独立审查/实际 QA 报告；将模块运行文件提交。`prepare-release.mjs:8–13` 会拒绝模块目录内任何已修改或未跟踪文件，打包的是 HEAD 的 Git blob 原始字节。
2. 在 PowerShell 运行下列本地打包命令。现有 `release` 目录不会被覆盖，已有成品应沿用并复核；不能删除它来掩盖候选冲突。

   ```powershell
   $taskReleaseTools = 'C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/release-tools'
   $taskSource = 'C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt'
   node "$taskReleaseTools/prepare-release.mjs" $taskSource
   ```

3. 通过 root 当前已授权的发行流程发布同提交 tag 与实际生成的 ZIP/module.json；本工具目录没有 publish 实现，不另造并行发布流程。随后运行：

   ```powershell
   node "$taskReleaseTools/verify-public.mjs" "$taskReleaseTools/release"
   ```

   它解析实际 GitHub tag 到 commit，重下载该 tag 的真实附件，比较 ZIP/manifest 原始 SHA、完整文件集合及 Git blob 内容，禁止 traversal、重复文件、symlink、错误根目录和 stream 路径。
4. 新 `verify-public.mjs:12–14` 已真实 HTTP 读取 `https://github.com/takaqiao/fvttAutoTranslateTool/releases/latest/download/module.json`，要求 status200，原始 SHA 等于指定 tag 的模块清单 SHA，并保存 `downloaded/latest-module.json` 原字节及 proof 的 latestURL/latestManifestSHA256。后续 ZIP 校验再次将同一 module SHA 绑定到包内 manifest；不再需要另造重复 latest 核验流程。
5. 在最终实际ZIP报告确认无 errors、全部 commands 的 ok 均为true后，运行 `verify-evidence.mjs`，参数可有多个绝对报告路径。下面的占位符必须替换成root实际新取得的最终clean报告；本复核不预造报告路径：

   ```powershell
   node "$taskReleaseTools/verify-evidence.mjs" '<final-package-report-placeholder：实际ZIP clean报告绝对路径>'
   ```

   root已将最终gate补为每个报告 `commands.every(ok===true)`；早期 `1790746880258` 虽有充分绑定的voltage sourcechanged成功command，但也保留首个Chain错误UUID harness失败，因此不能作为clean finalgate输入。早期report原样保留，所需成功case由root新实际ZIP报告加输入/consumer绑定充分的既有证据复用，并明确覆盖限制，不把旧report冒称clean。工具不会自动判断所有玩法入口是否已验收；确认 `evidence-verification.json` 的同commit、最终完整测试结果、候选字节与实际覆盖说明均正确。
6. 公开发布与 QA 证据就绪后再获取新基线，避免 30 分钟窗口在发版等待中耗尽。`capture.mjs` 是只读 SSH，但仍只由 root 在授权交付阶段执行：

   ```powershell
   node "$taskReleaseTools/capture.mjs"
   node "$taskReleaseTools/freeze.mjs" '<capture 返回的新 baseline 绝对路径>' "$taskReleaseTools/release"
   ```

   root 复核 `ready/plan.json` 的 PID/startTicks、id2/id4 services、完整世界图、八模块图、保护文件范围（上一计划为193个）、before/after/delta、同提交证据与 plan SHA。新 freeze `:7–10` 核 evidence verified/同commit/version/文件数/全过测试，并重读 testlog/native-report 原字节 SHA；`:15–17` 将 evidence-verification.json 一并复制、列入五个 inputs 冻结。新 deploy `:14,16` 精确接受同样五key，逐份 SHA 核验，并再次检查 evidence 语义。既有 first-read 四key错配已由 root 修复。
7. root 将这份完整 ready 内容放入全新、未尝试的远端 `0920` 目录后，以从本地固定得到的同一 plan SHA 调用。以下是远端 Node 参数，不能从本地运行 `deploy.mjs`：

   ```text
   node /root/fvtt-patch-backups/automation-release-0920-20260930/deploy.mjs inspect <planSHA256>
   node /root/fvtt-patch-backups/automation-release-0920-20260930/deploy.mjs apply <planSHA256> --authorized-setup-deploy
   ```

   inspect 必须通过后才 apply。deploy 只认上述真实 Linux 路径、原 `.19` 与 payload `.20` 字节、未消费事务、30 分钟基线和原 PID/startTicks。root 保存 deployment-receipt.json、postcheck.json、plan 与相关文件原始 SHA；最终 HTTP 会核所有新运行文件，不能以磁盘/版本号代替实际服务字节。

## 备份、失败与人工边界

`transaction.mjs:97–106` 确保 live/stage/remoteRoot 同文件系统；原模块按原样 cp 到 `backup-pf2e-third-party-automation-0.9.19`，逐文件 SHA 复核、fsync 后，先 rename live→retired，再 stage→live。native package cache 同步刷新，仅在 Setup执行；原世界不修改。

attempt.json 在备份/rename 前以 wx 写入并 fsync；即使尚未 swap，失败的已消费 attempt 也不能直接 apply 重试。第二次 rename 或 cache 失败由事务用已验证 retired/backup 恢复旧字节与原缓存。`transaction.mjs:49–67,108–113` 遇到其他写入者改了 live/retired/failed、世界、配置或保护模块时拒绝覆盖，返回 recovery-incomplete 留给 root 核对。

若同步 transaction 已 installed，但后面的 API/保护文件/HTTP 检查失败，外层 `deploy.mjs:26–29` 保留安装回执并报错；它不会自动恢复 `.19`。此时不能再 apply、不能称部署完成，也不能只为使脚本通过而改旧基线。先按实际 installed/rolled-back/refused/recovery-incomplete 状态审回执与当前字节。CLI 只提供 inspect/apply，没有直接 rollback CLI；显式恢复要复用经审查的 nativePairTransaction rollback 路径及当前 Setup/字节保护条件。

工具链现未见路径或事务阻断；四key错配已修复并独立复核。root 正等待 receipt 热路径改为 nonce 索引后再打包，新候选文件数以最终 manifest 为准，不能把前版148当新包数量。之后按上述顺序完成最终日志/QA、公开附件、同 commit evidence、新鲜 capture、freeze、inspect/apply 及 HTTP 字节核验。本结论不宣称 `.20` 已发布或安装。
