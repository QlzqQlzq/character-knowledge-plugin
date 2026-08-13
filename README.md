# MaiBot Character Knowledge Plugin

复用 MaiBot 的 VLM 图片描述，通过 AnimeTrace 等联网 API 以及可选本地角色库为图片补充角色标签的 MaiBot SDK v2 插件。

示例：

```text
[图片：蓝白发少女站在教室中，头顶有发光圆环。] 图片[アロナ]
[图片：白发少女穿黑色制服，头顶有深色圆环。] 图片[プラナ]
```

## Configuration

编辑 `config.toml`，或在 MaiBot WebUI 中配置。默认启用 AnimeTrace 联网检索；视觉服务和本地角色库默认关闭，且不包含任何 API Key。

```toml
[vision]
enabled = false
provider = "gemini" # gemini or openai
api_key = ""
base_url = "https://generativelanguage.googleapis.com/v1beta"
model = "gemini-3-flash"
max_upload_bytes = 4194304

[library]
enabled = false
file_name = "characters.json"
admin_qq = []
allow_unrestricted_admin = false
max_prompt_characters = 40
pending_addition_ttl_seconds = 300

[reverse_image]
anime_trace_enabled = true
anime_trace_url = "https://api.animetrace.com"
anime_trace_max_upload_bytes = 900000
timeout_seconds = 15
```

`plugin.max_characters_per_image` 控制单张图片最多保留的角色候选数，默认 3，可设置为 1–10；多人图较多时可以适当调高。

默认模式通过 SDK 的 `llm.generate` capability 调用 MaiBot 已配置的 `vlm` 任务生成通用图片描述，不需要在插件里重复填写模型密钥。

本插件也内置可单独配置的 Gemini 或 OpenAI-compatible VLM。启用后，它会代替默认描述调用，并额外负责二次元人物判断和本地角色库匹配；图片会发送到这里配置的 `base_url`，并使用这里的 `api_key` 和 `model`。

`anime_trace_enabled` 启用后，图片会上传到 AnimeTrace 用于二次元角色检索。图片会自动为该请求创建压缩副本，原图不会被改写。

## Recognition Modes

### Plugin VLM Mode

`vision.enabled = true` 时：

```text
plugin VLM: describe image + classify anime character + check private library
  -> not an anime character: do not query AnimeTrace
  -> anime character: query AnimeTrace if no private-library label exists
```

该模式由插件的 VLM 产生通用图片描述并替换图片文本，可避免 MaiBot 内置 VLM 再次识图。适合需要本地角色库的场景。

### MaiBot VLM + Online Mode

`vision.enabled = false` 且 `anime_trace_enabled = true` 时：

```text
MaiBot vlm task and AnimeTrace run in parallel
  -> match: output [图片：MaiBot VLM 描述] 图片[角色]
  -> no match: output [图片：MaiBot VLM 描述] 图片[未识别]
```

插件沿用 MaiBot 的图片描述提示词风格。未识别图片、真人和非人物图仍会获得普通图片描述；AnimeTrace 只负责补充角色标签，不会替代通用描述。

AnimeTrace 仅接受其返回的明确角色候选；低置信候选不会自动写入标签。多人图会保留多个明确角色标签，例如 `图片[角色A、角色B]`。

## Private Character Library

启用 `[library].enabled` 后，插件从 MaiBot 为本插件分配的持久化数据目录读取 `characters.json`。插件升级或替换源码目录不会覆盖角色库；旧版本源码目录中的 `data/characters.json` 会在首次启动时自动迁移。关系标签只来自本地库，例如 `图片[角色名]（自己）`。

生产环境应将允许管理角色库的 QQ 号写入：

```toml
[library]
admin_qq = ["123456789"]
```

`admin_qq` 留空时，聊天管理命令默认对所有人关闭。仅在受控的本地 WebUI 测试中，可显式设置 `allow_unrestricted_admin = true`；服务器上不应开启它。

### Commands

以下命令都需要本地角色库开启，并受 `admin_qq` 限制。命令会主动发送可见回执，适用于 QQ 和 WebUI。

```text
/角色添加 角色名 [关系]
/取消角色添加
/识图修正 角色名
/角色列表
/查看人设 角色名
/设置关系 角色名 关系
/添加别名 角色名 别名
/删除别名 角色名 别名
```

`/角色添加` 后，下一条独立图片会创建该角色的外观卡；等待默认在 5 分钟后失效，也可用 `/取消角色添加` 主动取消。`/识图修正` 会优先使用引用的旧图；不支持引用的平台会使用同一发送者在当前聊天最近识别过的图片。角色名称与别名必须保持唯一。

## Limits And Privacy

- `max_images_per_message` 默认 4，超过的图片不会由插件请求外部服务。
- `max_concurrency` 默认 1，避免大量图片同时触发限流。
- `message_timeout_seconds` 默认 110 秒；默认模式中的 MaiBot VLM 与 AnimeTrace 并行执行，超时或描述失败时图片会原样交还 MaiBot 后续链路。
- 相同图片按哈希进行有界、限时缓存，避免重复请求和内存持续增长。
- 缓存优先使用 MaiBot 提供的 SHA-256 图片唯一 ID；字段缺失时按图片内容计算 SHA-256，QQ 与 WebUI 可复用同一逻辑。
- 发送给 MaiBot VLM、插件视觉服务和 AnimeTrace 的图片副本都有体积上限，超限时会缩放并转为 JPEG。
- 视觉服务或 AnimeTrace 连续失败 2 次后会暂停请求 60 秒，随后自动恢复尝试。
- 开启 `vision` 时，图片会发送给你配置的视觉服务。
- 开启 AnimeTrace 时，图片会发送给 AnimeTrace。
- 本地角色库不会因为普通聊天自动新增或修改；只有管理员命令可以写入。
