# MaiBot Character Knowledge Plugin

通过 AnimeTrace 等联网 API 以及本地角色库为图片补充角色标签的 MaiBot SDK v2 插件。

示例：

```text
[图片：二次元角色图片，联网候选：アロナ] 图片[アロナ]
[图片：二次元角色图片，联网候选：プラナ] 图片[プラナ]
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

[library]
enabled = false
file_name = "characters.json"
admin_qq = []

[reverse_image]
anime_trace_enabled = true
anime_trace_url = "https://api.animetrace.com"
anime_trace_max_upload_bytes = 900000
timeout_seconds = 15
```

本插件内置可调用的 Gemini 或 OpenAI-compatible VLM。启用后，图片会跳过MaiBot 内置 VLM 而是发送到这里配置的 `base_url`，并使用这里的 `api_key` 和 `model`。

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

### Lightweight Online Mode

`vision.enabled = false` 且 `anime_trace_enabled = true` 时：

```text
AnimeTrace lookup
  -> match: keep original image and append [角色识别：图片[角色]]
  -> no match: keep original image unchanged
```

因此未识别图片、真人和非人物图仍会交给 MaiBot 内置 VLM，并保留它原有的图片提示词与描述能力。

AnimeTrace 仅接受其返回的明确角色候选；低置信候选不会自动写入标签。多人图会保留多个明确角色标签，例如 `图片[角色A、角色B]`。

## Private Character Library

启用 `[library].enabled` 后，插件从 `data/characters.json` 读取管理员维护的角色库。关系标签只来自本地库，例如 `图片[角色名]（自己）`。

生产环境应将允许管理角色库的 QQ 号写入：

```toml
[library]
admin_qq = ["123456789"]
```

留空时不限制管理员，方便 WebUI 本地测试。

### Commands

以下命令都需要本地角色库开启，并受 `admin_qq` 限制。命令会主动发送可见回执，适用于 QQ 和 WebUI。

```text
/角色添加 角色名 [关系]
/识图修正 角色名
/角色列表
/查看人设 角色名
/设置关系 角色名 关系
/添加别名 角色名 别名
/删除别名 角色名 别名
```

`/角色添加` 后，下一条独立图片会创建该角色的外观卡。`/识图修正` 会优先使用引用的旧图；不支持引用的平台会使用同一发送者在当前聊天最近识别过的图片。

## Limits And Privacy

- `max_images_per_message` 默认 4，超过的图片不会由插件请求外部服务。
- `max_concurrency` 默认 1，避免大量图片同时触发限流。
- 相同图片按哈希缓存，避免重复请求。
- 开启 `vision` 时，图片会发送给你配置的视觉服务。
- 开启 AnimeTrace 时，图片会发送给 AnimeTrace。
- 本地角色库不会因为普通聊天自动新增或修改；只有管理员命令可以写入。
