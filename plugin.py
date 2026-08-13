import asyncio
import base64
import binascii
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import time
from typing import Any, Literal

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .repository import CharacterRepository
from .vision import build_appearance_cards, identify_image
from .reverse_image import ReverseImageHint, search_anime_trace


@dataclass
class ServiceHealth:
    consecutive_failures: int = 0
    open_until: float = 0.0


class PluginSettings(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "scan-face"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用图片角色知识注入")
    config_version: str = Field(default="1.0.3", description="配置版本")
    max_images_per_message: int = Field(default=4, ge=1, le=20, description="单条消息最多识别图片数，超出的图片标记为未识别")
    max_concurrency: int = Field(default=1, ge=1, le=4, description="视觉接口最大并发数")
    timeout_seconds: int = Field(default=25, ge=5, le=120, description="单张图片的视觉接口超时秒数")
    message_timeout_seconds: int = Field(default=110, ge=10, le=115, description="单条消息全部图片处理的总超时秒数")
    cache_max_entries: int = Field(default=256, ge=16, le=4096, description="图片识别缓存最大条目数")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="图片识别缓存有效秒数")
    latest_image_max_entries: int = Field(default=64, ge=1, le=512, description="修正功能保留的最近图片最大用户数")
    latest_image_max_total_bytes: int = Field(default=16777216, ge=1048576, le=268435456, description="最近图片缓存总字节上限")
    circuit_breaker_failures: int = Field(default=2, ge=1, le=10, description="外部服务连续失败多少次后暂时熔断")
    circuit_breaker_cooldown_seconds: int = Field(default=60, ge=10, le=3600, description="外部服务熔断冷却秒数")


class VisionSettings(PluginConfigBase):
    __ui_label__ = "视觉识别"
    __ui_icon__ = "eye"
    __ui_order__ = 1

    enabled: bool = Field(default=False, description="启用后才会向配置的视觉接口发送图片")
    provider: Literal["gemini", "openai"] = Field(default="gemini", description="Gemini 原生接口或 OpenAI 兼容视觉接口")
    api_key: str = Field(default="", description="视觉接口密钥；建议通过服务器上的插件配置填写")
    base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Gemini API 根地址，OpenAI 模式填写 API 根地址")
    model: str = Field(default="gemini-3-flash", description="支持视觉输入和 JSON 输出的模型名")
    max_upload_bytes: int = Field(default=4194304, ge=100000, le=20971520, description="发送给视觉服务前的最大图片字节数")


class LibrarySettings(PluginConfigBase):
    __ui_label__ = "私有角色库"
    __ui_icon__ = "book-user"
    __ui_order__ = 2

    enabled: bool = Field(default=False, description="启用本地 OC、自己和朋友等权威角色库")
    file_name: str = Field(default="characters.json", description="MaiBot 分配的插件持久化目录中的角色库 JSON 文件名")
    admin_qq: list[str] = Field(default_factory=list, description="允许通过聊天管理角色库的 QQ 号")
    allow_unrestricted_admin: bool = Field(default=False, description="允许任何用户管理角色库，仅用于受控的 WebUI 本地测试")
    max_prompt_characters: int = Field(default=40, ge=1, le=200, description="单次视觉请求最多携带的本地角色数")
    pending_addition_ttl_seconds: int = Field(default=300, ge=30, le=3600, description="角色添加等待参考图的有效秒数")


class ReverseImageSettings(PluginConfigBase):
    __ui_label__ = "反向搜图"
    __ui_icon__ = "image-search"
    __ui_order__ = 3

    anime_trace_enabled: bool = Field(default=True, description="启用 AnimeTrace 二次元角色检索；轻量联网模式会直接写入明确候选，插件 VLM 模式用于补充未命中的公共角色")
    anime_trace_url: str = Field(default="https://api.animetrace.com", description="AnimeTrace API 地址")
    anime_trace_max_upload_bytes: int = Field(default=900000, ge=100000, le=900000, description="AnimeTrace 上传前的最大图片字节数；超出时自动缩放并压缩为 JPEG")
    timeout_seconds: int = Field(default=15, ge=5, le=60, description="反向搜图请求超时秒数")


class CharacterKnowledgeConfig(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    library: LibrarySettings = Field(default_factory=LibrarySettings)
    reverse_image: ReverseImageSettings = Field(default_factory=ReverseImageSettings)


class CharacterKnowledgePlugin(MaiBotPlugin):
    config_model = CharacterKnowledgeConfig

    def __init__(self) -> None:
        super().__init__()
        self._repository: CharacterRepository | None = None
        self._cache: OrderedDict[str, tuple[float, tuple[str, str]]] = OrderedDict()
        self._latest_recognitions: OrderedDict[tuple[str, str], tuple[bytes, str]] = OrderedDict()
        self._semaphore = asyncio.Semaphore(1)
        self._repository_lock = asyncio.Lock()
        self._pending_additions: dict[tuple[str, str], tuple[float, str, str]] = {}
        self._background_tasks: set[asyncio.Task[tuple[str, str]]] = set()
        self._config_generation = 0
        self._service_health = {"vision": ServiceHealth(), "anime_trace": ServiceHealth()}

    async def on_load(self) -> None:
        self._reload_repository()
        self._semaphore = asyncio.Semaphore(self.config.plugin.max_concurrency)
        self.ctx.logger.info("角色知识插件已加载：视觉=%s 私库=%s AnimeTrace=%s", self.config.vision.enabled, self.config.library.enabled, self.config.reverse_image.anime_trace_enabled)

    async def on_unload(self) -> None:
        self._cache.clear()
        self._latest_recognitions.clear()
        self._pending_additions.clear()
        self._config_generation += 1
        self._service_health = {"vision": ServiceHealth(), "anime_trace": ServiceHealth()}
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        self.ctx.logger.info("角色知识插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del scope, config_data
        self._config_generation += 1
        self._cache.clear()
        self._latest_recognitions.clear()
        self._pending_additions.clear()
        async with self._repository_lock:
            self._reload_repository()
        self._semaphore = asyncio.Semaphore(self.config.plugin.max_concurrency)
        self.ctx.logger.info("角色知识插件配置已更新：version=%s", version)

    @HookHandler(
        "chat.receive.before_process",
        name="inject_character_knowledge",
        description="在消息进入 MaiBot 前识别图片角色并附加紧凑身份标签。",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=120000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def inject_character_knowledge(self, message: Any, **kwargs: Any) -> dict[str, Any] | None:
        del kwargs
        if not self.config.plugin.enabled:
            return None
        images = self._image_components(message)
        if not images:
            return None
        key = self._pending_key(message)
        pending = self._pending_additions.pop(key, None)
        if pending is not None:
            created_at, name, relationship = pending
            if time.monotonic() - created_at <= self.config.library.pending_addition_ttl_seconds:
                return await self._create_pending_character(message, images, (created_at, name, relationship))
            stream_id = str(message.get("session_id") or "") if isinstance(message, dict) else ""
            if stream_id:
                await self.ctx.send.text("角色添加已超时，本次图片将按普通图片处理。", stream_id)
        replacements = await self._build_replacements(message, images)
        modified = dict(message)
        components = list(modified.get("raw_message") or [])
        output: list[dict[str, Any]] = []
        image_index = 0
        for component in components:
            if isinstance(component, dict) and component.get("type") == "image":
                output.extend(replacements[image_index])
                image_index += 1
            else:
                output.append(component)
        modified["raw_message"] = output
        return {"action": "continue", "modified_kwargs": {"message": modified}}

    async def _build_replacements(
        self, message: dict[str, Any], images: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        limit = self.config.plugin.max_images_per_message
        generation = self._config_generation
        tasks = [asyncio.create_task(self._recognize_limited(image, generation)) for image in images[:limit]]
        try:
            done, pending = await asyncio.wait(tasks, timeout=self.config.plugin.message_timeout_seconds)
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    self._track_background_task(task)
            raise
        if pending:
            self.ctx.logger.warning(
                "单条消息图片识别超过总预算，%s 张未完成图片将原样交给 MaiBot", len(pending)
            )
            for task in pending:
                self._track_background_task(task)
        recognized: list[tuple[str, str] | None] = []
        for task in tasks:
            if task not in done:
                recognized.append(None)
                continue
            try:
                recognized.append(task.result())
            except Exception as exc:
                self.ctx.logger.warning("单张图片识别异常，将原图交给 MaiBot: %s", exc)
                recognized.append(None)
        replacements: list[list[dict[str, Any]]] = []
        for index, image in enumerate(images):
            if index >= limit:
                replacements.append([image])
                continue
            result = recognized[index]
            if result is None:
                replacements.append([image])
                continue
            description, label = result
            image_bytes = self._decode_image(image)
            if image_bytes is not None:
                self._remember_latest(self._pending_key(message), image_bytes, label)
            if self.config.vision.enabled and description != "图片识别失败":
                replacements.append([{"type": "text", "data": f"[图片：{description}] {label}"}])
            elif label == "图片[未识别]":
                replacements.append([image])
            else:
                replacements.append([image, {"type": "text", "data": f"[角色识别：{label}]"}])
        return replacements

    async def _recognize_limited(self, image: dict[str, Any], generation: int) -> tuple[str, str]:
        async with self._semaphore:
            result = await self._recognize_image(image)
            if generation != self._config_generation:
                return "图片识别失败", "图片[未识别]"
            return result

    def _track_background_task(self, task: asyncio.Task[tuple[str, str]]) -> None:
        if task in self._background_tasks:
            return
        self._background_tasks.add(task)
        task.add_done_callback(self._finish_background_recognition)

    def _finish_background_recognition(self, task: asyncio.Task[tuple[str, str]]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            self.ctx.logger.warning("后台图片识别异常: %s", exc)

    @Command(
        "character_add",
        description="管理员创建角色库条目，随后发送一张参考图自动生成外观卡。",
        pattern=r"^/角色添加\s+(?P<name>\S+)(?:\s+(?P<relationship>[^\[\]\r\n]{1,32}?))?(?:\s+\[图片[：:].*)?$",
        timeout_ms=5000,
    )
    async def add_character(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        raw_message = message if isinstance(message, dict) else {}
        stream_id = str(raw_message.get("session_id") or "")
        if not self.config.library.enabled:
            return await self._command_feedback(stream_id, "私有角色库当前未启用。", success=False)
        if not self.config.plugin.enabled or not self.config.vision.enabled:
            return await self._command_feedback(stream_id, "视觉识别当前未启用，无法从参考图创建角色。", success=False)
        if not self._is_admin(raw_message):
            return await self._command_feedback(stream_id, "没有角色库管理权限。", success=False)
        matched = kwargs.get("matched_groups")
        name = str(matched.get("name") or "").strip() if isinstance(matched, dict) else ""
        relationship = str(matched.get("relationship") or "").strip() if isinstance(matched, dict) else ""
        if not name:
            return await self._command_feedback(stream_id, "用法：/角色添加 名字 [关系]，然后发送一张参考图。", success=False)
        self._discard_expired_additions()
        self._pending_additions[self._pending_key(raw_message)] = (time.monotonic(), name, relationship)
        while len(self._pending_additions) > self.config.plugin.latest_image_max_entries:
            self._pending_additions.pop(next(iter(self._pending_additions)))
        prompt = f"已准备创建“{name}”。请在本聊天单独发送一张参考图，下一条带图消息会用于建库。"
        return await self._command_feedback(stream_id, prompt, success=True)

    @Command("character_add_cancel", description="取消当前聊天中等待参考图的角色添加。", pattern=r"^/取消角色添加\s*$", timeout_ms=5000)
    async def cancel_character_add(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        del kwargs
        raw_message = message if isinstance(message, dict) else {}
        stream_id = str(raw_message.get("session_id") or "")
        if not self._is_admin(raw_message):
            return await self._command_feedback(stream_id, "没有角色库管理权限。", success=False)
        removed = self._pending_additions.pop(self._pending_key(raw_message), None)
        text = "已取消等待中的角色添加。" if removed is not None else "当前没有等待参考图的角色添加。"
        return await self._command_feedback(stream_id, text, success=True)

    @Command(
        "character_correct",
        description="管理员修正本聊天最近一次识图结果；引用旧图时优先修正引用图。",
        pattern=r"^/识图修正\s+(?P<name>\S+)\s*$",
        timeout_ms=120000,
    )
    async def correct_character(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        raw_message = message if isinstance(message, dict) else {}
        stream_id = str(raw_message.get("session_id") or "")
        if not self.config.library.enabled:
            return await self._command_feedback(stream_id, "私有角色库当前未启用。", success=False)
        if not self.config.plugin.enabled or not self.config.vision.enabled:
            return await self._command_feedback(stream_id, "视觉识别当前未启用，无法使用引用图片修正角色。", success=False)
        if not self._is_admin(raw_message):
            return await self._command_feedback(stream_id, "没有角色库管理权限。", success=False)
        matched = kwargs.get("matched_groups")
        name = str(matched.get("name") or "").strip() if isinstance(matched, dict) else ""
        target = self._repository.find_name(name) if name and self._repository else None
        if target is None:
            return await self._command_feedback(stream_id, f"角色库中不存在“{name}”。", success=False)
        canonical_name = target.name
        try:
            image_bytes, old_label = await self._image_for_correction(raw_message)
            async with self._semaphore:
                cards = await build_appearance_cards(
                    provider=self.config.vision.provider,
                    api_key=self.config.vision.api_key,
                    base_url=self.config.vision.base_url,
                    model=self.config.vision.model,
                    image_bytes=image_bytes,
                    timeout_seconds=self.config.plugin.timeout_seconds,
                    max_upload_bytes=self.config.vision.max_upload_bytes,
                )
            async with self._repository_lock:
                character = self._repository.append_appearance_cards(name=canonical_name, appearance_cards=cards)
        except Exception as exc:
            self.ctx.logger.warning("引用图片修正角色库失败: name=%s error=%s", canonical_name, exc)
            return await self._command_feedback(stream_id, "引用图片修正失败，详情见日志。", success=False)
        self._invalidate_recognition_cache()
        self.ctx.logger.info("管理员引用图片修正角色库成功: name=%s cards=%s", character.name, len(character.appearance_cards))
        return await self._command_feedback(
            stream_id,
            f"✅ 识图修正完成\n{old_label} → 图片[{character.name}]\n已补充外观卡，当前共 {len(character.appearance_cards)} 条。",
            success=True,
        )

    async def _image_for_correction(self, message: dict[str, Any]) -> tuple[bytes, str]:
        """Prefer a quoted image, otherwise use this sender's latest recognized image."""
        reference_id = self._referenced_message_id(message)
        stream_id = str(message.get("session_id") or "")
        if reference_id:
            referenced = await self.ctx.message.get_by_id(
                reference_id, stream_id=stream_id, include_binary_data=True
            )
            source = referenced.get("message") if isinstance(referenced, dict) and referenced.get("success") else None
            images = self._image_components(source)
            if not images:
                raise ValueError("被引用消息中没有可读取的图片")
            image_bytes = self._decode_image(images[0])
            if image_bytes is None:
                raise ValueError("被引用图片的二进制数据不可用")
            return image_bytes, "图片[原识别结果未知]"
        latest = self._latest_recognitions.get(self._pending_key(message))
        if latest is None:
            raise ValueError("未找到本聊天中你最近一次识别过的图片，请先发送图片后再修正")
        return latest

    async def _command_feedback(self, stream_id: str, text: str, *, success: bool) -> tuple[bool, str, int]:
        """Commands always send visible feedback because WebUI does not render return text."""
        if stream_id:
            await self.ctx.send.text(text, stream_id)
        return success, "", 1

    async def _admin_command(self, message: Any) -> tuple[dict[str, Any], str] | None:
        raw_message = message if isinstance(message, dict) else {}
        stream_id = str(raw_message.get("session_id") or "")
        if not self.config.library.enabled:
            await self._command_feedback(stream_id, "私有角色库当前未启用。", success=False)
            return None
        if not self._is_admin(raw_message):
            await self._command_feedback(stream_id, "没有角色库管理权限。", success=False)
            return None
        return raw_message, stream_id

    @Command("character_list", description="管理员查看本地角色库。", pattern=r"^/(?:角色列表|角色库)$", timeout_ms=5000)
    async def list_characters(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        del kwargs
        authorized = await self._admin_command(message)
        if authorized is None:
            return False, "", 1
        _, stream_id = authorized
        characters = self._repository.list_characters() if self._repository else ()
        if not characters:
            return await self._command_feedback(stream_id, "角色库为空。", success=True)
        lines = ["角色库："]
        for character in characters:
            relation = f"（{character.relationship}）" if character.relationship else ""
            lines.append(f"- {character.name}{relation}：{len(character.appearance_cards)} 条外观卡")
        return await self._command_feedback(stream_id, "\n".join(lines), success=True)

    @Command("character_view", description="管理员查看角色档案。", pattern=r"^/(?:查看人设|查看角色)\s+(?P<name>\S+)\s*$", timeout_ms=5000)
    async def view_character(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        authorized = await self._admin_command(message)
        if authorized is None:
            return False, "", 1
        _, stream_id = authorized
        matched = kwargs.get("matched_groups")
        name = str(matched.get("name") or "").strip() if isinstance(matched, dict) else ""
        character = self._repository.find_name(name) if self._repository else None
        if character is None:
            return await self._command_feedback(stream_id, f"角色库中不存在“{name}”。", success=False)
        aliases = "、".join(character.aliases) or "无"
        lines = [f"角色：{character.name}", f"关系：{character.relationship or '未设置'}", f"别名：{aliases}", "外观卡："]
        lines.extend(f"- {card}" for card in character.appearance_cards)
        return await self._command_feedback(stream_id, "\n".join(lines), success=True)

    @Command("character_relationship", description="管理员设置角色关系。", pattern=r"^/设置关系\s+(?P<name>\S+)\s+(?P<relationship>.{1,32})\s*$", timeout_ms=5000)
    async def set_relationship(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        authorized = await self._admin_command(message)
        if authorized is None:
            return False, "", 1
        _, stream_id = authorized
        matched = kwargs.get("matched_groups")
        name = str(matched.get("name") or "").strip() if isinstance(matched, dict) else ""
        relationship = str(matched.get("relationship") or "").strip() if isinstance(matched, dict) else ""
        try:
            async with self._repository_lock:
                character = self._repository.set_relationship(name=name, relationship=relationship) if self._repository else None
            if character is None:
                raise ValueError("角色库不可用")
        except ValueError as exc:
            return await self._command_feedback(stream_id, str(exc), success=False)
        self._invalidate_recognition_cache()
        return await self._command_feedback(stream_id, f"已将“{character.name}”的关系设为“{character.relationship or '未设置'}”。", success=True)

    @Command("character_alias_add", description="管理员添加角色别名。", pattern=r"^/添加别名\s+(?P<name>\S+)\s+(?P<alias>\S+)\s*$", timeout_ms=5000)
    async def add_alias(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        return await self._change_alias(message, kwargs, add=True)

    @Command("character_alias_delete", description="管理员删除角色别名。", pattern=r"^/删除别名\s+(?P<name>\S+)\s+(?P<alias>\S+)\s*$", timeout_ms=5000)
    async def delete_alias(self, message: Any = None, **kwargs: Any) -> tuple[bool, str, int]:
        return await self._change_alias(message, kwargs, add=False)

    async def _change_alias(self, message: Any, kwargs: dict[str, Any], *, add: bool) -> tuple[bool, str, int]:
        authorized = await self._admin_command(message)
        if authorized is None:
            return False, "", 1
        _, stream_id = authorized
        matched = kwargs.get("matched_groups")
        name = str(matched.get("name") or "").strip() if isinstance(matched, dict) else ""
        alias = str(matched.get("alias") or "").strip() if isinstance(matched, dict) else ""
        try:
            async with self._repository_lock:
                character = (
                    self._repository.add_alias(name=name, alias=alias)
                    if add and self._repository
                    else self._repository.remove_alias(name=name, alias=alias)
                    if self._repository
                    else None
                )
            if character is None:
                raise ValueError("角色库不可用")
        except ValueError as exc:
            return await self._command_feedback(stream_id, str(exc), success=False)
        action = "添加" if add else "删除"
        self._invalidate_recognition_cache()
        return await self._command_feedback(stream_id, f"已{action}“{character.name}”的别名“{alias}”。", success=True)

    def _reload_repository(self) -> None:
        filename = Path(self.config.library.file_name).name
        data_dir = self.ctx.paths.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / filename
        legacy_path = Path(__file__).parent / "data" / filename
        if not path.exists() and legacy_path.exists():
            try:
                shutil.copy2(legacy_path, path)
                self.ctx.logger.info("已将旧式角色库迁移到持久化目录: %s", path)
            except OSError as exc:
                self.ctx.logger.warning("迁移旧式角色库失败，将使用空角色库: %s", exc)
        repository = CharacterRepository(path)
        repository.ensure_exists()
        try:
            repository.reload()
        except (OSError, ValueError) as exc:
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                path.replace(backup)
                self.ctx.logger.error("角色库损坏，已备份到 %s 并创建空角色库: %s", backup, exc)
                repository.ensure_exists()
                repository.reload()
            except OSError:
                raise RuntimeError(f"角色库损坏且无法备份: {path}") from exc
        self._repository = repository

    @staticmethod
    def _pending_key(message: dict[str, Any]) -> tuple[str, str]:
        info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
        user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        return str(message.get("session_id") or ""), str(user.get("user_id") or "")

    def _is_admin(self, message: dict[str, Any]) -> bool:
        if self.config.library.allow_unrestricted_admin:
            return True
        allowed = {item.strip() for item in self.config.library.admin_qq if item.strip()}
        if not allowed:
            return False
        return self._pending_key(message)[1] in allowed

    def _discard_expired_additions(self) -> None:
        now = time.monotonic()
        ttl = self.config.library.pending_addition_ttl_seconds
        expired = [key for key, (created_at, _, _) in self._pending_additions.items() if now - created_at > ttl]
        for key in expired:
            self._pending_additions.pop(key, None)

    async def _create_pending_character(
        self, message: dict[str, Any], images: list[dict[str, Any]], pending: tuple[float, str, str]
    ) -> dict[str, Any]:
        created_at, name, relationship = pending
        stream_id = str(message.get("session_id") or "")
        if not self._is_admin(message) or not self._repository:
            if stream_id:
                await self.ctx.send.text("角色库创建权限失效，已取消。", stream_id)
            return {"action": "abort", "custom_result": {"reason": "角色库创建权限失效"}}
        if len(images) != 1:
            self._pending_additions[self._pending_key(message)] = pending
            if stream_id:
                await self.ctx.send.text("请单独发送一张参考图；角色添加仍在等待中。", stream_id)
            return {"action": "abort", "custom_result": {"reason": "角色添加需要单张参考图"}}
        encoded = images[0].get("binary_data_base64")
        if not isinstance(encoded, str) or not encoded:
            if stream_id:
                await self.ctx.send.text("参考图不可读取，已取消创建。", stream_id)
            return {"action": "abort", "custom_result": {"reason": "参考图不可读取"}}
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
            async with self._semaphore:
                cards = await build_appearance_cards(
                    provider=self.config.vision.provider,
                    api_key=self.config.vision.api_key,
                    base_url=self.config.vision.base_url,
                    model=self.config.vision.model,
                    image_bytes=image_bytes,
                    timeout_seconds=self.config.plugin.timeout_seconds,
                    max_upload_bytes=self.config.vision.max_upload_bytes,
                )
            async with self._repository_lock:
                self._repository.upsert(name=name, relationship=relationship, appearance_cards=cards)
        except Exception as exc:
            self.ctx.logger.warning("创建角色库条目失败: %s", exc)
            if time.monotonic() - created_at <= self.config.library.pending_addition_ttl_seconds:
                self._pending_additions[self._pending_key(message)] = pending
            if stream_id:
                await self.ctx.send.text("创建角色库条目失败，仍可在有效期内重新发送一张参考图；详情见日志。", stream_id)
            return {"action": "abort", "custom_result": {"reason": "创建角色库条目失败"}}
        self.ctx.logger.info("管理员创建角色库条目成功: name=%s relationship=%s", name, relationship)
        self._invalidate_recognition_cache()
        if stream_id:
            await self.ctx.send.text(f"已创建角色“{name}”，外观卡 {len(cards)} 条。", stream_id)
        return {"action": "abort", "custom_result": {"reason": f"已创建角色库条目：{name}"}}

    @staticmethod
    def _image_components(message: Any) -> list[dict[str, Any]]:
        if not isinstance(message, dict):
            return []
        raw_message = message.get("raw_message")
        if not isinstance(raw_message, list):
            return []
        return [item for item in raw_message if isinstance(item, dict) and item.get("type") == "image"]

    @staticmethod
    def _decode_image(image: dict[str, Any]) -> bytes | None:
        encoded = image.get("binary_data_base64")
        if not isinstance(encoded, str) or not encoded:
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return None

    @staticmethod
    def _referenced_message_id(message: dict[str, Any]) -> str:
        """Read both the normalized reply_to field and the raw reply component.

        Some adapters only preserve the latter in command payloads.
        """
        direct = str(message.get("reply_to") or "").strip()
        if direct:
            return direct
        raw_message = message.get("raw_message")
        if not isinstance(raw_message, list):
            return ""
        for component in raw_message:
            if not isinstance(component, dict) or component.get("type") != "reply":
                continue
            data = component.get("data")
            if isinstance(data, dict):
                reference_id = str(data.get("target_message_id") or "").strip()
            else:
                reference_id = str(data or "").strip()
            if reference_id:
                return reference_id
        return ""

    async def _recognize_image(self, image: dict[str, Any], *, generation: int | None = None) -> tuple[str, str]:
        generation = self._config_generation if generation is None else generation
        image_hash = str(image.get("hash") or "").strip()
        encoded = image.get("binary_data_base64")
        if not isinstance(encoded, str) or not encoded:
            return "图片未能读取", "图片[未识别]"
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return "图片未能读取", "图片[未识别]"
        if not image_hash:
            image_hash = hashlib.sha256(image_bytes).hexdigest()
        cache_key = f"{generation}:{image_hash}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        catalog = (
            self._repository.private_catalog(limit=self.config.library.max_prompt_characters)
            if self.config.library.enabled and self._repository
            else []
        )
        if self.config.library.enabled and self._repository:
            total = len(self._repository.list_characters())
            if total > len(catalog):
                self.ctx.logger.warning("本地角色库共 %s 个角色，本次仅有前 %s 个进入视觉提示词", total, len(catalog))
        if not self.config.vision.enabled:
            anime_trace_was_available = self._service_available("anime_trace")
            reverse_candidates = await self._reverse_search(image_bytes)
            online_label = self._online_label(reverse_candidates)
            recognized = (self._online_description(reverse_candidates), online_label)
            if anime_trace_was_available and self._service_health["anime_trace"].consecutive_failures == 0:
                self._cache_put(cache_key, recognized)
            return recognized

        result = await self._identify_with_vision(image_bytes, catalog)
        if result is None:
            reverse_candidates = await self._reverse_search(image_bytes)
            recognized = ("图片识别失败", self._online_label(reverse_candidates))
            return recognized
        label = self._label_candidates(result.candidates or ((result.candidate,) if result.candidate else ()))
        if not result.is_anime_character:
            recognized = (result.description, "图片[未识别]")
            self._cache_put(cache_key, recognized)
            return recognized
        reverse_candidates = await self._reverse_search(image_bytes)
        label = self._merge_labels(label, self._online_label(reverse_candidates))
        recognized = (result.description, label)
        self._cache_put(cache_key, recognized)
        return recognized

    def _cache_get(self, image_hash: str) -> tuple[str, str] | None:
        if not image_hash:
            return None
        cached = self._cache.get(image_hash)
        if cached is None:
            return None
        created_at, value = cached
        if time.monotonic() - created_at > self.config.plugin.cache_ttl_seconds:
            self._cache.pop(image_hash, None)
            return None
        self._cache.move_to_end(image_hash)
        return value

    def _cache_put(self, image_hash: str, value: tuple[str, str]) -> None:
        if not image_hash:
            return
        self._cache[image_hash] = (time.monotonic(), value)
        self._cache.move_to_end(image_hash)
        while len(self._cache) > self.config.plugin.cache_max_entries:
            self._cache.popitem(last=False)

    def _remember_latest(self, key: tuple[str, str], image_bytes: bytes, label: str) -> None:
        self._latest_recognitions[key] = (image_bytes, label)
        self._latest_recognitions.move_to_end(key)
        while len(self._latest_recognitions) > self.config.plugin.latest_image_max_entries:
            self._latest_recognitions.popitem(last=False)
        while self._latest_recognitions and sum(len(value[0]) for value in self._latest_recognitions.values()) > self.config.plugin.latest_image_max_total_bytes:
            self._latest_recognitions.popitem(last=False)

    def _invalidate_recognition_cache(self) -> None:
        self._cache.clear()

    async def _identify_with_vision(self, image_bytes: bytes, catalog: list[dict[str, object]]) -> Any:
        if not self._service_available("vision"):
            return None
        try:
            result = await identify_image(
                provider=self.config.vision.provider,
                api_key=self.config.vision.api_key,
                base_url=self.config.vision.base_url,
                model=self.config.vision.model,
                image_bytes=image_bytes,
                private_catalog=catalog,
                timeout_seconds=self.config.plugin.timeout_seconds,
                max_upload_bytes=self.config.vision.max_upload_bytes,
            )
            if result is None:
                raise ValueError("视觉服务返回无效识别结构")
            self._service_succeeded("vision")
            return result
        except Exception as exc:
            self._service_failed("vision", exc)
            self.ctx.logger.warning("图片角色识别失败: %s", exc)
            return None

    def _service_available(self, name: str) -> bool:
        health = self._service_health[name]
        if not health.open_until:
            return True
        if time.monotonic() < health.open_until:
            return False
        health.consecutive_failures = 0
        health.open_until = 0.0
        self.ctx.logger.info("外部服务 %s 熔断冷却结束，恢复尝试", name)
        return True

    def _service_succeeded(self, name: str) -> None:
        health = self._service_health[name]
        health.consecutive_failures = 0
        health.open_until = 0.0

    def _service_failed(self, name: str, error: Exception) -> None:
        health = self._service_health[name]
        health.consecutive_failures += 1
        if health.consecutive_failures < self.config.plugin.circuit_breaker_failures:
            return
        health.open_until = time.monotonic() + self.config.plugin.circuit_breaker_cooldown_seconds
        self.ctx.logger.warning(
            "外部服务 %s 连续失败 %s 次，熔断 %s 秒: %s",
            name,
            health.consecutive_failures,
            self.config.plugin.circuit_breaker_cooldown_seconds,
            error,
        )

    @staticmethod
    def _online_label(hints: tuple[ReverseImageHint, ...]) -> str:
        """Label every distinct confident AnimeTrace person in a multi-character image."""
        names: list[str] = []
        for hint in hints:
            if hint.provider != "AnimeTrace" or not hint.name or hint.not_confident or hint.name in names:
                continue
            names.append(hint.name)
        if not names:
            return "图片[未识别]"
        return f"图片[{'、'.join(names[:6])}]"

    @staticmethod
    def _online_description(hints: tuple[ReverseImageHint, ...]) -> str:
        candidates = [hint for hint in hints if hint.provider == "AnimeTrace" and hint.name and not hint.not_confident]
        names = []
        for candidate in candidates:
            if candidate.name not in names:
                names.append(candidate.name)
        if names:
            return f"二次元角色图片，联网候选：{'、'.join(names[:6])}"
        return "二次元图片（联网未获得明确角色）"

    @staticmethod
    def _merge_labels(primary: str, secondary: str) -> str:
        def names(label: str) -> list[str]:
            if not label.startswith("图片[") or "]" not in label:
                return []
            closing = label.index("]")
            body = label[3:closing]
            if body == "未识别":
                return []
            suffix = label[closing + 1 :]
            values = [item.strip() for item in body.split("、") if item.strip()]
            if suffix and len(values) == 1:
                values[0] += suffix
            return values

        merged: list[str] = []
        seen: set[str] = set()
        for value in [*names(primary), *names(secondary)]:
            plain = value.split("（", 1)[0].casefold()
            if plain and plain not in seen:
                seen.add(plain)
                merged.append(value)
        return f"图片[{'、'.join(merged[:6])}]" if merged else "图片[未识别]"

    def _label_candidate(self, candidate: Any) -> str:
        if candidate is None or candidate.conflicts or len(candidate.evidence) < 2:
            return "图片[未识别]"
        if candidate.kind == "private" and self.config.library.enabled and self._repository:
            character = self._repository.find_name(candidate.name)
            if character is None:
                return "图片[未识别]"
            suffix = f"（{character.relationship}）" if character.relationship else ""
            return f"图片[{character.name}]{suffix}"
        if candidate.kind != "public" or not candidate.name or not candidate.franchise:
            return "图片[未识别]"
        return f"图片[{candidate.name}]"

    def _label_candidates(self, candidates: tuple[Any, ...]) -> str:
        labels = [self._label_candidate(candidate) for candidate in candidates[:6]]
        result = "图片[未识别]"
        for label in labels:
            result = self._merge_labels(result, label)
        return result

    async def _reverse_search(self, image_bytes: bytes) -> tuple[ReverseImageHint, ...]:
        settings = self.config.reverse_image
        results: list[ReverseImageHint] = []
        if settings.anime_trace_enabled and self._service_available("anime_trace"):
            try:
                anime_trace = await search_anime_trace(
                    image_bytes=image_bytes,
                    base_url=settings.anime_trace_url,
                    timeout_seconds=settings.timeout_seconds,
                    max_upload_bytes=settings.anime_trace_max_upload_bytes,
                )
                results.extend(anime_trace)
                self._service_succeeded("anime_trace")
                self.ctx.logger.info(
                    "AnimeTrace 返回 %s 个候选: %s",
                    len(anime_trace),
                    "; ".join(
                        f"{item.name}（{item.franchise}，{'低置信' if item.not_confident else '明确'}）"
                        for item in anime_trace
                    )
                    or "无",
                )
            except Exception as exc:
                self._service_failed("anime_trace", exc)
                self.ctx.logger.info("AnimeTrace 未获得结果: %s", exc)
        return tuple(results[:20])


def create_plugin() -> CharacterKnowledgePlugin:
    return CharacterKnowledgePlugin()
