import asyncio
import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import VisionResult


def detect_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def build_prompt(private_catalog: list[dict[str, object]]) -> str:
    catalog = json.dumps(private_catalog, ensure_ascii=False, separators=(",", ":"))
    return (
        "识别图片中的主要人物角色。只输出 JSON，不要 Markdown。\n"
        "格式：{\"description\":\"不超过100字的客观中文图片描述\",\"is_anime_character\":true,\"kind\":\"private|public|unknown\",\"name\":\"\",\"franchise\":\"\","
        "\"evidence\":[\"最多3条可见特征\"],\"conflicts\":[\"冲突特征\"]}。\n"
        "private 只能从本地角色库中逐字选择 name；看不清、多人难分、没有人物或没有足够证据时必须 unknown。"
        "不要猜测，不要把风格或相似度当成证据。public 仅在你对角色名和作品名都有明确把握时使用。\n"
        "输出额外字段 is_anime_character：仅当图片主体是二次元、动画或游戏风格人物时为 true；真人、风景、物品、文字梗图和非人物图片必须为 false。\n"
        f"本地角色库：{catalog}"
    )


async def identify_image(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    image_bytes: bytes,
    private_catalog: list[dict[str, object]],
    timeout_seconds: int,
) -> VisionResult | None:
    prompt = build_prompt(private_catalog)
    if provider == "gemini":
        content = await _call_gemini(api_key, base_url, model, prompt, image_bytes, timeout_seconds)
    elif provider == "openai":
        content = await _call_openai(api_key, base_url, model, prompt, image_bytes, timeout_seconds)
    else:
        raise ValueError(f"不支持的视觉提供方: {provider}")
    try:
        return VisionResult.from_dict(json.loads(_extract_json(content)))
    except (json.JSONDecodeError, ValueError):
        return None


async def build_appearance_cards(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    image_bytes: bytes,
    timeout_seconds: int,
) -> list[str]:
    prompt = (
        "为管理员创建二次元角色的可维护外观卡。只输出 JSON：{\"appearance_cards\":[\"...\"]}。"
        "给出3到5条中文客观特征，优先发色发型、眼睛、标志性头饰/饰品、稳定服装设计；"
        "不要写背景、姿势、表情、画质，也不要猜角色名称。每条不超过55字。"
    )
    if provider == "gemini":
        content = await _call_gemini(api_key, base_url, model, prompt, image_bytes, timeout_seconds)
    elif provider == "openai":
        content = await _call_openai(api_key, base_url, model, prompt, image_bytes, timeout_seconds)
    else:
        raise ValueError(f"不支持的视觉提供方: {provider}")
    parsed = json.loads(_extract_json(content))
    values = parsed.get("appearance_cards") if isinstance(parsed, dict) else None
    if not isinstance(values, list):
        raise ValueError("外观卡响应缺少 appearance_cards")
    cards = [str(value).strip() for value in values if str(value).strip()]
    if len(cards) < 2:
        raise ValueError("外观卡不足两条")
    return cards[:5]


async def _call_openai(
    api_key: str, base_url: str, model: str, prompt: str, image_bytes: bytes, timeout_seconds: int
) -> str:
    mime_type = detect_image_mime_type(image_bytes)
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            }
        ],
    }
    url = base_url.rstrip("/") + "/chat/completions"
    response = await _post_json(url, {"Authorization": f"Bearer {api_key}"}, payload, timeout_seconds)
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("OpenAI 兼容接口未返回 choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("OpenAI 兼容接口未返回文本内容")
    return message["content"]


async def _call_gemini(
    api_key: str, base_url: str, model: str, prompt: str, image_bytes: bytes, timeout_seconds: int
) -> str:
    url = base_url.rstrip("/") + f"/models/{model}:generateContent?key={api_key}"
    payload = {
        "generationConfig": {"temperature": 0, "maxOutputTokens": 300, "responseMimeType": "application/json"},
        "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": detect_image_mime_type(image_bytes), "data": base64.b64encode(image_bytes).decode("ascii")}}]}],
    }
    response = await _post_json(url, {}, payload, timeout_seconds)
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ValueError("Gemini 未返回 candidates")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise ValueError("Gemini 未返回内容部分")
    return "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))


async def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json", **headers})

    def send() -> dict[str, Any]:
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"视觉接口 HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"视觉接口连接失败: {exc.reason}") from exc

    return await asyncio.to_thread(send)


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    return stripped[start : end + 1]
