import asyncio
import io
import json
import uuid
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ReverseImageHint:
    provider: str
    text: str
    name: str = ""
    franchise: str = ""
    not_confident: bool = True


def prepare_anime_trace_image(image_bytes: bytes, *, max_bytes: int = 900_000) -> tuple[bytes, str]:
    """Make a bounded JPEG upload copy for AnimeTrace without changing the original.

    AnimeTrace rejects larger multipart requests with HTTP 413. The exact limit is
    not documented, so leave some margin below one MiB and reduce size gradually.
    """
    if len(image_bytes) <= max_bytes:
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return image_bytes, "image/jpeg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return image_bytes, "image/gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return image_bytes, "image/webp"
        return image_bytes, "image/png"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as source:
            original = source.convert("RGB")
            # Repeatedly reduce both geometry and quality. This guarantees that
            # even high-entropy PNG screenshots do not fall back to the rejected
            # original upload merely because one JPEG pass was still too large.
            for edge in (1600, 1280, 1024, 800, 640, 512, 384, 256):
                image = original.copy()
                image.thumbnail((edge, edge))
                for quality in (88, 80, 72, 64, 56, 48, 40):
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=quality, optimize=True)
                    result = output.getvalue()
                    if len(result) <= max_bytes:
                        return result, "image/jpeg"
    except Exception as exc:
        raise ValueError("无法为 AnimeTrace 压缩图片") from exc
    raise ValueError(f"图片压缩后仍超过 AnimeTrace 上传上限 {max_bytes} bytes")


def _multipart_body(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    boundary = f"----MaiBotCharacter{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="image"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("ascii")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    return head + image_bytes + tail, boundary


async def search_anime_trace(
    *, image_bytes: bytes, base_url: str, timeout_seconds: int, max_upload_bytes: int = 900_000
) -> tuple[ReverseImageHint, ...]:
    """Return untrusted AnimeTrace candidates for a VLM to verify visually.

    AnimeTrace is a recognition hint service, not an authority: it can return a
    plausible but incorrect work or character even when ``not_confident`` is false.
    """
    upload_bytes, upload_mime_type = prepare_anime_trace_image(image_bytes, max_bytes=max_upload_bytes)
    body, boundary = _multipart_body(upload_bytes, upload_mime_type)
    url = base_url.rstrip("/") + "/v1/search"
    request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    def send() -> dict[str, object]:
        with urlopen(request, timeout=timeout_seconds) as response:
            content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise RuntimeError("AnimeTrace 响应超过 2000000 bytes")
            return json.loads(content.decode("utf-8"))

    response = await asyncio.to_thread(send)
    entries = response.get("data") if isinstance(response, dict) else None
    if not isinstance(entries, list):
        return ()

    candidates: list[ReverseImageHint] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        not_confident = bool(entry.get("not_confident"))
        characters = entry.get("character")
        if not isinstance(characters, list):
            continue
        for character in characters:
            if not isinstance(character, dict):
                continue
            name = str(character.get("character") or "").strip()
            franchise = str(character.get("work") or "").strip()
            key = (name, franchise)
            if not name or not franchise or key in seen:
                continue
            seen.add(key)
            confidence = "低置信" if not_confident else "服务报告置信"
            candidates.append(
                ReverseImageHint(
                    "AnimeTrace", f"{name}｜{franchise}｜{confidence}", name, franchise, not_confident
                )
            )
    return tuple(candidates[:12])
