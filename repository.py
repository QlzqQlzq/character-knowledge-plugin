import json
import hashlib
from pathlib import Path

from .models import Character


class CharacterRepository:
    """只读加载管理员维护的本地角色库。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._characters: tuple[Character, ...] = ()

    def reload(self) -> None:
        if not self._path.exists():
            self._characters = ()
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        entries = raw.get("characters") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise ValueError("角色库必须包含 characters 列表")
        characters: list[Character] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            character_id = str(entry.get("id") or "").strip()
            name = str(entry.get("name") or "").strip()
            aliases = entry.get("aliases") or []
            appearance_cards = entry.get("appearance_cards") or []
            if not character_id or not name or not isinstance(aliases, list) or not isinstance(appearance_cards, list):
                continue
            characters.append(
                Character(
                    character_id=character_id,
                    name=name,
                    aliases=tuple(str(item).strip() for item in aliases if str(item).strip()),
                    relationship=str(entry.get("relationship") or "").strip(),
                    appearance_cards=tuple(str(item).strip() for item in appearance_cards if str(item).strip()),
                )
            )
        self._characters = tuple(characters)

    def ensure_exists(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"characters": []})

    def private_catalog(self, *, limit: int = 40) -> list[dict[str, object]]:
        return [
            {
                "id": character.character_id,
                "name": character.name,
                "aliases": list(character.aliases),
                "appearance_cards": list(character.appearance_cards),
            }
            for character in self._characters[:limit]
        ]

    def _write(self, raw: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._path)

    def find_name(self, proposed_name: str) -> Character | None:
        normalized = proposed_name.strip().casefold()
        if not normalized:
            return None
        for character in self._characters:
            names = (character.name, *character.aliases)
            if any(normalized == name.casefold() for name in names):
                return character
        return None

    def upsert(self, *, name: str, relationship: str, appearance_cards: list[str]) -> Character:
        """写入管理员确认的新角色，名称相同则覆盖其外观卡。"""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("角色名不能为空")
        cards = [card.strip() for card in appearance_cards if card.strip()]
        if len(cards) < 2:
            raise ValueError("至少需要两条稳定外观卡")
        raw = {"characters": []}
        if self._path.exists():
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("characters"), list):
                raw = loaded
        existing_entry = next(
            (
                entry
                for entry in raw["characters"]
                if isinstance(entry, dict)
                and str(entry.get("name") or "").strip().casefold() == normalized_name.casefold()
            ),
            None,
        )
        for character in self._characters:
            if character.character_id == str((existing_entry or {}).get("id") or ""):
                continue
            if normalized_name.casefold() in {character.name.casefold(), *(alias.casefold() for alias in character.aliases)}:
                raise ValueError(f"名称“{normalized_name}”已被角色“{character.name}”使用")
        entries = [entry for entry in raw["characters"] if entry is not existing_entry]
        # Names are commonly Chinese, so an ASCII-only slug would collapse every such
        # entry to the same ID. The digest is stable and collision-resistant in practice.
        identifier = str((existing_entry or {}).get("id") or "").strip() or f"char-{hashlib.sha256(normalized_name.casefold().encode('utf-8')).hexdigest()[:12]}"
        entries.append(
            {
                "id": identifier,
                "name": normalized_name,
                "aliases": list((existing_entry or {}).get("aliases") or []),
                "relationship": relationship.strip(),
                "appearance_cards": cards[:5],
            }
        )
        raw["characters"] = entries
        self._write(raw)
        self.reload()
        character = self.find_name(normalized_name)
        if character is None:
            raise RuntimeError("角色库写入后未找到新角色")
        return character

    def append_appearance_cards(self, *, name: str, appearance_cards: list[str], limit: int = 15) -> Character:
        """合并管理员从已发图片确认的补充外观卡，保留既有身份和关系。"""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("角色名不能为空")
        cards = [card.strip() for card in appearance_cards if card.strip()]
        if len(cards) < 2:
            raise ValueError("至少需要两条稳定外观卡")
        raw = {"characters": []}
        if self._path.exists():
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("characters"), list):
                raw = loaded
        target: dict[str, object] | None = None
        for entry in raw["characters"]:
            if isinstance(entry, dict) and str(entry.get("name") or "").strip().casefold() == normalized_name.casefold():
                target = entry
                break
        if target is None:
            raise ValueError(f"角色库中不存在“{normalized_name}”")
        existing = target.get("appearance_cards")
        if not isinstance(existing, list):
            existing = []
        merged: list[str] = []
        seen: set[str] = set()
        for card in [*existing, *cards]:
            normalized_card = str(card).strip()
            key = normalized_card.casefold()
            if normalized_card and key not in seen:
                seen.add(key)
                merged.append(normalized_card)
        target["appearance_cards"] = merged[-limit:]
        self._write(raw)
        self.reload()
        character = self.find_name(normalized_name)
        if character is None:
            raise RuntimeError("角色库修正后未找到角色")
        return character

    def list_characters(self) -> tuple[Character, ...]:
        return self._characters

    def set_relationship(self, *, name: str, relationship: str) -> Character:
        return self._update_entry(name=name, update=lambda entry: entry.__setitem__("relationship", relationship.strip()))

    def add_alias(self, *, name: str, alias: str) -> Character:
        normalized_alias = alias.strip()
        if not normalized_alias:
            raise ValueError("别名不能为空")
        conflict = self.find_name(normalized_alias)
        target = self.find_name(name)
        if conflict is not None and (target is None or conflict.character_id != target.character_id):
            raise ValueError(f"名称或别名“{normalized_alias}”已属于角色“{conflict.name}”")

        def update(entry: dict[str, object]) -> None:
            aliases = entry.get("aliases")
            values = [str(item).strip() for item in aliases] if isinstance(aliases, list) else []
            if normalized_alias.casefold() not in {item.casefold() for item in values}:
                values.append(normalized_alias)
            entry["aliases"] = values

        return self._update_entry(name=name, update=update)

    def remove_alias(self, *, name: str, alias: str) -> Character:
        normalized_alias = alias.strip()
        if not normalized_alias:
            raise ValueError("别名不能为空")

        def update(entry: dict[str, object]) -> None:
            aliases = entry.get("aliases")
            values = [str(item).strip() for item in aliases] if isinstance(aliases, list) else []
            filtered = [item for item in values if item.casefold() != normalized_alias.casefold()]
            if len(filtered) == len(values):
                raise ValueError(f"角色“{entry.get('name')}”没有别名“{normalized_alias}”")
            entry["aliases"] = filtered

        return self._update_entry(name=name, update=update)

    def _update_entry(self, *, name: str, update: object) -> Character:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("角色名不能为空")
        if not callable(update):
            raise TypeError("update 必须是可调用对象")
        raw = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {"characters": []}
        entries = raw.get("characters") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise ValueError("角色库必须包含 characters 列表")
        target = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict)
                and normalized_name.casefold()
                in {str(entry.get("name") or "").strip().casefold(), *(str(item).strip().casefold() for item in entry.get("aliases") or [])}
            ),
            None,
        )
        if target is None:
            raise ValueError(f"角色库中不存在“{normalized_name}”")
        update(target)
        self._write(raw)
        self.reload()
        character = self.find_name(str(target.get("name") or ""))
        if character is None:
            raise RuntimeError("角色库更新后未找到角色")
        return character
