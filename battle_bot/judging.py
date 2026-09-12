from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from .models import Player
from .settings import AI_JUDGE_TIMEOUT_SECONDS


# MODEL_NAME is the primary deployment setting; GEMINI_MODEL remains supported.
GEMINI_MODEL = os.getenv("MODEL_NAME") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"


LOGGER = logging.getLogger("battle_of_creation.judging")
_AI_SEMAPHORE = asyncio.Semaphore(3)


@dataclass(frozen=True)
class Judgement:
    winner_id: int
    loser_id: int
    reason: str
    source: str = "local"


_POWER_WORDS = {
    "battle",
    "blade",
    "boss",
    "combat",
    "dragon",
    "fighter",
    "god",
    "king",
    "magic",
    "power",
    "sword",
    "warrior",
}
_CHAOS_WORDS = {"chaos", "cat", "funny", "meow", "monster", "nine", "tail", "tails"}


def _score(player: Player) -> tuple[int, int]:
    """Score only explicit capability signals, never writing quality or length."""
    words = re.findall(r"[a-z0-9]+", player.submission.lower())
    capability_signal = sum(4 for word in words if word in _POWER_WORDS)
    # A stable tie-breaker makes the no-AI judge reproducible after a restart.
    tie_break = int(hashlib.sha256(
        f"{player.user_id}:{player.submission.casefold()}".encode("utf-8")
    ).hexdigest()[:8], 16) % 7
    return capability_signal, tie_break


def _submission_signal(submission: str) -> str:
    words = re.findall(r"[a-z0-9]+", submission.lower())
    if any(word in _POWER_WORDS for word in words):
        return "battle-ready power and pressure"
    if any(word in _CHAOS_WORDS for word in words):
        return "unpredictable chaos and surprise value"
    return "the clearest stated matchup advantage"


def _variant_index(player_one: Player, player_two: Player, round_number: int) -> int:
    seed = (
        f"{round_number}:{player_one.user_id}:{player_one.submission.casefold()}:"
        f"{player_two.user_id}:{player_two.submission.casefold()}"
    )
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % 8


def _ensure_both_submissions_in_reason(
    reason: str,
    winner: Player,
    loser: Player,
) -> str:
    reason = " ".join(reason.strip().split())
    if not reason:
        return ""
    if winner.submission.casefold() not in reason.casefold():
        reason = f"{winner.submission} wins because {reason}"
    if loser.submission.casefold() not in reason.casefold():
        reason = f"{reason} It overcame {loser.submission} in this matchup."
    return reason[:950]


def judge_match(
    player_one: Player,
    player_two: Player,
    round_number: int = 0,
) -> Judgement:
    score_one = _score(player_one)
    score_two = _score(player_two)
    if score_one > score_two or score_one == score_two and player_one.user_id < player_two.user_id:
        winner, loser = player_one, player_two
    else:
        winner, loser = player_two, player_one

    signal = _submission_signal(winner.submission)
    close = abs(sum(score_one) - sum(score_two)) <= 2
    margin = "by the slimmest possible margin" if close else "with a clear matchup edge"
    fallback_reasons = (
        f"{winner.submission} beats {loser.submission} {margin} because it showed {signal}. "
        f"By decree of the arena's extremely unbiased royal court, {loser.submission} may now appeal to the nearest wall 😭",
        f"The evidence on the card favors {winner.submission} over {loser.submission}: {signal} is the deciding factor. "
        f"The lord of the bracket has spoken, and {loser.submission}'s victory speech has been postponed indefinitely 👑",
        f"{winner.submission} takes the crown from {loser.submission} on {signal}; that is the only matchup evidence this fallback judge can safely use. "
        f"A tragic day for {loser.submission}, whose royal entrance had excellent production value but no winning clause 😭",
        f"In this round, {winner.submission} had {signal}, while {loser.submission} brought enough chaos to make the scoreboard nervous. "
        f"Unfortunately, the throne is occupied and the crown is refusing visitors 👑",
        f"The stated abilities give {winner.submission} the advantage over {loser.submission}, especially through {signal}. "
        f"The court considered a retrial, then remembered it enjoys being dramatically correct 😭",
        f"{winner.submission} wins this clash with {signal} against {loser.submission}; the matchup was {"painfully close" if close else "decisively tilted"}. "
        f"Still, the royal scoreboard has issued its ruling, and complaints must be submitted in triplicate 👑",
        f"{winner.submission} clears {loser.submission} because {signal} matters more here than pure theatrical confidence. "
        f"The defeated side may keep its dignity, its soundtrack, and absolutely none of the crown 😭",
        f"The card gives {winner.submission} the win over {loser.submission} through {signal}. "
        f"Somewhere, a sarcastic monarch is whispering, 'A bold strategy—shame it lost' 👑",
        f"{winner.submission} edges past {loser.submission} on the matchup evidence: {signal}. "
        f"The arena's noble committee has ruled that dramatic posing is not, by itself, a legal victory condition 😭",
        f"The lore available on the card gives {winner.submission} the advantage over {loser.submission} through {signal}. "
        f"The crown has chosen its champion, and {loser.submission} has been invited to leave through the extremely ceremonial side door 👑",
    )
    reason = fallback_reasons[_variant_index(player_one, player_two, round_number)]
    return Judgement(winner_id=winner.user_id, loser_id=loser.user_id, reason=reason, source="local")


async def judge_match_with_ai(
    player_one: Player,
    player_two: Player,
    round_number: int,
) -> Judgement:
    """Use AI for fresh matchup commentary, with a deterministic local fallback."""
    local_judgement = judge_match(player_one, player_two, round_number)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        LOGGER.warning("Gemini judge disabled: GEMINI_API_KEY is not set; using local judge")
        return local_judgement

    prompt = f"""
You are the evidence-based judge for a friendly Discord tournament called Battle
of Creation. Compare two submissions and choose who would win in a neutral
one-on-one matchup.

Submission one:
{player_one.submission}

Submission two:
{player_two.submission}

Research instructions:
- Use Google Search to identify each submission when it refers to a known
  character, real person, creature, object, technology, historical subject, or
  other real/canonical thing.
- Search the exact submitted name first, then search the name plus feats,
  abilities, specifications, accomplishments, or documented capabilities.
- Prefer reliable primary, official, encyclopedic, or otherwise well-supported
  sources. Ignore fan exaggeration, unsupported claims, and search-result
  wording that only repeats the submission.
- If a submission is original, ambiguous, or has no trustworthy search results,
  use only the abilities and feats explicitly stated in that submission. For a
  normal real object, animal, or person, use realistic capabilities.
- Do not invent powers, feats, equipment, transformations, or versions.
- Use the strongest clearly established standard version only when the identity
  is unambiguous. If versions conflict, state the conservative interpretation.
- Decide from overall demonstrated feats and capabilities: attack power,
  durability, speed, range, abilities, intelligence, experience, stamina,
  matchup advantages, and relevant limitations.
- Description length, word count, grammar, writing quality, and level of detail
  must never count as evidence or an advantage. A short submission can win.

Response rules:
- Choose exactly one winner. The winner must be either "one" or "two".
- Write one fresh reason of 2-4 sentences.
- Mention both exact submission names in the reason.
- Explain the decisive feat or capability advantage in this matchup.
- The tone must be approximately 50% funny arena commentary, 30% sarcastic lord/royal roast, and 20% real lore or researched matchup evidence.
- The real-lore portion must name a concrete feat, ability, limitation, or matchup fact; never invent lore. For original or ambiguous submissions, use only what the submission explicitly states.
- The sarcastic lord tone should be witty and theatrical, never hateful, discriminatory, threatening, or genuinely cruel.
- Write a fresh reason every time. Do not reuse sentence frames, generic scoreboard language, or filler jokes.
- Never use these stale phrases: "clearer explicit capability signal", "stronger overall matchup profile", "narrowly edges", or "imaginary entrance fireworks".
- Return only valid JSON: {{"winner":"one"|"two","reason":"..."}}
"""

    try:
        async with _AI_SEMAPHORE:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=AI_JUDGE_TIMEOUT_SECONDS * 1000,
                ),
            )
            try:
                request_kwargs = {
                    "model": GEMINI_MODEL,
                    "input": prompt,
                    "response_format": {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "winner": {
                                    "type": "string",
                                    "enum": ["one", "two"],
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["winner", "reason"],
                            "additionalProperties": False,
                        },
                    },
                }
                try:
                    response = await client.aio.interactions.create(
                        **request_kwargs,
                        tools=[{"type": "google_search"}],
                    )
                except Exception as grounded_error:
                    error_text = str(grounded_error).lower()
                    quota_markers = ("quota", "429", "too_many_requests", "rate limit")
                    if not any(marker in error_text for marker in quota_markers):
                        raise
                    LOGGER.warning(
                        "Gemini Search grounding hit quota; retrying without Search: %s",
                        type(grounded_error).__name__,
                    )
                    response = await client.aio.interactions.create(**request_kwargs)
            finally:
                await client.aio.aclose()
        content = getattr(response, "output_text", "") or ""
        data = json.loads(content)
        winner_choice = str(data.get("winner", "")).lower().strip()
        raw_reason = str(data.get("reason", "")).strip()
        if winner_choice == "one":
            winner, loser = player_one, player_two
        elif winner_choice == "two":
            winner, loser = player_two, player_one
        else:
            return local_judgement
        reason = _ensure_both_submissions_in_reason(raw_reason, winner, loser)
        if len(reason) < 30:
            LOGGER.warning("Gemini judge returned an invalid reason; using local judge")
            return local_judgement
        return Judgement(winner_id=winner.user_id, loser_id=loser.user_id, reason=reason, source="gemini")
    except Exception:
        LOGGER.exception("Gemini judge request failed; using local judge")
        return local_judgement
