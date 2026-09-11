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


LOGGER = logging.getLogger("battle_of_creation.judging")
_AI_SEMAPHORE = asyncio.Semaphore(3)


@dataclass(frozen=True)
class Judgement:
    winner_id: int
    loser_id: int
    reason: str


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
_GENERIC_ROLES = {
    "animal",
    "baby",
    "cat",
    "child",
    "dancer",
    "dog",
    "farmer",
    "fighter",
    "human",
    "person",
    "soldier",
    "student",
    "teacher",
    "worker",
}


def _is_generic_role(submission: str) -> bool:
    """Return whether a submission is only an ordinary role or broad noun."""
    words = re.findall(r"[a-z0-9]+", submission.casefold())
    return bool(words) and len(words) <= 3 and all(word in _GENERIC_ROLES for word in words)


def _score(player: Player) -> tuple[int, int]:
    """Score only explicit capability signals, never writing quality or length."""
    words = re.findall(r"[a-z0-9]+", player.submission.lower())
    capability_signal = sum(4 for word in words if word in _POWER_WORDS)
    # A generic role has no special feats by itself. This prevents the local
    # fallback from selecting an ordinary "dancer" over an identifiable entity
    # merely because the latter's name contains no capability keywords.
    generic_penalty = -8 if _is_generic_role(player.submission) else 0
    # A stable tie-breaker makes the no-AI judge reproducible after a restart.
    tie_break = int(hashlib.sha256(
        f"{player.user_id}:{player.submission.casefold()}".encode("utf-8")
    ).hexdigest()[:8], 16) % 7
    return capability_signal + generic_penalty, tie_break


def _submission_signal(submission: str) -> str:
    words = re.findall(r"[a-z0-9]+", submission.lower())
    if any(word in _POWER_WORDS for word in words):
        return "stronger battle-ready wording"
    if any(word in _CHAOS_WORDS for word in words):
        return "a memorable chaos factor"
    return "the clearer explicit capability signal"


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

    winner_submission = winner.submission
    loser_submission = loser.submission
    signal = _submission_signal(winner_submission)
    close = abs(sum(score_one) - sum(score_two)) <= 2
    logical = (
        f"{winner_submission} gets the edge over {loser_submission} because it showed "
        f"{signal} and a stronger overall matchup profile."
    )
    funny_bits = (
        "The judge gave it the tiny crown and told the other choice to update its "
        "battle résumé 😭",
        "Basically, the arena heard that submission and immediately turned the "
        "dramatic music up 😭",
        "The loser still had serious main-character energy, but the scoreboard "
        "had already chosen violence—in the harmless tournament sense 😭",
        "One choice brought the strategy; the other accidentally brought a very "
        "confident reaction image 😭",
        "The matchup was close enough to need a replay, but the winner's imaginary "
        "entrance fireworks settled the argument 😭",
        "That is the sort of result that makes the defeated submission stare at "
        "the bracket like it has personally betrayed them 😭",
        "The winner walked in with a plan; the loser walked in with excellent "
        "plot-twist potential 😭",
        "The arena committee has reviewed the evidence and confiscated the loser's "
        "victory music for now 😭",
    )
    funny = funny_bits[_variant_index(player_one, player_two, round_number)]
    reason = f"{logical} {funny}"
    if close:
        reason = (
            f"{winner_submission} narrowly edges {loser_submission}: {signal} gave it "
            f"the logical advantage, even though both choices made this a genuinely "
            f"close call. {funny}"
        )
    return Judgement(winner_id=winner.user_id, loser_id=loser.user_id, reason=reason)


async def judge_match_with_ai(
    player_one: Player,
    player_two: Player,
    round_number: int,
) -> Judgement:
    """Use AI for fresh matchup commentary, with a deterministic local fallback."""
    local_judgement = judge_match(player_one, player_two, round_number)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
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
- Treat a generic role or broad noun by itself (for example "dancer",
  "student", "person", "cat", or "object") as an ordinary real-world entity,
  not as a secretly powerful fictional character. It can only become a
  specific fictional or exceptional identity if the submission clearly names
  that identity or states the relevant feats.
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
- Never award a win to a generic role over a clearly identifiable character or
  entity solely because the generic role was described more favorably.

Response rules:
- Choose exactly one winner. The winner must be either "one" or "two".
- Write one fresh reason of 2-4 sentences.
- Mention both exact submission names in the reason.
- Explain the decisive feat or capability advantage in this matchup.
- Make the reason roughly half logical comparison and half playful humor.
- Be friendly. No hateful, discriminatory, threatening, or genuinely insulting content.
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
                response = await client.aio.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=8192,
                        tools=[
                            types.Tool(
                                google_search=types.GoogleSearch(),
                            )
                        ],
                    ),
                )
            finally:
                await client.aio.aclose()
        content = response.text or ""
        data = json.loads(content)
        winner_choice = str(data.get("winner", "")).lower().strip()
        raw_reason = str(data.get("reason", "")).strip()
        if winner_choice == "one":
            winner, loser = player_one, player_two
        elif winner_choice == "two":
            winner, loser = player_two, player_one
        else:
            return local_judgement
        if _is_generic_role(winner.submission) and not _is_generic_role(loser.submission):
            LOGGER.warning(
                "Rejected generic-role AI winner %r over identifiable submission %r",
                winner.submission,
                loser.submission,
            )
            return local_judgement
        reason = _ensure_both_submissions_in_reason(raw_reason, winner, loser)
        if len(reason) < 30:
            return local_judgement
        return Judgement(winner_id=winner.user_id, loser_id=loser.user_id, reason=reason)
    except Exception as error:
        LOGGER.warning("AI judge unavailable; using local judge: %s", type(error).__name__)
        return local_judgement
