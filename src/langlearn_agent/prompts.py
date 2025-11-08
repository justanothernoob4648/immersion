from __future__ import annotations

from textwrap import dedent

from .config import LanguageLearningSession


def build_system_prompt(session: LanguageLearningSession) -> str:
    """Compose the system prompt that programs the tutoring behavior."""

    props_clause = (
        "Surface immersive props such as menus, tickets, or signage in concise markdown, "
        "label them clearly, and keep them under 120 words."
        if session.deliver_props
        else "Focus on vivid verbal descriptions instead of standalone props."
    )

    vocab_clause = "".join(
        f"\n- Key vocabulary to recycle naturally: '{word}'"
        for word in session.supplemental_vocab
    )

    return dedent(
        f"""
        You are an encouraging language coach conducting a role play with {session.learner_name}.
        Stay in character for the scenario the user provided, answer primarily in {session.target_language},
        and keep your turns crisp (<12 seconds of speech).
        {props_clause}
        When the learner hesitates, explicitly mispronounces repeated syllables, or stays silent,
        briefly restate the missing phrase in {session.native_language} and provide an immediate
        retry prompt back in {session.target_language}. Avoid long grammar lectures mid-dialogue;
        instead, inject short meta tips (<15 words) between turns.
        {vocab_clause}
        Always close each turn with an explicit, actionable prompt that nudges the learner to
        respond verbally.
        """
    ).strip()


def build_bootstrap_messages(session: LanguageLearningSession) -> list[dict[str, str]]:
    """Seed the LLM context with the scenario and prop instructions."""

    system_prompt = build_system_prompt(session)
    scenario_brief = dedent(
        f"""
        Scenario briefing:
        - Learner proficiency: {session.proficiency}
        - Requested scene: {session.scenario_prompt}
        - Native language: {session.native_language}
        - Target language: {session.target_language}
        - Output props as: {session.prop_channel}
        Start by greeting the learner, summarizing the scene, and sharing the first prop/stimulus.
        """
    ).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": scenario_brief},
    ]


__all__ = [
    "build_system_prompt",
    "build_bootstrap_messages",
]
