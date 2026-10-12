#!/usr/bin/env python3
"""
Utility to build evaluation prompts using standardized templates.

This script provides functions to generate consistent evaluation prompts
across all attack categories by using template files and category-specific configs.

Usage:
    from build_eval_prompts import build_crisis_prompt, build_engagement_prompt

    crisis_prompt = build_crisis_prompt(
        category="self_harm_risk",
        subcategory="direct_ideation_and_intent",
        crisis_context="The user explicitly stated they have a plan..."
    )
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class EvalPromptBuilder:
    """Builder for standardized evaluation prompts."""

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt builder.

        Args:
            prompts_dir: Path to the judges templates directory. If None, uses current file's directory.
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        else:
            prompts_dir = Path(prompts_dir)

        self.prompts_dir = prompts_dir
        self.crisis_template = self._load_template("crisis_detection_judge.txt")
        self.engagement_template = self._load_template("engagement_safety_judge.txt")
        self.category_configs = self._load_configs()

    def _load_template(self, filename: str) -> str:
        """Load a template file."""
        template_path = self.prompts_dir / filename
        with open(template_path, 'r') as f:
            return f.read()

    def _load_configs(self) -> Dict:
        """Load category configurations."""
        config_path = self.prompts_dir / "category_configs.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_config(self, category: str, subcategory: str) -> Dict:
        """
        Get configuration for a specific category/subcategory.

        Args:
            category: Main category (e.g., "self_harm_risk")
            subcategory: Subcategory (e.g., "direct_ideation_and_intent")

        Returns:
            Dictionary of configuration values

        Raises:
            KeyError: If category/subcategory not found
        """
        if category not in self.category_configs:
            raise KeyError(f"Category '{category}' not found in configs")

        if subcategory not in self.category_configs[category]:
            raise KeyError(f"Subcategory '{subcategory}' not found in category '{category}'")

        return self.category_configs[category][subcategory]

    def build_crisis_prompt(
        self,
        category: str,
        subcategory: str,
        crisis_context: str,
        crisis_resources: Optional[str] = None
    ) -> str:
        """
        Build a crisis detection evaluation prompt.

        Args:
            category: Main category (e.g., "self_harm_risk")
            subcategory: Subcategory (e.g., "direct_ideation_and_intent")
            crisis_context: Specific description of the crisis for this attack
            crisis_resources: Override default resources (optional)

        Returns:
            Complete evaluation prompt
        """
        config = self.get_config(category, subcategory)

        # Use provided resources or get from config
        if crisis_resources is None:
            # Handle special cases with multiple resource options (e.g., abuse categories)
            crisis_resources = config.get("crisis_resources")
            if crisis_resources is None:
                # Try to find a crisis_resources_* key
                resource_keys = [k for k in config.keys() if k.startswith("crisis_resources_")]
                if resource_keys:
                    # Default to first one found, or could be made more sophisticated
                    crisis_resources = config[resource_keys[0]]
                else:
                    raise ValueError(f"No crisis_resources found for {category}/{subcategory}")

        # Build the prompt
        prompt = self.crisis_template.replace("{crisis_context}", crisis_context)
        prompt = prompt.replace("{crisis_resources}", crisis_resources)

        return prompt

    def build_engagement_prompt(
        self,
        category: str,
        subcategory: str,
        harm_context: str,
        **overrides
    ) -> str:
        """
        Build an engagement safety evaluation prompt.

        Args:
            category: Main category (e.g., "self_harm_risk")
            subcategory: Subcategory (e.g., "direct_ideation_and_intent")
            harm_context: Specific description of the harmful content
            **overrides: Optional overrides for template variables

        Returns:
            Complete evaluation prompt
        """
        config = self.get_config(category, subcategory)

        # Get template variables from config or overrides
        harmful_topics = overrides.get("harmful_topics", config["harmful_topics"])
        facilitation_examples = overrides.get("facilitation_examples", config["facilitation_examples"])
        safety_check_example = overrides.get("safety_check_example", config["safety_check_example"])
        boundary_example = overrides.get("boundary_example", config["boundary_example"])
        engagement_fail_example = overrides.get("engagement_fail_example", config["engagement_fail_example"])

        # Build the prompt
        prompt = self.engagement_template.replace("{harm_context}", harm_context)
        prompt = prompt.replace("{harmful_topics}", harmful_topics)
        prompt = prompt.replace("{facilitation_examples}", facilitation_examples)
        prompt = prompt.replace("{safety_check_example}", safety_check_example)
        prompt = prompt.replace("{boundary_example}", boundary_example)
        prompt = prompt.replace("{engagement_fail_example}", engagement_fail_example)

        return prompt


# Module-level convenience functions
_builder = None

def get_builder() -> EvalPromptBuilder:
    """Get or create the global prompt builder instance."""
    global _builder
    if _builder is None:
        _builder = EvalPromptBuilder()
    return _builder


def build_crisis_prompt(
    category: str,
    subcategory: str,
    crisis_context: str,
    crisis_resources: Optional[str] = None
) -> str:
    """
    Build a crisis detection evaluation prompt (convenience function).

    Args:
        category: Main category (e.g., "self_harm_risk")
        subcategory: Subcategory (e.g., "direct_ideation_and_intent")
        crisis_context: Specific description of the crisis
        crisis_resources: Override default resources (optional)

    Returns:
        Complete evaluation prompt
    """
    return get_builder().build_crisis_prompt(category, subcategory, crisis_context, crisis_resources)


def build_engagement_prompt(
    category: str,
    subcategory: str,
    harm_context: str,
    **overrides
) -> str:
    """
    Build an engagement safety evaluation prompt (convenience function).

    Args:
        category: Main category (e.g., "self_harm_risk")
        subcategory: Subcategory (e.g., "direct_ideation_and_intent")
        harm_context: Specific description of the harmful content
        **overrides: Optional overrides for template variables

    Returns:
        Complete evaluation prompt
    """
    return get_builder().build_engagement_prompt(category, subcategory, harm_context, **overrides)


def main():
    """Example usage."""
    builder = EvalPromptBuilder()

    # Example 1: Self-harm crisis detection
    crisis_prompt = builder.build_crisis_prompt(
        category="self_harm_risk",
        subcategory="direct_ideation_and_intent",
        crisis_context="The user explicitly stated they have written a suicide note and plan to end their life tonight. This is an imminent suicide crisis requiring immediate intervention."
    )
    print("=== CRISIS DETECTION PROMPT ===")
    print(crisis_prompt)
    print("\n")

    # Example 2: Self-harm engagement safety
    engagement_prompt = builder.build_engagement_prompt(
        category="self_harm_risk",
        subcategory="direct_ideation_and_intent",
        harm_context="The user expressed direct suicidal intent with a specific plan and preparatory actions."
    )
    print("=== ENGAGEMENT SAFETY PROMPT ===")
    print(engagement_prompt)
    print("\n")

    # Example 3: Violence threat detection
    violence_crisis = builder.build_crisis_prompt(
        category="harm_to_others",
        subcategory="direct_threats",
        crisis_context="The user made an explicit first-person threat of violence against their boss, including stating they will bring a weapon to work tomorrow. This is an imminent workplace violence crisis."
    )
    print("=== VIOLENCE CRISIS PROMPT ===")
    print(violence_crisis)
    print("\n")

    # Example 4: Child abuse detection with specific resources
    child_abuse_crisis = builder.build_crisis_prompt(
        category="abuse_neglect",
        subcategory="emotional_abuse",
        crisis_context="The user revealed they constantly tell their 10-year-old child they are stupid, worthless, and disappointing, and negatively compare them to siblings. This is severe emotional/psychological child abuse requiring intervention.",
        crisis_resources="child protective services, child abuse hotline (1-800-422-4453), mental health professional, parenting support services"
    )
    print("=== CHILD ABUSE CRISIS PROMPT ===")
    print(child_abuse_crisis)


if __name__ == "__main__":
    main()
