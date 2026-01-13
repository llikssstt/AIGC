import sys
import os
from unittest.mock import MagicMock
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdxl_app.engine.prompt_utils import PromptCompiler
from sdxl_app.config import STYLE_PRESETS, DEFAULT_NEGATIVE_PROMPT, INPAINT_NEGATIVE_APPEND
from sdxl_app.engine.llm_service import PoetryInterpretation

def test_prompt_compilation():
    print("=== Testing Prompt Compilation Logic ===\n")

    # Mock LLM Service
    mock_llm = MagicMock()
    
    # Scene: 孤舟蓑笠翁 (Lone boat, bamboo hat old man)
    # Mocking the structured return from LLM
    mock_interpretation = PoetryInterpretation(
        subject_description="1 elderly man, long white beard, wearing traditional straw raincoat and bamboo conical hat",
        action_description="sitting alone on a small wooden boat, holding a fishing rod, fishing",
        environment_description="vast river, frozen water, heavy falling snow, misty mountains in background",
        composition_description="wide shot, minimalist composition, large negative space",
        mood_description="solitary, peaceful, cold, zen",
        visual_elements=["old man", "snowy river", "fishing boat", "falling snow"],
        raw_response="{...}"
    )
    mock_llm.interpret_poetry.return_value = mock_interpretation

    # Initialize Compiler
    compiler = PromptCompiler(
        style_presets=STYLE_PRESETS,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        inpaint_negative_append=INPAINT_NEGATIVE_APPEND,
        llm_service=mock_llm
    )

    # Test Case 1: Poetry Input with LLM Success
    print(f"Input: 孤舟蓑笠翁，独钓寒江雪 (Style: 水墨)")
    bundle = compiler.compile_generation(style_preset="水墨", scene_text="孤舟蓑笠翁，独钓寒江雪")
    
    print(f"\n[Generated Prompt Output]:\n{bundle.final_prompt}")
    print(f"\n[Negative Prompt]:\n{bundle.negative_prompt}")
    
    print("-" * 50)
    
    # Verify exact parts
    assert "(1 elderly man, long white beard, wearing traditional straw raincoat and bamboo conical hat:1.3)" in bundle.final_prompt
    assert "sitting alone on a small wooden boat" in bundle.final_prompt
    assert "shuimo style" in bundle.final_prompt # Base style should be there

    # Test Case 2: Poetry Input with "No Humans" (Empty Mountain)
    mock_interpretation_empty = PoetryInterpretation(
        subject_description="no humans",
        action_description="none",
        environment_description="dense deep forest, ancient trees",
        composition_description="low angle",
        mood_description="quiet",
        visual_elements=["forest", "trees"],
        raw_response="{...}"
    )
    mock_llm.interpret_poetry.return_value = mock_interpretation_empty
    
    print(f"\nInput: 空山不见人 (Style: 青绿)")
    bundle_empty = compiler.compile_generation(style_preset="青绿", scene_text="空山不见人")
    
    print(f"\n[Generated Prompt Output]:\n{bundle_empty.final_prompt}")
    print(f"\n[Negative Prompt]:\n{bundle_empty.negative_prompt}")
    
    # Verify Negative Prompt Logic
    assert "humans, person" in bundle_empty.negative_prompt
    assert "(no humans)" not in bundle_empty.final_prompt # Should not generate prompt with subject if empty
    
    print("\n=== Verification Passed ===")

if __name__ == "__main__":
    test_prompt_compilation()
