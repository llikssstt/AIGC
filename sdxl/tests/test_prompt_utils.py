from sdxl_app.engine.prompt_utils import PromptCompiler


def test_prompt_compiler_generation():
    compiler = PromptCompiler(
        style_presets={"A": "STYLE_A"},
        negative_prompt="NEG",
        inpaint_negative_append=", INP",
    )

    bundle = compiler.compile_generation("A", "a cat")
    assert bundle.global_prompt == "STYLE_A"
    assert bundle.final_prompt == "STYLE_A, a cat"
    assert bundle.negative_prompt == "NEG"

    card = compiler.generation_card(
        style_preset="A",
        scene_text="a cat",
        bundle=bundle,
        seed=123,
        steps=30,
        cfg=7.5,
        width=1024,
        height=1024,
    )
    assert card["edit_type"] == "generate"
    assert card["style_preset"] == "A"
    assert card["seed"] == 123


def test_prompt_compiler_edit_empty_global_prompt():
    compiler = PromptCompiler(
        style_presets={"A": "STYLE_A"},
        negative_prompt="NEG",
        inpaint_negative_append=", INP",
    )

    bundle = compiler.compile_edit("", "add a red sun")
    assert bundle.final_prompt == "add a red sun, in the masked area only"
    assert bundle.negative_prompt == "NEG, INP"

    bundle2 = compiler.compile_edit("", "")
    assert bundle2.final_prompt == "in the masked area only"


def test_prompt_compiler_import_card():
    compiler = PromptCompiler(
        style_presets={"A": "STYLE_A"},
        negative_prompt="NEG",
        inpaint_negative_append=", INP",
    )
    card = compiler.import_card()
    assert card["edit_type"] == "import"
    assert card["final_prompt"] == ""
