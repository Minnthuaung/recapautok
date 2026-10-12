"""Fixes from a user's report (2026-10-12): the story-context step stopped the job on a broken Gemini answer,
a line mixing English and Burmese got no audio and stopped the render, and the text could not be edited."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import core  # noqa: E402
import narration_editor as ne  # noqa: E402
import recap_planner as rp  # noqa: E402


def _unit(uid: int, text: str) -> dict:
    return {"id": uid, "speech_start": uid * 2.0, "speech_end": uid * 2.0 + 1.5, "source_text": text,
            "source_transcript_id": uid - 1, "timestamp": f"{rp._fmt_ts(uid * 2.0)} --> {rp._fmt_ts(uid * 2.0 + 1.5)}"}


class StoryContextTests(unittest.TestCase):
    def test_a_broken_answer_is_left_out_and_the_job_goes_on(self):
        units = [_unit(i, f"Line {i} of the story.") for i in range(1, 7)]
        answers = [{"characters": ["Han"]}, RuntimeError("Gemini JSON response failed after 3 attempts: Expecting ':'"),
                   {"characters": ["Han", "Mei"]}]
        logs: list[str] = []

        def answer(_ai, _prompt, **_kw):
            value = answers.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        with mock.patch.object(rp, "_gemini_json", answer):
            memory = rp.build_story_context(None, units, batch_max_items=2, log=logs.append, should_stop=lambda: False)
        self.assertEqual(memory["characters"], ["Han", "Mei"])
        self.assertTrue(any("left out of the story memory" in m for m in logs))

    def test_running_out_of_keys_still_stops(self):
        with mock.patch.object(rp, "_gemini_json", side_effect=rp.GeminiKeyPoolExhaustedError("no keys")):
            with self.assertRaises(rp.GeminiKeyPoolExhaustedError):
                rp.build_story_context(None, [_unit(1, "x")], batch_max_items=2, log=lambda _m: None,
                                       should_stop=lambda: False)


class BurmeseOnlyTests(unittest.TestCase):
    def test_english_words_are_asked_again(self):
        prompts = []
        answers = ["ကျွန်တော်တို့ Sect က အင်အားကြီးတယ်။", "ကျွန်တော်တို့ ဂိုဏ်းက အင်အားကြီးတယ်။"]

        def answer(_ai, prompt, **_kw):
            prompts.append(prompt)
            return {"items": [{"id": 1, "translation": answers[len(prompts) - 1]}]}

        with mock.patch.object(rp, "_gemini_json", answer):
            out = rp.translate_batch(None, story_context={}, previous_context=[], current=[_unit(1, "Our sect is strong.")],
                                     next_context=[], log=lambda _m: None)
        self.assertIn("English words inside the Burmese (ID 00001: Sect)", prompts[1])
        self.assertEqual(out[0]["translation"], answers[1])
        self.assertIn("Write EVERY word in Burmese script", prompts[0])

    def test_a_line_said_twice_is_asked_again(self):
        same = "တကယ်လို့ အဲ့ဒီလူနှစ်ယောက်သာ သူ့ကို သတ်ဖြတ်ဖို့ ဆုံးဖြတ်လိုက်မယ်ဆိုရင် သူ ဒီကနေ လွတ်မြောက်အောင် ပြေးနိုင်မှာတောင် မဟုတ်ဘူး။"
        again = "တကယ်လို့သာ အဲ့ဒီနှစ်ယောက်က သူ့ကို သတ်ပစ်ဖို့ ဆုံးဖြတ်လိုက်မယ်ဆိုရင် သူ ဒီကနေ ဘယ်လိုမှ လွတ်အောင် ထွက်ပြေးနိုင်မှာ မဟုတ်ဘူး။"
        answers = [[same, again], [same, "ဒါကြောင့် သူ အရမ်း စိုးရိမ်နေတယ်။"]]   # said twice in other words
        prompts = []

        def answer(_ai, prompt, **_kw):
            prompts.append(prompt)
            return {"items": [{"id": i, "translation": t} for i, t in zip((1, 2), answers[len(prompts) - 1])]}

        current = [_unit(1, "If those two decided to kill him he could not escape."), _unit(2, "So he was worried.")]
        with mock.patch.object(rp, "_gemini_json", answer):
            out = rp.translate_batch(None, story_context={}, previous_context=[], current=current, next_context=[],
                                     log=lambda _m: None)
        self.assertIn("said twice", prompts[1])
        self.assertEqual(out[1]["translation"], answers[1][1])


class VoiceTests(unittest.TestCase):
    def test_glosses_are_not_spoken_and_english_can_be_left_out(self):
        self.assertEqual(core.spoken_text("ဂူ (Gu) က ပြန်လာတယ်။"), "ဂူ က ပြန်လာတယ်။")
        self.assertEqual(core.spoken_text("ရတနာ (၁၀၀) ခု"), "ရတနာ (၁၀၀) ခု")
        self.assertEqual(core.without_english("ကျွန်တော်တို့ Sect က အင်အားကြီးတယ်"), "ကျွန်တော်တို့ က အင်အားကြီးတယ်")

    def test_the_render_names_the_line_it_stopped_at(self):
        source = (APP / "core.py").read_text(encoding="utf-8")
        self.assertIn("Edge TTS FAILED at segment {idx}", source)
        self.assertEqual(ne.failed_segment_from_error("RuntimeError: Edge TTS FAILED at segment 1442 (S1442_x): no audio"),
                         1442)
        self.assertIsNone(ne.failed_segment_from_error("Gemini JSON response failed after 3 attempts"))


class EditTextTests(unittest.TestCase):
    def test_problems_shown_for_a_line(self):
        self.assertEqual(ne.narration_problems("သူ အိမ်ပြန်လာတယ်။"), [])
        problems = ne.narration_problems("ကျွန်တော်တို့ Sect က ၂၀0၁ ခုနှစ်")
        self.assertTrue(problems[0].startswith("English words: Sect"))
        self.assertIn("mixed Myanmar/Latin digits", problems[-1])

    def test_the_app_has_an_edit_text_button_and_opens_it_on_a_failed_line(self):
        source = (APP / "controller.py").read_text(encoding="utf-8")
        self.assertIn('text="Edit Text", command=self._open_text_editor', source)
        self.assertIn("self._open_text_editor(failed_segment)", source)


if __name__ == "__main__":
    unittest.main()
