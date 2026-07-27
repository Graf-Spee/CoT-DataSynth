"""English IFEval instruction registry.

This mirrors the English portion of OpenCompass/M-IFEval's registry. IFEval's
standard English benchmark uses `en:` instruction ids.
"""

from .instructions import en_instructions

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

EN_INSTRUCTION_DICT = {
    _KEYWORD + "existence": en_instructions.KeywordChecker,
    _KEYWORD + "frequency": en_instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": en_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": en_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": en_instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": en_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": en_instructions.ParagraphChecker,
    _LENGTH + "number_words": en_instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": en_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": en_instructions.PlaceholderChecker,
    _CONTENT + "postscript": en_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": en_instructions.BulletListChecker,
    _FORMAT + "constrained_response": en_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": en_instructions.HighlightSectionChecker,
    _FORMAT + "multiple_sections": en_instructions.SectionChecker,
    _FORMAT + "json_format": en_instructions.JsonFormat,
    _FORMAT + "title": en_instructions.TitleChecker,
    _COMBINATION + "two_responses": en_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": en_instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": en_instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": en_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": en_instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": en_instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": en_instructions.CommaChecker,
    _STARTEND + "quotation": en_instructions.QuotationChecker,
}

INSTRUCTION_DICT = {"en:" + key: value for key, value in EN_INSTRUCTION_DICT.items()}

