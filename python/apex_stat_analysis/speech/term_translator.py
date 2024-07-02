from apex_stat_analysis.speech.terms import ApexTermBase, ApexTerms, strip_punctuation


class ApexTranslator:
    def __init__(self):
        self._apex_term_word_lim = max(term.get_max_variation_len()
                                       for term in ApexTerms.ALL_TERMS)
        self._apex_terms_by_sizes = {
            n: tuple(filter(lambda term: term.has_variation_len(n), ApexTerms.ALL_TERMS))
            for n in range(self._apex_term_word_lim, 0, -1)
        }

    def translate_terms(self, text: str) -> tuple[ApexTermBase]:
        words = strip_punctuation(text).split(' ')
        reconstructed: list[ApexTermBase] = []
        idx = 0
        while idx < len(words):
            apex_term, words_inc = self._translate_term(words[idx:idx + self._apex_term_word_lim])
            if apex_term is not None:
                reconstructed.append(apex_term)
            idx += words_inc
        return tuple(reconstructed)

    def _translate_term(self, words: list[str]) -> tuple[ApexTermBase | None, int]:
        for num_to_test in range(self._apex_term_word_lim, 0, -1):
            apex_term: ApexTermBase | None = next(
                filter(lambda _at: _at.has_variation(words[:num_to_test]),
                       self._get_apex_terms_of_size(num_to_test)),
                None)
            if apex_term is not None:
                return apex_term, num_to_test

        return None, 1

    def _get_apex_terms_of_size(self, n) -> tuple[ApexTermBase]:
        return self._apex_terms_by_sizes.get(n, tuple())

    @staticmethod
    def get_apex_terms(self):
        return self.ALL_TERMS
