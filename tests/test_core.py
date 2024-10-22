from typing import Text, List, Tuple
from unittest import TestCase
from src.core import Core, CoreInstance
from src.core.scorers import UNLIConfidenceBoostScorer
from src.core.entailers import SoftEntailer, Entailer


class TestCore(TestCase):
    def setUp(self):
        """
        """
        self.test_cases = [
            {
                "topic": "Kalki Koechlin",
                "input": "Kalki Koechlin is an Indian actress and writer known for her work in Hindi films. Born on January 10, 1984, in Pondicherry, India, Kalki spent her early years in Auroville before moving to London to study drama and theatre. She made her acting debut in the critically acclaimed film \"Dev.D\" in 2009, for which she received the Filmfare Award for Best Supporting Actress.\n\nKalki has since appeared in a variety of films, showcasing her versatility as an actress. Some of her notable performances include \"Zindagi Na Milegi Dobara,\" \"Margarita with a Straw,\" and \"Gully Boy.\" She has garnered praise for her unconventional choice of roles and her ability to portray complex characters with depth and authenticity.\n\nIn addition to her acting career, Kalki is also a talented writer and has written for various publications on topics such as feminism, mental health, and social issues. She is known for her outspoken views on gender equality and has been a vocal advocate for women's rights in India. Kalki Koechlin continues to be a prominent figure in the Indian film industry, known for her bold and fearless approach to her craft.",
                "decomposed": [
                    "She received an award.",
                    "She received the Filmfare Award.",
                    "She received the Filmfare Award for Best Supporting Actress.",
                    "She received the Filmfare Award for Best Supporting Actress for D\".",
                    "Kalki has appeared in a variety of films.",
                    "Kalki has appeared in a variety of films since.",
                    "Kalki has showcased her versatility as an actress.",
                    "Kalki has showcased her versatility as an actress in films.",
                    "Some of her performances include \"Zindagi Na Milegi Dobara.\"",
                    "Kalki has an acting career.",
                    "Kalki is a talented writer.",
                    "Kalki has written for various publications.",
                    "Kalki has written for various publications on topics such as feminism.",
                    "Kalki has written for various publications on topics such as mental health.",
                    "Kalki has written for various publications on topics such as social issues.",
                    "She is known for her outspoken views.",
                    "She is known for her outspoken views on gender equality.",
                    "She has been a vocal advocate.",
                    "She has been a vocal advocate for women's rights.",
                    "She has been a vocal advocate for women's rights in India.",
                    "Kalki Koechlin is a prominent figure."
                    "Kalki Koechlin is a prominent figure in the Indian film industry.",
                    "Kalki Koechlin is known for her bold approach.",
                    "Kalki Koechlin is known for her fearless approach.",
                    "Kalki Koechlin is known for her bold and fearless approach.",
                    "Kalki Koechlin is known for her bold and fearless approach to her craft."
                ],
            }
        ]
        
    def test_core_with_simplest_decomposer(self):
        """Test whether the simplest decomposer (Text -> List[Text])
        works with the Core decorator.
        """
        
        for test_case in self.test_cases:
            
            @Core(cache_dir=".cache/", silent=False)
            def decomposer(text) -> List[str]:
                assert text == test_case["input"], "The input text is not the same as the expected text."
                return test_case["decomposed"]
            
            self.assertLess(len(decomposer(test_case["input"])), len(test_case["decomposed"]))
            
    def test_importance_weighted_core(self):
        """Test whether it's possible to run
        """
        
        for test_case in self.test_cases:
            
            @Core(
                result_parser=lambda result: [CoreInstance(text=r, sent=None, topic=result[1]) for r in result[0]],
                result_merger=lambda selected, inputs, result: ([inputs[s].text for s in selected], result[1]),
                claim_level_checkworthy_scorer=UNLIConfidenceBoostScorer(
                    bleached_templates=[
                        "{topic} is a person.",
                        "{topic} breathes."
                        "{topic} exists."
                        "{topic} is a name."
                        "{topic} is unique."
                        "{topic} is famous."
                        "{topic} has some abilities."
                        "somebody knows {topic}."
                        "{topic} is a star."
                    ],
                    entailer=SoftEntailer(
                        model_name="Zhengping/roberta-large-unli",
                        device="cpu",
                        internal_batch_size=256,
                        max_length=256,
                        cache_dir=".cache/"
                    ),
                    cap_entailer=Entailer(
                        model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                        device="cpu",
                        internal_batch_size=256,
                        max_length=256,
                        cache_dir=".cache/"
                    )
                ),
                cache_dir=".cache/",
                silent=False
            )
            def decomposer(text) -> Tuple[List[str], Text]:
                assert text == test_case["input"], "The input text is not the same as the expected text."
                return test_case["decomposed"], test_case["topic"]
            
            self.assertLess(len(decomposer(test_case["input"])), len(test_case["decomposed"]))
