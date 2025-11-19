from typing import Optional, List, Dict, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

DECONJUGATION_INPUT = "こらえられなかった"

TEST_WORDS = [
    DECONJUGATION_INPUT,
    "送らなかった",
    "くり返さない",
    "増えてあげない",
    "繰り返してみる",
    "食べている",
    "行ってしまった",
]


@dataclass
class ConjugationNode:
    """Represents a node within a conjugation tree."""
    terminates: bool = True
    ends: bool = False
    node_type: str = "transforms"
    conj_type: str = ""
    transforms: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    children: List['ConjugationNode'] = field(default_factory=list)
    alternate_forms: List[str] = field(default_factory=list)
    label: Optional[str] = None
    verb_type: Optional[Union[str, List[str]]] = None


class JapaneseDeconjugator:
    """
    Deconjugates Japanese verbs and adjectives using exhaustive logic trees.
    """

    VERB_TYPES: Tuple[str, ...] = (
        "ichidan",
        "godan_u",
        "godan_tsu",
        "godan_ru",
        "godan_ku",
        "godan_gu",
        "godan_nu",
        "godan_bu",
        "godan_mu",
        "godan_su",
        "suru",
        "kuru",
        "kuru-alt",
        "iku",
        "iku-alt",
    )
    VERB_DICTIONARY_ENDINGS: Tuple[str, ...] = ("る", "う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む")
    SPECIAL_DICTIONARY_VERBS: Tuple[str, ...] = ("する", "くる", "来る")

    def __init__(self, dictionary_path: Optional[str] = None, require_dictionary: bool = False):
        """
        Initialize the deconjugator.

        Args:
            dictionary_path: Path to the SQLite dictionary database. If None, no dictionary validation is performed.
            require_dictionary: If True, requires dictionary validation for all results. If False, returns results without validation.
        """
        if dictionary_path:
            resolved_dictionary = Path(dictionary_path)
            self.dictionary_path: Optional[Path] = resolved_dictionary if resolved_dictionary.exists() else None
        else:
            self.dictionary_path = None

        self._dictionary_conn: Optional[sqlite3.Connection] = None
        self._dictionary_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.require_dictionary = require_dictionary

        self.verb_tree = self._build_verb_tree()
        self.adjective_tree = self._build_adjective_tree()
        self.nai_tree = self._build_nai_tree()

    def __del__(self):
        if getattr(self, "_dictionary_conn", None):
            try:
                self._dictionary_conn.close()
            except Exception:
                pass
            finally:
                self._dictionary_conn = None

    def _build_verb_tree(self) -> ConjugationNode:
        """Build the verb conjugation tree based on the provided diagram."""

        # Root node (dictionary form)
        dictionary_root_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="root",
            conj_type="",
            transforms={}
        )

        # た (past tense)
        past_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('た', 'る'),

                'godan_su': ('した', 'す'),

                'godan_ku': ('いた', 'く'),
                'godan_gu': ('いだ', 'ぐ'),

                'godan_mu': ('んだ', 'む'),
                'godan_bu': ('んだ', 'ぶ'),
                'godan_nu': ('んだ', 'ぬ'),

                'godan_u': ('った', 'う'),
                'godan_tsu': ('った', 'つ'),
                'godan_ru': ('った', 'る'),

                'suru': ('した', 'する'),
                'iku': ('いった', 'いく'),
                'iku-alt': ('行った', '行く'),
                'kuru': ('きた', 'くる'),
                'kuru-alt': ('来た', '来る'),
            }
        )

        tara_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('ら', '')}
        )

        past_negative_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('なかった', 'る'),

                'godan_su': ('しなかった', 'す'),

                'godan_ku': ('かなかった', 'く'),
                'godan_gu': ('がなかった', 'ぐ'),

                'godan_mu': ('まなかった', 'む'),
                'godan_bu': ('ばなかった', 'ぶ'),
                'godan_nu': ('ななかった', 'ぬ'),

                'godan_u': ('わなかった', 'う'),
                'godan_tsu': ('たなかった', 'つ'),
                'godan_ru': ('らなかった', 'る'),

                'suru': ('しなかった', 'する'),
                'iku': ('いかなかった', 'いく'),
                'iku-alt': ('行かなかった', '行く'),
                'kuru': ('こなかった', 'くる'),
                'kuru-alt': ('来なかった', '来る'),
            }
        )

        nai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="nai",
            transforms={
                 'ichidan': ('ない', 'る'),

                'godan_su': ('さない', 'す'),

                'godan_ku': ('かない', 'く'),
                'godan_gu': ('がない', 'ぐ'),

                'godan_mu': ('まない', 'む'),
                'godan_bu': ('ばない', 'ぶ'),
                'godan_nu': ('なない', 'ぬ'),

                'godan_u': ('わない', 'う'),
                'godan_tsu': ('たない', 'つ'),
                'godan_ru': ('らない', 'る'),

                'suru': ('しない', 'する'),
                'kuru': ('こない', 'くる'),
                'kuru-alt': ('来ない', '来る'),
            }
        )

        nu_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={
                 'ichidan': ('ぬ', 'る'),

                'godan_su': ('さぬ', 'す'),

                'godan_ku': ('かぬ', 'く'),
                'godan_gu': ('がぬ', 'ぐ'),

                'godan_mu': ('まぬ', 'む'),
                'godan_bu': ('ばぬ', 'ぶ'),
                'godan_nu': ('なぬ', 'ぬ'),

                'godan_u': ('わぬ', 'う'),
                'godan_tsu': ('たぬ', 'つ'),
                'godan_ru': ('らぬ', 'る'),

                'suru': ('せぬ', 'する'),
                'kuru': ('こぬ', 'くる'),
                'kuru-alt': ('来ぬ', '来る'),
            }
        )

        zu_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={
                 'ichidan': ('ず', 'る'),

                'godan_su': ('さず', 'す'),

                'godan_ku': ('かず', 'く'),
                'godan_gu': ('がず', 'ぐ'),

                'godan_mu': ('まず', 'む'),
                'godan_bu': ('ばず', 'ぶ'),
                'godan_nu': ('なず', 'ぬ'),

                'godan_u': ('わず', 'う'),
                'godan_tsu': ('たず', 'つ'),
                'godan_ru': ('らず', 'る'),

                'suru': ('せず', 'する'),
                'kuru': ('こず', 'くる'),
                'kuru-alt': ('来ず', '来る'),
            }
        )

        tai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="i",
            transforms={
                 'ichidan': ('たい', 'る'),

                'godan_su': ('したい', 'す'),

                'godan_ku': ('きたい', 'く'),
                'godan_gu': ('ぎたい', 'ぐ'),

                'godan_mu': ('みたい', 'む'),
                'godan_bu': ('びたい', 'ぶ'),
                'godan_nu': ('にたい', 'ぬ'),

                'godan_u': ('いたい', 'う'),
                'godan_tsu': ('ちたい', 'つ'),
                'godan_ru': ('りたい', 'る'),

                'suru': ('したい', 'する'),
                'kuru': ('きたい', 'くる'),
                'kuru-alt': ('来たい', '来る'),
            }
        )

        masu_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={
                 'ichidan': ('ます', 'る'),

                'godan_su': ('します', 'す'),

                'godan_ku': ('きます', 'く'),
                'godan_gu': ('ぎます', 'ぐ'),

                'godan_mu': ('みます', 'む'),
                'godan_bu': ('びます', 'ぶ'),
                'godan_nu': ('にます', 'ぬ'),

                'godan_u': ('います', 'う'),
                'godan_tsu': ('ちます', 'つ'),
                'godan_ru': ('ります', 'る'),

                'suru': ('します', 'する'),
                'iku': ('いきます', 'いく'),
                'iku-alt': ('行きます', '行く'),
                'kuru': ('きます', 'くる'),
                'kuru-alt': ('来ます', '来る'),
            }
        )

        mashou_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('ましょう', 'る'),

                'godan_su': ('しましょう', 'す'),

                'godan_ku': ('きましょう', 'く'),
                'godan_gu': ('ぎましょう', 'ぐ'),

                'godan_mu': ('みましょう', 'む'),
                'godan_bu': ('びましょう', 'ぶ'),
                'godan_nu': ('にましょう', 'ぬ'),

                'godan_u': ('いましょう', 'う'),
                'godan_tsu': ('ちましょう', 'つ'),
                'godan_ru': ('りましょう', 'る'),

                'suru': ('しましょう', 'する'),
                'kuru': ('きましょう', 'くる'),
                'kuru-alt': ('来ましょう', '来る'),
            }
        )

        mashi_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('まし', 'る'),

                'godan_su': ('しまし', 'す'),

                'godan_ku': ('きまし', 'く'),
                'godan_gu': ('ぎまし', 'ぐ'),

                'godan_mu': ('みまし', 'む'),
                'godan_bu': ('びまし', 'ぶ'),
                'godan_nu': ('にまし', 'ぬ'),

                'godan_u': ('いまし', 'う'),
                'godan_tsu': ('ちまし', 'つ'),
                'godan_ru': ('りまし', 'る'),

                'suru': ('しまし', 'する'),
                'iku': ('いきまし', 'いく'),
                'iku-alt': ('行きまし', '行く'),
                'kuru': ('きまし', 'くる'),
                'kuru-alt': ('来まし', '来る'),
            }
        )

        mashi_ta_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="",
            transforms={'all': ('た', '')}
        )

        mashi_te_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('て', '')}
        )

        masen_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('ません', 'る'),

                'godan_su': ('しません', 'す'),

                'godan_ku': ('きません', 'く'),
                'godan_gu': ('ぎません', 'ぐ'),

                'godan_mu': ('みません', 'む'),
                'godan_bu': ('びません', 'ぶ'),
                'godan_nu': ('にません', 'ぬ'),

                'godan_u': ('いません', 'う'),
                'godan_tsu': ('ちません', 'つ'),
                'godan_ru': ('りません', 'る'),

                'suru': ('しません', 'する'),
                'kuru': ('きません', 'くる'),
                'kuru-alt': ('来ません', '来る'),
            }
        )

        masen_deshita_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('でした', '')}
        )

        # て (te-form)
        te_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('て', 'る'),

                'godan_su': ('して', 'す'),

                'godan_ku': ('いて', 'く'),
                'godan_gu': ('いで', 'ぐ'),

                'godan_mu': ('んで', 'む'),
                'godan_bu': ('んで', 'ぶ'),
                'godan_nu': ('んで', 'ぬ'),

                'godan_u': ('って', 'う'),
                'godan_tsu': ('って', 'つ'),
                'godan_ru': ('って', 'る'),

                'suru': ('して', 'する'),
                'iku': ('いって', 'いく'),
                'iku-alt': ('行って', '行く'),
                'kuru': ('きて', 'くる'),
                'kuru-alt': ('来て', '来る'),
            }
        )

        # Auxiliary verb nodes - these match the BASE auxiliary and then conjugate
        # They should match AFTER the て-form node processes
        te_ageru_node = ConjugationNode(
            terminates=False,  # Don't terminate here, need to continue to base verb
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てあげる",
            transforms={'all': ('あげる', '')}
        )

        te_kureru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てくれる",
            transforms={'all': ('くれる', '')}
        )

        te_morau_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てもらう",
            transforms={'all': ('もらう', '')}
        )

        te_oku_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="ておく",
            transforms={'all': ('おく', '')}
        )

        te_shimau_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てしまう",
            transforms={'all': ('しまう', '')}
        )

        te_iku_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="ていく",
            transforms={'all': ('いく', '')}
        )

        te_kuru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てくる",
            transforms={'all': ('くる', '')}
        )

        te_miru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てみる",
            transforms={'all': ('みる', '')}
        )

        te_iru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="ている",
            transforms={'all': ('いる', '')}
        )

        te_ru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てる",
            transforms={'all': ('る', '')}
        )

        te_aru_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てある",
            transforms={'all': ('ある', '')}
        )

        te_nai_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="auxiliary",
            conj_type="",
            label="てない",
            transforms={'all': ('ない', '')}
        )

        chimau_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={
                'ichidan': ('ちまう', 'る'),

                'godan_su': ('しちまう', 'す'),

                'godan_ku': ('いちまう', 'く'),
                'godan_gu': ('いじまう', 'ぐ'),

                'godan_mu': ('んじまう', 'む'),
                'godan_bu': ('んじまう', 'ぶ'),
                'godan_nu': ('んじまう', 'ぬ'),

                'godan_u': ('っちまう', 'う'),
                'godan_tsu': ('っちまう', 'つ'),
                'godan_ru': ('っちまう', 'る'),

                'suru': ('しちまう', 'する'),
                'iku': ('いっちまう', 'いく'),
                'iku-alt': ('行っちまう', '行く'),
                'kuru': ('きちまう', 'くる'),
                'kuru-alt': ('来ちまう', '来る'),
            }
        )

        chau_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={
                'ichidan': ('ちゃう', 'る'),

                'godan_su': ('しちゃう', 'す'),

                'godan_ku': ('いちゃう', 'く'),
                'godan_gu': ('いじゃう', 'ぐ'),

                'godan_mu': ('んじゃう', 'む'),
                'godan_bu': ('んじゃう', 'ぶ'),
                'godan_nu': ('んじゃう', 'ぬ'),

                'godan_u': ('っちゃう', 'う'),
                'godan_tsu': ('っちゃう', 'つ'),
                'godan_ru': ('っちゃう', 'る'),

                'suru': ('しちゃう', 'する'),
                'iku': ('いっちゃう', 'いく'),
                'iku-alt': ('行っちゃう', '行く'),
                'kuru': ('きちゃう', 'くる'),
                'kuru-alt': ('来ちゃう', '来る'),
            }
        )

        to_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="",
            transforms={'all': ('と', '')}
        )

        ikenai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="nai",
            alternate_forms=["行けない"],
            transforms={'all': ('いけない', '')}
        )

        reba_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('れば', 'る'),

                'godan_su': ('せば', 'す'),

                'godan_ku': ('けば', 'く'),
                'godan_gu': ('げば', 'ぐ'),

                'godan_mu': ('めば', 'む'),
                'godan_bu': ('べば', 'ぶ'),
                'godan_nu': ('ねば', 'ぬ'),

                'godan_u': ('えば', 'う'),
                'godan_tsu': ('てば', 'つ'),
                'godan_ru': ('れば', 'る'),

                'suru': ('すれば', 'する'),
            }
        )

        nakereba_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('なければ', 'る'),

                'godan_su': ('さなければ', 'す'),

                'godan_ku': ('かなければ', 'く'),
                'godan_gu': ('がなければ', 'ぐ'),

                'godan_mu': ('まなければ', 'む'),
                'godan_bu': ('ばなければ', 'ぶ'),
                'godan_nu': ('ななければ', 'ぬ'),

                'godan_u': ('わなければ', 'う'),
                'godan_tsu': ('たなければ', 'つ'),
                'godan_ru': ('らなければ', 'る'),

                'suru': ('しなければ', 'する'),
                'iku': ('いかなければ', 'いく'),
                'iku-alt': ('行かなければ', '行く'),
                'kuru': ('こなければ', 'くる'),
                'kuru-alt': ('来なければ', '来る'),
            }
        )

        nakereba_naranai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="nai",
            alternate_forms=["成らない"],
            transforms={'all': ('ならない', '')}
        )

        nakereba_ikenai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="nai",
            alternate_forms=["行けない"],
            transforms={'all': ('いけない', '')}
        )

        stem_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('', 'る'),

                'godan_su': ('し', 'す'),

                'godan_ku': ('き', 'く'),
                'godan_gu': ('ぎ', 'ぐ'),

                'godan_mu': ('み', 'む'),
                'godan_bu': ('び', 'ぶ'),
                'godan_nu': ('に', 'ぬ'),

                'godan_u': ('い', 'う'),
                'godan_tsu': ('ち', 'つ'),
                'godan_ru': ('り', 'る'),

                'suru': ('し', 'する'),
                'iku': ('いき', 'いく'),
                'iku-alt': ('行き', '行く'),
                'kuru': ('き', 'くる'),
                'kuru-alt': ('来', '来る'),
            }
        )

        stem_tai_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="i",
            transforms={'all': ('たい', '')}
        )

        hajimeru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["始める"],
            transforms={'all': ('はじめる', '')}
        )

        owaru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["終わる"],
            transforms={'all': ('おわる', '')}
        )

        kaeru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["返る", "帰る"],
            transforms={'all': ('かえる', '')}
        )

        kiru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["切る"],
            transforms={'all': ('きる', '')}
        )

        sugiru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["過ぎる"],
            transforms={'all': ('すぎる', '')}
        )

        makuru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["捲る"],
            transforms={'all': ('まくる', '')}
        )

        sou_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('そう', '')}
        )

        gachi_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('がち', '')}
        )

        ppanashi_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('っぱなし', '')}
        )

        toku_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('とく', '')}
        )

        rareru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={
                'ichidan': ('られる', 'る'),

                'godan_su': ('される', 'す'),

                'godan_ku': ('かれる', 'く'),
                'godan_gu': ('がれる', 'ぐ'),

                'godan_mu': ('まれる', 'む'),
                'godan_bu': ('ばれる', 'ぶ'),
                'godan_nu': ('なれる', 'ぬ'),

                'godan_u': ('われる', 'う'),
                'godan_tsu': ('たれる', 'つ'),
                'godan_ru': ('られる', 'る'),

                'suru': ('される', 'する'),
                'kuru': ('こられる', 'くる'),
                'kuru-alt': ('来られる', '来る'),
            }
        )

        saseru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={
                'ichidan': ('させる', 'る'),

                'godan_su': ('させる', 'す'),

                'godan_ku': ('かせる', 'く'),
                'godan_gu': ('がせる', 'ぐ'),

                'godan_mu': ('ませる', 'む'),
                'godan_bu': ('ばせる', 'ぶ'),
                'godan_nu': ('なせる', 'ぬ'),

                'godan_u': ('わせる', 'う'),
                'godan_tsu': ('たせる', 'つ'),
                'godan_ru': ('らせる', 'る'),

                'suru': ('させる', 'する'),
                'kuru': ('こさせる', 'くる'),
                'kuru-alt': ('来させる', '来る'),
            }
        )

        saseru_saserareru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('られる', 'る')}
        )

        sasu_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={
                'ichidan': ('さす', 'る'),

                'godan_su': ('さす', 'す'),

                'godan_ku': ('かす', 'く'),
                'godan_gu': ('がす', 'ぐ'),

                'godan_mu': ('ます', 'む'),
                'godan_bu': ('ばす', 'ぶ'),
                'godan_nu': ('なす', 'ぬ'),

                'godan_u': ('わす', 'う'),
                'godan_tsu': ('たす', 'つ'),
                'godan_ru': ('らす', 'る'),

                'suru': ('さす', 'する'),
                'kuru': ('こさす', 'くる'),
                'kuru-alt': ('来さす', '来る'),
            }
        )

        sasu_sareru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('される', 'す')}
        )

        rou_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={
                'ichidan': ('よう', 'る'),

                'godan_su': ('そう', 'す'),

                'godan_ku': ('こう', 'く'),
                'godan_gu': ('ごう', 'ぐ'),

                'godan_mu': ('もう', 'む'),
                'godan_bu': ('ぼう', 'ぶ'),

                'godan_nu': ('のう', 'ぬ'),

                'godan_u': ('おう', 'う'),
                'godan_tsu': ('とう', 'つ'),
                'godan_ru': ('ろう', 'る'),

                'suru': ('しよう', 'する'),
                'kuru': ('こよう', 'くる'),
                'kuru-alt': ('来よう', '来る'),
            }
        )

        rou_tosuru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('とする', '')}
        )

        rou_toomou_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["と思う"],
            transforms={'all': ('とおもう', '')}
        )

        o_prefix_stem_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="",
            transforms={'all': ('お', '')}
        )

        honorific_ninaru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            alternate_forms=["に成る"],
            transforms={'all': ('になる', '')}
        )

        honorific_suru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('する', '')}
        )

        past_node.children = [tara_node]
        past_negative_node.children = [tara_node]
        mashi_node.children = [mashi_ta_node, mashi_te_node]
        mashi_ta_node.children = [tara_node]
        masen_node.children = [masen_deshita_node]
        te_node.children = []
        to_node.children = [ikenai_node]
        nakereba_node.children = [nakereba_naranai_node, nakereba_ikenai_node]
        stem_node.children = [stem_tai_node, hajimeru_node, owaru_node, kaeru_node, kiru_node, sugiru_node, makuru_node, sou_node, gachi_node, ppanashi_node, toku_node]
        saseru_node.children = [saseru_saserareru_node]
        sasu_node.children = [sasu_sareru_node]
        rou_node.children = [rou_tosuru_node, rou_toomou_node]
        o_prefix_stem_node.children = [honorific_ninaru_node, honorific_suru_node]

        # Put auxiliary verbs at top level AND as children of te_node
        dictionary_root_node.children = [
            past_node, past_negative_node, nai_node, nu_node, zu_node, tai_node,
            masu_node, mashou_node, mashi_node, masen_node, te_node,
            # Auxiliary verbs at top level to catch conjugated forms
            te_ageru_node, te_kureru_node, te_morau_node, te_oku_node, te_shimau_node,
            te_iku_node, te_kuru_node, te_miru_node, te_iru_node, te_ru_node, te_aru_node, te_nai_node,
            chimau_node, chau_node, to_node, reba_node, nakereba_node, stem_node,
            rareru_node, saseru_node, sasu_node, rou_node, o_prefix_stem_node
        ]

        return dictionary_root_node

    def _build_adjective_tree(self) -> ConjugationNode:
        """Build the adjective conjugation tree."""

        dictionary_root_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="root",
            conj_type="",
            transforms={}
        )

        katta_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('かった', 'い')}
        )

        kunai_node = ConjugationNode(
            terminates=False,
            ends=False,
            node_type="transforms",
            conj_type="nai",
            transforms={'all': ('くない', 'い')}
        )

        kunakatta_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('くなかった', 'い')}
        )

        tara_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('ら', '')}
        )

        ku_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('く', 'い')}
        )

        ku_naru = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('なる', '')}
        )

        kute_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('て', '')}
        )

        kereba_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('ければ', 'い')}
        )

        garu_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="verb",
            transforms={'all': ('がる', 'い')}
        )

        sou_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('そう', 'い')}
        )

        sa_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('さ', 'い')}
        )

        me_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('め', 'い')}
        )

        karou_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('かろう', 'い')}
        )

        ge_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('げ', 'い')}
        )

        sugi_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('すぎ', 'い')}
        )

        sugi_ru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('る', '')}
        )

        katta_node.children = [tara_node]
        kunakatta_node.children = [tara_node]
        ku_node.children = [ku_naru, kute_node]
        sugi_node.children = [sugi_ru_node]
        dictionary_root_node.children = [katta_node, kunakatta_node, kunai_node, ku_node, kereba_node, garu_node, sou_node, sa_node, me_node, karou_node, ge_node, sugi_node]

        return dictionary_root_node

    def _build_nai_tree(self) -> ConjugationNode:
        """Build the nai-form conjugation tree."""

        dictionary_root_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="root",
            conj_type="",
            transforms={}
        )

        desu_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('です', '')}
        )

        katta_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なかった', 'ない')}
        )
        tara_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('ら', '')}
        )

        ku_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なく', 'ない')}
        )

        ku_naru = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('なる', '')}
        )

        kute_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="appends",
            conj_type="",
            transforms={'all': ('て', '')}
        )

        kereba_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なければ', 'ない')}
        )

        sasou_node = ConjugationNode(
            terminates=True,
            ends=True,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なさそう', 'ない')}
        )

        sugi_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なすぎ', 'ない')}
        )

        sasugi_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="transforms",
            conj_type="",
            transforms={'all': ('なさすぎ', 'ない')}
        )

        ru_node = ConjugationNode(
            terminates=True,
            ends=False,
            node_type="appends",
            conj_type="verb",
            transforms={'all': ('る', '')}
        )

        katta_node.children = [tara_node, desu_node]
        ku_node.children = [ku_naru, kute_node]
        sugi_node.children = [ru_node]
        sasugi_node.children = [ru_node]
        dictionary_root_node.children = [desu_node, katta_node, ku_node, kereba_node, sasou_node, sasugi_node, sugi_node]

        return dictionary_root_node

    def _normalize_tree_name(self, tree_name: str) -> str:
        mapping = {
            "verb": "verb",
            "verbs": "verb",
            "adjective": "i",
            "adjectives": "i",
            "i": "i",
            "i_adjective": "i",
            "nai": "nai",
        }
        key = tree_name.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported tree type: {tree_name}")
        return mapping[key]

    def _get_tree(self, tree_name: str) -> ConjugationNode:
        normalized = self._normalize_tree_name(tree_name)
        if normalized == "verb":
            return self.verb_tree
        if normalized == "i":
            return self.adjective_tree
        if normalized == "nai":
            return self.nai_tree
        raise ValueError(f"Unsupported tree type: {tree_name}")

    def _candidate_types(self, tree_name: str) -> Tuple[str, ...]:
        normalized = self._normalize_tree_name(tree_name)
        if normalized == "verb":
            return self.VERB_TYPES
        return (normalized,)

    def _looks_like_dictionary(self, word: str, tree_name: str) -> bool:
        normalized = self._normalize_tree_name(tree_name)
        if normalized == "verb":
            return word.endswith(self.VERB_DICTIONARY_ENDINGS) or word in self.SPECIAL_DICTIONARY_VERBS
        if normalized == "i":
            return word.endswith("い")
        if normalized == "nai":
            return word.endswith("ない")
        return False

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        if not self.dictionary_path:
            return None
        if self._dictionary_conn is None:
            try:
                self._dictionary_conn = sqlite3.connect(str(self.dictionary_path))
                self._dictionary_conn.row_factory = sqlite3.Row
            except sqlite3.Error:
                self._dictionary_conn = None
        return self._dictionary_conn

    def _lookup_dictionary(self, term: str) -> Optional[Dict[str, Any]]:
        """Look up a term in the dictionary, checking both main entries and alternate forms."""
        if term in self._dictionary_cache:
            return self._dictionary_cache[term]

        connection = self._get_db_connection()
        if connection is None:
            self._dictionary_cache[term] = None
            return None

        cursor = connection.cursor()
        result: Optional[Dict[str, Any]] = None
        try:
            # First check main entries
            cursor.execute("SELECT id, term FROM entries WHERE term = ?", (term,))
            row = cursor.fetchone()
            if row:
                result = {
                    "entry_id": row["id"],
                    "term": row["term"],
                    "match_type": "term",
                }
            else:
                # Check alternate forms
                cursor.execute(
                    """
                    SELECT af.entry_id, af.alt_form, e.term
                    FROM alternate_forms af
                    JOIN entries e ON e.id = af.entry_id
                    WHERE af.alt_form = ?
                    """,
                    (term,)
                )
                row = cursor.fetchone()
                if row:
                    result = {
                        "entry_id": row["entry_id"],
                        "term": row["term"],
                        "match_type": "alternate",
                        "alt_form": row["alt_form"],
                    }
        except sqlite3.Error:
            result = None
        finally:
            cursor.close()

        self._dictionary_cache[term] = result
        return result

    def _get_all_forms(self, term: str) -> Set[str]:
        """Get all forms of a term (main term + all alternate forms)."""
        forms = {term}

        dict_entry = self._lookup_dictionary(term)
        if not dict_entry:
            return forms

        connection = self._get_db_connection()
        if not connection:
            return forms

        cursor = connection.cursor()
        try:
            # Get the main term
            main_term = dict_entry.get("term", term)
            forms.add(main_term)

            # Get all alternate forms for this entry
            cursor.execute(
                "SELECT alt_form FROM alternate_forms WHERE entry_id = ?",
                (dict_entry["entry_id"],)
            )
            for row in cursor.fetchall():
                forms.add(row["alt_form"])
        except sqlite3.Error:
            pass
        finally:
            cursor.close()

        return forms

    def _is_auxiliary_path(self, path: List[str]) -> bool:
        """Check if the path contains auxiliary verb markers."""
        auxiliary_markers = {"appends", "transforms"}
        auxiliary_words = {"あげる", "くれる", "もらう", "おく", "しまう", "いく", "くる", "みる", "いる", "ある"}

        for step in path:
            if any(aux in step for aux in auxiliary_words):
                return True
        return False

    def _score_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and filter results based on dictionary validation."""
        scored: List[Dict[str, Any]] = []

        for record in results:
            path = record.get("path") or []
            depth = len(path)
            score = 0.0

            # Check if base form exists in dictionary (if dictionary is available)
            dictionary_info = self._lookup_dictionary(record["base"])

            # Only filter based on dictionary if we require it
            if self.require_dictionary and not dictionary_info:
                # Only skip if this is supposed to be a dictionary form
                if path and path[-1] == "dictionary":
                    continue
                # If this is an intermediate form in an auxiliary path, keep it for now
                # It will be validated when we reach the actual base verb
                if not self._is_auxiliary_path(path):
                    continue

            if dictionary_info:
                record["dictionary_entry"] = dictionary_info

            # Base score for verb tree
            if record.get("tree") == "verb":
                score += 1.5

            # Score based on path depth (prefer shorter paths)
            if path:
                score += max(0.0, 3.0 - 0.4 * depth)
                if path[-1] == "dictionary":
                    score += 2.5

            # Prefer longer suffix matches (more specific conjugation patterns)
            # e.g., "なかった" should score higher than "った"
            removed = record.get("removed", "")
            if removed:
                # Award points based on the length of the removed suffix
                suffix_score = min(len(removed) * 0.3, 3.0)
                score += suffix_score

            # Give bonus for matching well-known auxiliary verb patterns
            auxiliary_labels = {
                "ている", "てある", "ておく", "てくる", "ていく", "てみる",
                "てしまう", "てあげる", "てくれる", "てもらう"
            }
            if path and any(label in auxiliary_labels for label in path):
                score += 1.5

            # Score for valid verb type
            verb_type_value = record.get("verb_type")
            if isinstance(verb_type_value, str):
                if verb_type_value in self.VERB_TYPES:
                    score += 0.5
            elif isinstance(verb_type_value, (list, tuple, set)):
                if any(isinstance(v, str) and v in self.VERB_TYPES for v in verb_type_value):
                    score += 0.5

            # Prefer entries with dictionary validation (if available)
            if dictionary_info:
                if dictionary_info["match_type"] == "term":
                    score += 10.0
                else:
                    score += 7.0
            else:
                # Without dictionary, use heuristics
                base = record.get("base", "")

                # Check for auxiliary verb patterns that shouldn't be treated as dictionary forms
                auxiliary_patterns = [
                    "ている", "てある", "ておく", "てくる", "ていく", "てみる",
                    "てしまう", "てあげる", "てくれる", "てもらう", "てる",
                    "ちゃう", "ちまう", "でいる"
                ]
                has_auxiliary = any(base.endswith(pattern) for pattern in auxiliary_patterns)

                # Prefer forms that end with dictionary endings
                if base.endswith(self.VERB_DICTIONARY_ENDINGS) or base in self.SPECIAL_DICTIONARY_VERBS:
                    if has_auxiliary:
                        # Penalize forms with auxiliary patterns
                        score += 1.0
                    else:
                        score += 5.0

                # Prefer paths that reach "dictionary" node
                if path and path[-1] == "dictionary":
                    if has_auxiliary:
                        # Don't give bonus to auxiliary patterns
                        score += 0.5
                    else:
                        score += 3.0
                else:
                    score += 1.0

            confidence = max(0.0, min(score / 12.0, 1.0))
            record["score"] = score
            record["confidence"] = round(confidence, 4)
            scored.append(record)

        # Sort by raw score (not confidence), dictionary match, path length, and base form
        scored.sort(
            key=lambda item: (
                -item.get("score", 0.0),  # Use raw score for better differentiation
                0 if item.get("dictionary_entry") else 1,
                len(item.get("path") or []),
                item.get("base", ""),
            )
        )
        return scored

    def _iter_matches(self, word: str, tree_name: str, node: ConjugationNode, depth: int) -> List[Dict[str, Any]]:
        """Generate all possible matches for a word at a given node."""
        matches: List[Dict[str, Any]] = []
        candidate_types = self._candidate_types(tree_name)

        for key, (suffix_remove, suffix_add) in node.transforms.items():
            # Skip non-matching keys
            if key != "all" and key not in candidate_types:
                continue

            # Handle empty suffix_remove (appends)
            if not suffix_remove:
                if depth == 0:
                    continue
                if suffix_add and word.endswith(suffix_add):
                    continue

            # Check if word ends with the suffix to remove
            if suffix_remove and not word.endswith(suffix_remove):
                continue

            # Build the candidate
            if suffix_remove:
                stem = word[:-len(suffix_remove)]
            else:
                stem = word

            candidate = stem + suffix_add

            # Skip no-op transformations
            if candidate == word and not suffix_remove and not suffix_add:
                continue

            # Determine allowed verb types for this match
            # When node.verb_type is set, it restricts what the auxiliary can be
            # But we still want to allow "all" to match
            if key == "all":
                # For "all" key, we use the node's verb_type if specified
                if node.verb_type is None:
                    allowed_types: List[Optional[str]] = [None]
                elif isinstance(node.verb_type, (list, tuple, set)):
                    allowed_types = list(node.verb_type)
                else:
                    allowed_types = [node.verb_type]
            else:
                # For specific keys, use that key as the type
                allowed_types = [key]

            if not allowed_types:
                allowed_types = [None]

            for effective_type in allowed_types:
                # Skip if effective type doesn't match candidate types
                if isinstance(effective_type, str) and effective_type not in candidate_types and effective_type is not None:
                    continue

                matches.append(
                    {
                        "stem": stem,
                        "candidate": candidate,
                        "transform_key": key,
                        "verb_type": effective_type,
                        "removed": suffix_remove,
                        "added": suffix_add,
                    }
                )

        return matches

    def _evaluate_node(
        self,
        word: str,
        tree_name: str,
        node: ConjugationNode,
        path: List[str],
        depth: int,
        max_depth: int,
        visited: Dict[Tuple[str, str], int],
        results: List[Dict[str, Any]],
    ) -> None:
        """Evaluate a single node in the conjugation tree."""
        for match in self._iter_matches(word, tree_name, node, depth):
            stem = match["stem"]
            candidate = match["candidate"]
            transform_key = match["transform_key"]
            removed = match.get("removed", "")
            added = match.get("added", "")
            effective_type = match.get("verb_type")

            label = node.label or (transform_key if transform_key != "all" else node.node_type)
            verb_type = effective_type if effective_type is not None else (transform_key if transform_key != "all" else None)
            current_path = path + [label]
            recursions_used = len(current_path)

            record: Dict[str, Any] = {
                "base": candidate,
                "tree": self._normalize_tree_name(tree_name),
                "path": current_path,
                "verb_type": verb_type,
                "removed": removed,
                "added": added,
                "node_type": node.node_type,
                "stem": stem,
                "recursions": recursions_used,
            }

            # If node terminates, add to results
            if node.terminates:
                results.append(record)

                # Handle alternate forms
                if node.alternate_forms:
                    alternate_stem = candidate[:-len(added)] if added else candidate
                    for alt in node.alternate_forms:
                        alt_record = record.copy()
                        alt_record["base"] = alternate_stem + alt
                        alt_record["alternate"] = True
                        results.append(alt_record)

            # Stop if max depth reached
            if depth >= max_depth:
                continue

            # Process children
            if node.children:
                for child in node.children:
                    self._evaluate_node(candidate, tree_name, child, current_path, depth + 1, max_depth, visited, results)

            # Process conj_type transitions (e.g., て + verb auxiliary)
            if node.conj_type:
                # When we have conj_type, we're transitioning to a different tree
                # The candidate now becomes the input for that tree
                self._deconjugate_recursive(candidate, node.conj_type, current_path, depth + 1, max_depth, visited, results)

            # Continue recursion based on whether suffix was removed
            if removed:
                # If we removed a suffix, continue exploring from candidate
                self._deconjugate_recursive(candidate, tree_name, current_path, depth + 1, max_depth, visited, results)

                # Also explore from stem if different
                if stem != candidate:
                    self._deconjugate_recursive(stem, tree_name, current_path, depth + 1, max_depth, visited, results)

                    # Try other trees from stem
                    normalized_current = self._normalize_tree_name(tree_name)
                    for extra_tree in ("verb", "nai", "i"):
                        if normalized_current != extra_tree:
                            self._deconjugate_recursive(stem, extra_tree, current_path, depth + 1, max_depth, visited, results)
            elif not node.ends:
                # If no suffix removed and node doesn't end, continue from candidate
                self._deconjugate_recursive(candidate, tree_name, current_path, depth + 1, max_depth, visited, results)

    def _deconjugate_recursive(
        self,
        word: str,
        tree_name: str,
        path: List[str],
        depth: int,
        max_depth: int,
        visited: Dict[Tuple[str, str], int],
        results: List[Dict[str, Any]],
    ) -> None:
        """Recursively deconjugate a word through the tree."""
        if depth > max_depth:
            return

        normalized_name = self._normalize_tree_name(tree_name)
        visited_key = (word, normalized_name)
        previous_depth = visited.get(visited_key)
        if previous_depth is not None and depth >= previous_depth:
            return
        visited[visited_key] = depth

        # Check if this looks like a dictionary form
        if self._looks_like_dictionary(word, normalized_name):
            results.append(
                {
                    "base": word,
                    "tree": normalized_name,
                    "path": path + ["dictionary"],
                    "node_type": "dictionary",
                }
            )

        # Explore tree
        tree = self._get_tree(normalized_name)
        for child in tree.children:
            self._evaluate_node(word, normalized_name, child, path, depth, max_depth, visited, results)

    def _unique_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge results with the same base form, collecting all paths."""
        # Group by base form
        by_base: Dict[str, List[Dict[str, Any]]] = {}
        for record in results:
            base = record.get("base", "")
            if base not in by_base:
                by_base[base] = []
            by_base[base].append(record)

        # For each base, merge all paths
        unique: List[Dict[str, Any]] = []
        for base, records in by_base.items():
            # Sort by path length to get shortest first
            records.sort(key=lambda r: len(r.get("path", [])))

            # Take the first record as the template
            merged = records[0].copy()

            # Collect all unique paths
            all_paths = []
            seen_paths: Set[Tuple[str, ...]] = set()
            for record in records:
                path_tuple = tuple(record.get("path", []))
                if path_tuple not in seen_paths:
                    seen_paths.add(path_tuple)
                    all_paths.append(record.get("path", []))

            # Store all paths in the merged record
            merged["all_paths"] = all_paths
            merged["path_count"] = len(all_paths)

            unique.append(merged)

        return unique

    def deconjugate(
        self,
        word: str,
        word_type: str = "verb",
        max_depth: int = 6,
        return_all: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Deconjugate a Japanese word to its dictionary form(s).

        Args:
            word: The conjugated word to deconjugate
            word_type: The type of word ("verb", "adjective", "i", "nai")
            max_depth: Maximum recursion depth for conjugation chains
            return_all: If True, return all valid results; if False, return only the best

        Returns:
            List of deconjugation results with dictionary validation and confidence scores
        """
        visited: Dict[Tuple[str, str], int] = {}
        results: List[Dict[str, Any]] = []
        normalized = self._normalize_tree_name(word_type)
        self._deconjugate_recursive(word, normalized, [], 0, max_depth, visited, results)

        unique_results = self._unique_results(results)
        scored_results = self._score_results(unique_results)

        if return_all:
            return scored_results
        else:
            return scored_results[:1] if scored_results else []


# Example usage
if __name__ == "__main__":
    deconjugator = JapaneseDeconjugator()

    print("Japanese Deconjugator Test\n" + "=" * 50)

    for word in TEST_WORDS:
        results = deconjugator.deconjugate(word, "verb")
        print(f"\nInput: {word}")
        if results:
            # Show top 5 results
            for index, entry in enumerate(results[:5], 1):
                print(f"  [{index}] Base: {entry['base']}")
                print(f"       Tree: {entry.get('tree')}")
                print(f"       Verb type: {entry.get('verb_type') or 'all'}")
                print(f"       Recursions: {entry.get('recursions', len(entry.get('path', [])))}")
                print(f"       Path: {' → '.join(entry.get('path', [])) if entry.get('path') else '(no path)'}")
                print(f"       Confidence: {entry.get('confidence', 0.0):.2f}")
                dictionary_info = entry.get("dictionary_entry")
                if dictionary_info:
                    extra = f"match={dictionary_info['match_type']}"
                    if dictionary_info.get("alt_form"):
                        extra += f", alt_form={dictionary_info['alt_form']}"
                    print(
                        f"       Dictionary: "
                        f"term={dictionary_info['term']} (id={dictionary_info['entry_id']}, {extra})"
                    )
                else:
                    print("       Dictionary: no direct match found")

            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more results")
        else:
            print("  Could not deconjugate (no dictionary-backed results)")
