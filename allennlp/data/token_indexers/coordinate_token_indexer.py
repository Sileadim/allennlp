from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

import numpy as np
from allennlp.common.checks import ConfigurationError


@TokenIndexer.register("raw_coordinates")
class CoordinateTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as their raw coordinates.

    # Parameters

    namespace : `str`, optional (default=`tokens`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : `bool`, optional (default=`False`)
        If `True`, we will call `token.lower()` before getting an index for the token from the
        vocabulary.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    feature_name : `str`, optional (default=`text`)
        We will use the :class:`Token` attribute with this name as input
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
            self,
            namespace: str = "tokens",
            lowercase_tokens: bool = False,
            start_tokens: List[str] = None,
            end_tokens: List[str] = None,
            feature_name: str = "coordinates",
            token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self.feature_name = feature_name

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(
            self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        coords_list: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            coords = getattr(token, self.feature_name)
            if coords is None:
                coords = np.zeros(4)
            coords_list.append(coords)

        return {"token_coordinates": coords_list}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"token_coordinates": []}


@TokenIndexer.register("coordinates_bin")
class CoordinateBinTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as their raw coordinates.

    # Parameters

    namespace : `str`, optional (default=`tokens`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : `bool`, optional (default=`False`)
        If `True`, we will call `token.lower()` before getting an index for the token from the
        vocabulary.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    feature_name : `str`, optional (default=`text`)
        We will use the :class:`Token` attribute with this name as input
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
            self,
            namespace: str = "tokens",
            lowercase_tokens: bool = False,
            start_tokens: List[str] = None,
            end_tokens: List[str] = None,
            feature_name: str = "coordinates",
            nbins: int = 100,
            coordinate_index=0,
            token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self.feature_name = feature_name
        self.nbins = nbins
        self.coordinate_index = coordinate_index

    def get_coordinate_bin(self, token):
        coords = getattr(token, self.feature_name)
        if coords is None:
            coords = np.zeros(4)
        bins = coords * self.nbins
        bins = bins.astype(np.int64)
        bins.clip(0, self.nbins - 1, out=bins)
        # need to add +1, because 0 is for padding token which get masked out in later stages
        return bins[self.coordinate_index] + 1

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # pass
        for i in range(self.nbins):
            counter[self.namespace][str(i)] += 1

    @overrides
    def tokens_to_indices(
            self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        bins_list: List[int] = []
        # we are hacking a little bit around here, the idx is not generated by the vocabulary but by the
        # coordinate lookup
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            bin = self.get_coordinate_bin(token)
            bins_list.append(bin)

        return {"token_bins": bins_list}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"token_bins": []}
