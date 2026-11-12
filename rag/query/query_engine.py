from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from rag.config import Config
from rag.logger import Logger
from rag.query.prompts import RAG_SYSTEM_PROMPT
from rag.query.retrieval_export import export_retrieval


@dataclass
class _QueryState:
    query_text: str
    top_k: int
    system_prompt: Optional[str]
    source_nodes: list = field(default_factory=list)
    sources: list[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    answer: str = ""


class QueryEngine:
    """Retrieve relevant context and generate an answer for a user query."""

    def __init__(self, config: Config, index: Any, llm: Any, logger: Optional[Logger] = None):
        self.config = config
        self.index = index
        self.llm = llm
        self.logger = logger or Logger("QueryEngine")

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.run(
            query_text,
            top_k=top_k,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
        )

    def run(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the query pipeline, passing its state through each step."""
        try:
            retrieval_top_k = similarity_top_k or top_k or self.config.search_top_k
            state = _QueryState(query_text, retrieval_top_k, system_prompt)

            # Step 1: validate the incoming question.
            state = self._step_1_validate(state)
            # Step 2: retrieve nodes using the validated question.
            state = self._step_2_retrieve(state)
            # Step 3: format the retrieved nodes as sources.
            state = self._step_3_format_sources(state)
            # Step 4: build the context from those sources.
            state = self._step_4_build_context(state)
            # Step 5: generate an answer from the question and context.
            state = self._step_5_generate_answer(state)
            # Step 6: return the final response.
            return self._step_6_build_result(state)
        except Exception as exc:
            self.logger.error(f"Query failed: {str(exc)}")
            raise

    def _step_1_validate(self, state: _QueryState) -> _QueryState:
        state.query_text = self._validate_query(state.query_text)
        return state

    def _step_2_retrieve(self, state: _QueryState) -> _QueryState:
        state.source_nodes = self._retrieve_sources(state.query_text, state.top_k)
        return state

    def _step_3_format_sources(self, state: _QueryState) -> _QueryState:
        state.sources = self._format_sources(state.source_nodes)
        export_retrieval(state.query_text, state.sources)
        return state

    def _step_4_build_context(self, state: _QueryState) -> _QueryState:
        state.context = self._build_context(state.sources)
        return state

    def _step_5_generate_answer(self, state: _QueryState) -> _QueryState:
        state.answer = self._generate_answer(
            state.query_text,
            state.context,
            state.system_prompt,
        )
        return state

    @staticmethod
    def _step_6_build_result(state: _QueryState) -> Dict[str, Any]:
        return {"answer": state.answer, "sources": state.sources, "query": state.query_text}

    @staticmethod
    def _validate_query(query_text: str) -> str:
        if not query_text or not query_text.strip():
            raise ValueError("Query cannot be empty")
        return query_text.strip()

    def _retrieve_sources(self, query_text: str, top_k: int) -> list:
        query_mode = (
            VectorStoreQueryMode.HYBRID
            if getattr(self.config, "hybrid_search", False)
            else VectorStoreQueryMode.DEFAULT
        )
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode=query_mode,
        )
        min_score = getattr(self.config, "min_relevance_score", None)
        translated_query = self._translate_query_to_other_language(query_text)
        retrieval_queries = [query_text]
        if translated_query.casefold() != query_text.casefold():
            retrieval_queries.append(translated_query)

        result_sets = []
        for retrieval_query in retrieval_queries:
            nodes = retriever.retrieve(retrieval_query)
            if min_score is not None:
                nodes = [node for node in nodes if node.score is not None and node.score >= min_score]
            result_sets.append(nodes)

        if len(result_sets) == 1:
            return result_sets[0][:top_k]
        return self._fuse_results(result_sets, top_k)

    def _translate_query_to_other_language(self, query_text: str) -> str:
        try:
            response = self.llm.chat([
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "The search query may be in Italian or English. Translate it into the other "
                        "language: Italian to English, or English to Italian. If it is in another "
                        "language, translate it into English. Return only the translated query, "
                        "preserving names, numbers, acronyms, and technical terms. Do not answer it."
                    ),
                ),
                ChatMessage(role=MessageRole.USER, content=query_text),
            ])
        except Exception as exc:
            self.logger.error(f"Query translation failed; using original query: {exc}")
            return query_text

        translated_query = response.message.content
        if not isinstance(translated_query, str) or not translated_query.strip():
            return query_text
        return translated_query.strip().strip('"').strip()

    @staticmethod
    def _fuse_results(result_sets: list[list], top_k: int, rrf_constant: int = 60) -> list:
        fused_scores: dict[str, float] = {}
        fused_nodes: dict[str, Any] = {}

        for nodes in result_sets:
            for rank, result in enumerate(nodes, start=1):
                node = result.node
                node_id = getattr(node, "node_id", None) or str(id(node))
                fused_nodes.setdefault(node_id, node)
                fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1 / (rrf_constant + rank)

        ordered = sorted(
            fused_nodes.items(),
            key=lambda item: fused_scores[item[0]],
            reverse=True,
        )[:top_k]
        return [
            NodeWithScore(node=node, score=fused_scores[node_id])
            for node_id, node in ordered
        ]

    @staticmethod
    def _format_sources(source_nodes: list) -> list[Dict[str, Any]]:
        return [
            {
                "content": item.node.get_content(),
                "metadata": item.node.metadata,
                "score": item.score,
            }
            for item in source_nodes
        ]

    @staticmethod
    def _build_context(sources: list[Dict[str, Any]]) -> str:
        formatted_sources = []
        for position, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            document = (
                metadata.get("document_title")
                or metadata.get("source_file")
                or metadata.get("source")
                or "Documento sconosciuto"
            )
            section = metadata.get("section_path")
            page = metadata.get("page")
            citation = f"[Fonte {position}; documento: {document}"
            if section:
                citation += f"; sezione: {section}"
            if page is not None:
                citation += f"; pagina: {page}"
            formatted_sources.append(f"{citation}]\n{source['content']}")
        context = "\n\n".join(formatted_sources)
        return context or "Nessuna fonte pertinente è stata trovata."

    def _generate_answer(
        self,
        query_text: str,
        context: str,
        system_prompt: Optional[str],
    ) -> str:
        response = self.llm.chat([
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt or RAG_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Fonti recuperate:\n{context}\n\nDomanda:\n{query_text}",
            ),
        ])
        return response.message.content or ""
