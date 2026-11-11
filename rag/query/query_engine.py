from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
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
        nodes = retriever.retrieve(query_text)
        min_score = getattr(self.config, "min_relevance_score", None)
        if min_score is not None:
            nodes = [node for node in nodes if node.score is not None and node.score >= min_score]
        return nodes

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
