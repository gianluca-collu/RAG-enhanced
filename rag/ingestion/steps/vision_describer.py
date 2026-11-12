import logging
import base64
import mimetypes
from typing import Optional, Tuple

from openai import AzureOpenAI


logger = logging.getLogger(__name__)

VISION_PROMPT = """Stai documentando una procedura aziendale per una Knowledge Base.
Descrivi SOLO l'informazione operativa di questa schermata, NON il suo aspetto grafico.

Rispondi in modo concreto a:
- Quale funzione o passo della procedura rappresenta.
- Quali campi, pulsanti o azioni sono rilevanti per l'utente.
- Quali dati, valori o etichette specifiche sono leggibili (riportali testualmente).

Ignora loghi, barre, colori, intestazioni ripetute e impaginazione.
Non inventare ciò che non è leggibile.
Massimo 5 righe, testo autonomo e ricercabile."""


CLASSIFICATION_PROMPT = """
Classifica questa visuale per una Knowledge Base RAG.

Rispondi solo:

DIAGRAM se contiene un processo, schema, flowchart, architettura,
ad esempio blocchi rettangolari relazionati tra loro da frecce

IMAGE se contiene una schermata applicativa, un'interfaccia,
un grafico, un modulo o informazioni utili.

DECORATIVE se contiene solamente logo, copertina, intestazione,
piè di pagina, firma, icona, elemento grafico ripetuto
o decorazione senza contenuto informativo.
"""


class VisionDescriber:
    """Classifica e descrive immagini usando Azure OpenAI Vision."""

    def __init__(self, vision_client: Optional[AzureOpenAI], settings) -> None:
        """
        Inizializza il servizio Vision.

        vision_client è il client Azure OpenAI.
        settings contiene deployment e flag di descrizione.
        Se Vision non è configurato, nessuna visuale viene descritta.
        """
        self.vision = vision_client
        self.settings = settings

    def describe(
        self,
        image_name: str,
        image_bytes: bytes,
        caption: str = "",
    ) -> Optional[Tuple[str, str]]:
        """
        Classifica e descrive una visuale.

        Prima capisce se la visuale è IMAGE o DIAGRAM.
        Poi verifica se quel tipo va descritto in base ai settings.
        Se la descrizione è valida, restituisce (tipo, descrizione).
        Altrimenti restituisce None.
        """
        if not image_bytes:
            return None

        kind = self._classify(image_name, image_bytes)
        logger.debug(f"{image_name}: classificata come {kind}")

        if kind == "DECORATIVE":
            logger.debug(f"{image_name}: scartata perché DECORATIVE")
            return None

        if not self._should_describe(kind):
            logger.warning(f"{image_name}: descrizione disabilitata per {kind}")
            return None

        description = self._describe(image_name, image_bytes, caption)

        if not description:
            return None

        return kind, description

    def _image_messages(
        self,
        prompt: str,
        image_name: str,
        image_bytes: bytes,
    ) -> list:
        """
        Crea il payload multimodale per Azure OpenAI Vision.

        Converte i bytes dell'immagine in base64.
        Crea una data URL compatibile con il modello Vision.
        Restituisce il messaggio pronto per chat.completions.create().
        """
        mime = mimetypes.guess_type(image_name)[0] or "image/png"
        encoded = base64.b64encode(image_bytes).decode("ascii")

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ]

    def _classify(self, image_name: str, image_bytes: bytes) -> str:
        """Classifica la visuale come IMAGE o DIAGRAM."""

        if self.vision is None:
            logger.warning(f"{image_name}: client Vision assente")
            return "IMAGE"

        try:
            response = self.vision.chat.completions.create(
                model=self._vision_deployment(),
                messages=self._image_messages(
                    prompt=CLASSIFICATION_PROMPT,
                    image_name=image_name,
                    image_bytes=image_bytes,
                ),
                max_completion_tokens=20,
            )

            answer = (response.choices[0].message.content or "").strip().upper()

            if "DECORATIVE" in answer:
                return "DECORATIVE"

            if "DIAGRAM" in answer:
                return "DIAGRAM"

            return "IMAGE"

        except Exception as error:
            logger.error(
                f"{image_name}: ERRORE CLASSIFICAZIONE "
                f"{type(error).__name__}: {error}"
            )
            return "IMAGE"

    def _describe(
        self,
        image_name: str,
        image_bytes: bytes,
        caption: str = "",
    ) -> str:
        """Genera la descrizione della visuale."""

        if self.vision is None:
            logger.warning(f"{image_name}: client Vision assente")
            return ""

        prompt = VISION_PROMPT

        if caption:
            prompt += f"\n\nCaption disponibile: {caption}"

        try:
            response = self.vision.chat.completions.create(
                model=self._vision_deployment(),
                messages=self._image_messages(
                    prompt=prompt,
                    image_name=image_name,
                    image_bytes=image_bytes,
                ),
                max_completion_tokens=600,
            )

            description = (
                response.choices[0].message.content or ""
            ).strip()

            logger.debug(
                f"{image_name}: descrizione ricevuta, "
                f"{len(description)} caratteri"
            )

            return description

        except Exception as error:
            logger.error(
                f"{image_name}: ERRORE DESCRIZIONE "
                f"{type(error).__name__}: {error}"
            )
            return ""

    def _should_describe(self, kind: str) -> bool:
        """Decide cosa descrivere in base alla configurazione."""

        if kind == "IMAGE":
            return getattr(self.settings, "describe_images", True)

        if kind == "DIAGRAM":
            return getattr(self.settings, "describe_diagrams", True)

        return False

    def _vision_deployment(self) -> str:
        """Restituisce il nome del deployment Vision."""

        deployment = (
            getattr(self.settings, "vision_deployment", None)
            or getattr(self.settings, "openai_deployment_name", None)
        )

        if not deployment:
            raise ValueError(
                "settings.vision_deployment non configurato"
            )

        return deployment

