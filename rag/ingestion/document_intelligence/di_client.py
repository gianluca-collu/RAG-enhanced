from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential


def create_document_intelligence_client(settings) -> DocumentIntelligenceClient:
    """Crea il client Azure Document Intelligence dalle impostazioni.

    Usa API version 2024-11-30 (GA) che supporta nativamente:
    PDF, DOCX, XLSX, PPTX, HTML, immagini (JPG, PNG, BMP, TIFF, HEIF).
    """
    return DocumentIntelligenceClient(
        endpoint=settings.document_intelligence_endpoint,
        credential=AzureKeyCredential(settings.document_intelligence_key),
        api_version="2024-11-30",
    )

