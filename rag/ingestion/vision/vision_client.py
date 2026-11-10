from openai import AzureOpenAI


def create_vision_client(settings) -> AzureOpenAI:
    """Crea il client Azure OpenAI usato per la descrizione delle immagini."""
    return AzureOpenAI(
        api_key=settings.openai_key,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.openai_url,
    )

