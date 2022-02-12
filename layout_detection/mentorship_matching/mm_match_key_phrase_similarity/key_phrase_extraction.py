import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv()

ta_key = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
ta_endpoint = os.getenv('AZURE_TEXT_ANALYTICS_URL')

def authenticate_ta_client():
    """
    
    Authenticating the Azure Text Analytics Client

    Returns
    -------
    text_analytics_client : azure.ai.textanalytics.TextAnalyticsClient
        Authenticated Azure Text Analytics Client.

    """
    ta_credential = AzureKeyCredential(ta_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=ta_endpoint, 
            credential=ta_credential)
    return text_analytics_client

def extract_key_phrases(ta_client, documents : list):
    """
    
    Parameters
    ----------
    ta_client : azure.ai.textanalytics.TextAnalyticsClient
        Azure Text Analytics Client.
    documents : list
        List of text documents from which key phrases are to be extracted.

    Returns
    -------
    list
        List of key phrases.

    """
    try:
        response = ta_client.extract_key_phrases(documents = documents)[0]

        if not response.is_error:
            return response.key_phrases
        else:
            print(response.id, response.error)
            return []

    except Exception as err:
        print("Encountered exception in keyphrase extraction . {}".format(err))
        return []