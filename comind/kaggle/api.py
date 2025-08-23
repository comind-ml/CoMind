from kaggle.api.kaggle_api_extended import KaggleApi

api: KaggleApi = None

def authenticate_kaggle_api() -> "KaggleApi": # type: ignore
    """Authenticates the Kaggle API and returns an authenticated API object, or raises an error if authentication fails."""
    global api

    if api is not None:
        return api
    
    try:
        api = KaggleApi()
        api.authenticate()
        api.competitions_list()  # a cheap op that requires authentication
        return api
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        raise PermissionError(
            "Kaggle authentication failed! Please ensure you have valid Kaggle API credentials "
            "configured. Refer to the Kaggle API documentation for guidance on setting up "
            "your API token."
        ) from e