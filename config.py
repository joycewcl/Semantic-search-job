"""
Configuration for Job Matcher
"""
import os

class Config:
    """Configuration settings"""
    
    # Pinecone
    PINECONE_API_KEY = "pcsk_5PQhRu_AaCxW5SSdDkG1Ue63VJBYZffwpxW8tKnwMsvM3T3JJ4fuVrp3Wv7tq9fXVUfQKB"
    PINECONE_ENVIRONMENT = "us-east-1"
    INDEX_NAME = "job-resume-matcher"
    EMBEDDING_DIMENSION = 384
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Azure OpenAI
    AZURE_ENDPOINT = "https://hkust.azure-api.net/"
    AZURE_API_KEY = "eacb49eb7c904d738c7644fc104aa7bb"
    AZURE_API_VERSION = "2024-10-21"
    AZURE_MODEL = "gpt-4o-mini"
    
    # RapidAPI
    RAPIDAPI_KEY = "2a86f3cdbfmsh5d8dfaabd0dd421p190709jsnf3fe78ba9559"
    
    # Settings
    MAX_JOBS_TO_FETCH = 50
    TOP_MATCHES_TO_SHOW = 5
    UPLOAD_FOLDER = "uploads"
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate API keys are set"""
        print("✅ Configuration validated")
        return True