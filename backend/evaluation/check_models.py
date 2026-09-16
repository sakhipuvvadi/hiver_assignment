import os
from dotenv import load_dotenv
from groq import Groq

# Load backend/.env
load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env"
    )
)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found!")
    exit()

client = Groq(api_key=api_key)

models = client.models.list()

print("\nAVAILABLE GROQ MODELS:\n")
print("=" * 60)

for model in models.data:
    print(model.id)

print("=" * 60)
print(f"\nTotal models: {len(models.data)}")