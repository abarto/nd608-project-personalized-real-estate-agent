## nd608 - Project: Personalized Real Estate Agent

### Introduction

This is repo contains my implementation for Udacity's Generative AI Nanodregree "Project: Personalized Real Estate Agent.

The project uses [`poetry`](https://python-poetry.org/) for dependency management. Use `poetry install --no-root` on your environment to install the required python packages.

The code uses [`python-dotenv`](https://pypi.org/project/python-dotenv/) to load an `.env` file (not provided) from the same directory to set the `OPENAI_API_KEY` value environment variable. If this library is not available within Udacity's workspace, the client's `api_key` parameter will have to be set manually.

### Contents

| File/Folder | Description |
|-------------|-------------|
| [`Udacity-Project-Environment`](Udacity-Project-Environment) | The empty project environment provided by Udacity. |
| [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb) | Jupyter Notebook where we generate the synthetic real estate listings as well as setting up the [LanceDb](https://lancedb.com/) vector database with the listings' embeddings. |
| [`Generate-Real-Estate-Listings.html`](Generate-Real-Estate-Listings.html) | Static HTML version of [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb). |
| [`Demo-HomeMatch.ipynb`](Demo-HomeMatch.ipynb) | Jupyter Notebook where we set-up the environment and invoke the *HomeMatch* [Gradio](https://www.gradio.app/) app |
| [`Demo-HomeMatch.html`](Demo-HomeMatch.html) | Static HTML version of [`Demo-HomeMatch.ipynb`](Demo-HomeMatch.ipynb)|
| [`app.py`](app.py) | Python module with the *HomeMatch* [Gradio](https://www.gradio.app/) app code. |
| [`db.py`](db.py) | Python module with code related to database functions. |
| [`models.py`](models.py) | Python module with models shared between [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb) and the `app.py` module. |
| [`data`](data) | Folder containing the [LanceDb](https://lancedb.com/) data files as well as `pickle` files used as intermediate storage during the real estate listings generation. |
| [`images`](images) | Folder containing screenshots of the running *HomeMatch* app. |
| [`pyproject.toml`](pyproject.toml) | [`poetry`](https://python-poetry.org/) project specification. |
| [`poetry.lock`](poetry.lock) | [`poetry`](https://python-poetry.org/) dependency locks. |

### Real Estate Listings Generation

We've used [OpenAI](https://openai.com/)'s `gpt-4-turbo` (through [LangChain](https://www.langchain.com/) abstractions) to generate [Pydantic](https://docs.pydantic.dev/latest/) objects with specific features of the properties (number of rooms, number of bathrooms, etc.) as well as a textual description. We've specifically asked the model to generate matter-of-fact text so we can spruce it app when presenting the listing through the *HomeMatch* app.

We then took those descriptions and fed them to [OpenAI](https://openai.com/)'s `dall-e-2` to
generate realistic images of the listings.

We then paired the text and the image to generate [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) embeddings which we stored on a [LanceDB](https://lancedb.com/) table with this schema:

```python
class RealEstateListingLanceRecord(LanceModel):
    neighborhood: str
    price: int
    bedrooms: int
    bathrooms: int | float
    description: str
    neighborhood_description: str
    image_bytes: bytes
    vector: Vector(768)

    @property
    def image_as_pil(self):
        return Image.open(BytesIO(self.image_bytes))
```

This table is going to be used to perform semantic searches on the *HomeMatch* app.

Please check the [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb) Jupyter Notebook for specific details about the implementation.

### The *HomeMatch* app

We've used [Gradio](https://www.gradio.app/) to build the *HomeMatch* app, to make the project consistent to other offerings presented during the lessons.

The app takes a textual description of the user's needs, and generates embeddings out of them to perform a semantic search on listings vector database. It then applies additional filters based on the user's specific inputs to get to the final recommendation.

It then takes the listing description and passes it to a [OpenAI](https://openai.com/)'s completion model prompted to recreate the description using a more enticing and colorful language while trying to draw attention to features that match the user supplied needs.

The output of the model, as well as the stored image is then shown to the user.

You can run the app opening [`Demo-HomeMatch.ipynb`](Demo-HomeMatch.ipynb) with Jupyter, or from the console with: 

```
python -m app
```

### Screenshots

![First Screenshot](/images/Screenshot_1.png)
![Second Screenshot](/images/Screenshot_2.png)
![Third Screenshot](/images/Screenshot_3.png)