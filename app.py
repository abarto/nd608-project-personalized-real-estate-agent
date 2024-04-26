"""
This module contains HomeMatch app implementation.
"""

import logging
from functools import partial
from textwrap import dedent

import gradio as gr
import lancedb
import openai
from langchain.prompts import PromptTemplate
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, PreTrainedTokenizerBase

from db import get_db, get_listings_table
from models import RealEstateListingLanceRecord

logger = logging.getLogger(__name__)


system_message = {
    "role": "system",
    "content": "You are a writer and a real estate agent capable to rewriting property descriptions using a colorful and attention seeking language.",
}


prompt = PromptTemplate(
    template=dedent(
        """\
        Based on the following property description:
                    
        {description}

        and property neighborhood description:
                    
        {neighborhood_description}
                    
        and the following customer preferences:
                    
        {preferences}

        Generate a new description of which puts emphasis on the features
        of the property and neighborhood that match the customer preferences.
        The new description needs to be easy to read, fun, and creative but
        it cannot add things that are not present in the old descriptions.
                    
        Return the new description in Markdown format.
    """
    ),
    input_variables=["description", "preferences"],
)


def search_listing(
    table: lancedb.table.Table,
    model: CLIPModel,
    tokenizer: PreTrainedTokenizerBase,
    preferences: str,
    number_of_bedrooms: int,
    number_of_bathrooms: int | float,
    has_solar_panels: bool,
    price_range: int,
) -> RealEstateListingLanceRecord:
    logger.info(
        "preferences: %s, number_of_bedrooms: %s, number_of_bathrooms: %s, "
        "has_solar_panels: %s, price_range: %s",
        preferences,
        number_of_bedrooms,
        number_of_bathrooms,
        has_solar_panels,
        price_range,
    )

    inputs = tokenizer(preferences, padding=True, truncation=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)[0].cpu().detach().numpy()

    query = table.search(text_features)

    if number_of_bedrooms > 0:
        query = query.where(f"bedrooms >= {number_of_bedrooms}")

    if number_of_bathrooms > 0:
        query = query.where(f"bathrooms >= {number_of_bedrooms}")

    if has_solar_panels:
        query = query.where(f"has_solar_panels IS TRUE")

    if price_range == 1:
        query = query.where(f"price < 300000")
    elif price_range == 2:
        query = query.where(f"30000 <= price AND price < 400000")
    elif price_range == 3:
        query = query.where(f"40000 <= price AND price < 500000")
    elif price_range == 4:
        query = query.where(f"50000 <= price AND price < 600000")
    elif price_range == 5:
        query = query.where(f"60000 <= price")

    listings = query.limit(1).to_pydantic(RealEstateListingLanceRecord)

    return listings[0] if listings else None


def get_recommendation(
    table: lancedb.table.Table,
    model: CLIPModel,
    tokenizer: PreTrainedTokenizerBase,
    openai_client: openai.Client,
    preferences: str,
    number_of_bedrooms: int,
    number_of_bathrooms: int | float,
    has_solar_panels: bool,
    price_range: int,
) -> tuple[Image.Image, str]:
    logger.info(
        "preferences: %s, number_of_bedrooms: %s, number_of_bathrooms: %s "
        "has_solar_panels: %s, price_range: %s",
        preferences,
        number_of_bedrooms,
        number_of_bathrooms,
        has_solar_panels,
        price_range,
    )

    listing = search_listing(
        table,
        model,
        tokenizer,
        preferences,
        number_of_bedrooms,
        number_of_bathrooms,
        has_solar_panels,
        price_range,
    )

    if listing is None:
        return (None, "### No properties matched your criteria")

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            system_message,
            {
                "role": "user",
                "content": prompt.format(
                    description=listing.description,
                    neighborhood_description=listing.neighborhood_description,
                    preferences=preferences,
                ),
            },
        ],
        temperature=0.4,
    )

    logger.info("response: %s", response)

    return (listing.image_as_pil, response.choices[0].message.content)


def create_app() -> gr.Blocks:
    db = get_db()
    table = get_listings_table(db)
    openai_client = openai.Client()

    clip_model = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)

    with gr.Blocks() as app:
        gr.Markdown("# HomeMatch")
        with gr.Row():
            preferences = gr.TextArea(
                label="What are you looking for?",
                placeholder=(
                    "How big do you want your house to be?\n"
                    "What are 3 most important things for you in choosing this property?\n"
                    "Which amenities would you like?\n"
                    "Which transportation options are important to you?\n"
                    "How urban do you want your neighborhood to be?"
                ),
            )
            with gr.Column():
                number_of_bedrooms = gr.Number(label="Number of bedrooms")
                number_of_bathrooms = gr.Number(label="Number of bathrooms")
                has_solar_panels = gr.Checkbox(label="Has solar panels fitted")
                price_range = gr.Dropdown(
                    label="Price Range",
                    choices=[
                        ("Less than $200,000", 1),
                        ("Between than $200,000 and $300,000", 2),
                        ("Between than $300,000 and $400,000", 3),
                        ("Between than $400,000 and $500,000", 4),
                        ("More than $500,000", 5),
                    ],
                )
        with gr.Row():
            image = gr.Image(
                label="Image of the property", width=512, height=512, scale=1
            )
        with gr.Row():
            desciption = gr.Markdown("### Click Submit to get a recommendation.")
        with gr.Row():
            submit_btn = gr.Button("Submit")
            submit_btn.click(
                partial(get_recommendation, table, model, tokenizer, openai_client),
                inputs=[
                    preferences,
                    number_of_bathrooms,
                    number_of_bedrooms,
                    has_solar_panels,
                    price_range,
                ],
                outputs=[image, desciption],
            )

    return app


if __name__ == "__main__":
    # Load environment variables from a .env file. Alternatively you can
    # manually set the value of OPENAI_API_KEY on this cell.
    import logging
    from os import environ

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    if "OPENAI_API_KEY" not in environ:
        environ["OPENAI_API_KEY"] = "your-openai-api-key"

    environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(level=logging.INFO)

    home_match_app = create_app()
    home_match_app.launch(share=False)
