"""
This module contains models that are shared between the
listing generation notebook and the user facing app.
"""

from io import BytesIO

from lancedb.pydantic import LanceModel, Vector
from PIL import Image


class RealEstateListingLanceRecord(LanceModel):
    neighborhood: str
    price: int
    bedrooms: int
    bathrooms: int | float
    has_solar_panels: bool
    description: str
    neighborhood_description: str
    image_bytes: bytes
    vector: Vector(768)

    @property
    def image_as_pil(self):
        return Image.open(BytesIO(self.image_bytes))
