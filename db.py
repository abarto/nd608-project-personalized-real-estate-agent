"""
This module contains functions related to database access.
"""

from pathlib import Path

import lancedb

from models import RealEstateListingLanceRecord

REAL_ESTATE_LISTINGS_TABLE_NAME = "real_estate_listings"


def get_db(uri: lancedb.URI = Path("data") / "lancedb") -> lancedb.DBConnection:
    """Get LanceDB connection"""
    return lancedb.connect(uri)


def init_db(db: lancedb.DBConnection):
    """Initializes the database re-creating the listings table"""
    db.drop_table(REAL_ESTATE_LISTINGS_TABLE_NAME, ignore_missing=True)
    db.create_table(
        REAL_ESTATE_LISTINGS_TABLE_NAME, schema=RealEstateListingLanceRecord
    )


def get_listings_table(db: lancedb.DBConnection) -> lancedb.table.Table:
    """Get the real estate listings table"""
    return db.open_table(REAL_ESTATE_LISTINGS_TABLE_NAME)
