from tools import Strand
import tensorflow as tf
from typing import Literal
from pydantic import BaseModel

class Nucleotide(BaseModel):
    type: Literal["A", "T", "G", "U"]

class Sequence(BaseModel):
    nucleotides: list[Nucleotide]

class Coordinate(BaseModel):
    x: float
    y: float
    z: float
