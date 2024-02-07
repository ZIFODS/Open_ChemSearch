import pandera as pa
from pandera.typing import Series


class MoleculesSchema(pa.SchemaModel):
    SMILES: Series[str]
    Molecule: Series[pa.Object]
