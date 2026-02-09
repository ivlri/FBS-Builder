from dataclasses import dataclass

GRID_STEP = 20

@dataclass(frozen=True)
class BlockType:
    id: int
    length: int
    height: int
    name: str

    def num_cells(self, grid_step: int) -> int:
        return self.length // grid_step

    def num_rows(self, row_height: int = 300) -> int:
        return self.height // row_height


@dataclass(frozen=True)
class WallInstance:
    id: int
    length: int
    height: int
    weight: int
    grid_step: int

    @property
    def num_cells(self) -> int:
        return self.length // self.grid_step

    @property
    def num_rows(self) -> int:
        return self.height // 300

    @property
    def num_layers(self) -> int:
        return self.height // 600


@dataclass(frozen=True)
class Opening:
    center_x: int   # center X from wall start (mm)
    center_y: int   # center Y from floor (mm)
    width: int       # width (mm), max 600
    height: int      # height (mm), max 600
