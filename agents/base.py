# agents/base.py


class ImmuneCell:
    """면역 세포 기본 클래스"""

    def __init__(self, cell_id, activation_threshold=0.5):
        self.cell_id = cell_id
        self.activation_threshold = activation_threshold
        self.activation_level = 0.0
        self.memory_strength = 0.0
